"""JET-Forest, BLINK, FlashEntropySearch 与 matchms 统一多引擎与多后端基准评测运行框架。

涵盖：
1. 引擎与后端可用性自动化动态探测 (detect_available_engines: jetf-gpu, jetf-cpu-mt, jetf-cpu-1t, blink, flashentropy, matchms)
2. 离线索引构建、快照加载与 GPU 显存驻留统计 (Index & VRAM Resident Breakdown)
3. 统一批处理吞吐量与单查询分位数时延压测 (Batch Throughput QPS & P50/P95/P99 Latency under Batch Size)
4. 多后端与跨引擎加速比矩阵 (Speedup vs 1T, vs MT, vs matchms)
5. 异构流水线剪枝效能拆解 (K1 Root, K2 Leaf, K3a U_ind Pruning Breakdown)
6. 逐位保真与数学一致性校验 (Recall@K vs matchms, Zero False Dismissals)
7. 结构化评测报告输出 (Console Tables, JSON, CSV)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import json
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

from jetf.benchmarks.adapter import (
    check_matchms_available,
    jetf_library_to_matchms,
    jetf_peaks_to_matchms,
)
from jetf.benchmarks.blink_adapter import BlinkBenchmarkEngine, check_blink_available
from jetf.benchmarks.dataset import (
    BenchmarkDataset,
    sample_query_spectra,
    sample_query_spectra_from_forest,
    slice_parsed_library,
)
from jetf.benchmarks.flashentropy_adapter import (
    FlashEntropyBenchmarkEngine,
    check_flashentropy_available,
    score_flashentropy_pair,
)
from jetf.bounds import adaptive_numba_threads
from jetf.builder import build_forest_index
from jetf.cleaning import DEFAULT_CLEAN_CONFIG, MatchmsCleanConfig
from jetf.gpu import (
    GpuForestIndex,
    is_cuda_available,
    require_cuda,
    search_forest_batch_gpu,
)
from jetf.preprocessing import preprocess_library
from jetf.query import IonModePolicy, PrecursorWindow, QueryConfig, SearchMode, is_eligible
from jetf.results import SearchOutcome
from jetf.scoring import score_greedy_cosine
from jetf.search import search_forest, search_forest_batch
from jetf.serialization import load_forest_snapshot
from jetf.structure import DEFAULT_FOREST_SPEC, ForestIndex
from jetf.types import IonMode, SpectrumMeta, SpectrumPeaks

if TYPE_CHECKING:
    import matchms

logger = logging.getLogger(__name__)


def detect_available_engines() -> dict[str, bool]:
    """探测当前运行环境中各检索引擎与执行后端的可用状态。"""
    cuda_ok = is_cuda_available()
    status = {
        "jetf-gpu": cuda_ok,
        "jetf-cpu-mt": True,
        "jetf-cpu-1t": True,
        "jetf": True,
        "matchms": False,
        "blink": False,
        "flashentropy": False,
    }

    try:
        check_matchms_available()
        status["matchms"] = True
    except Exception:
        pass

    try:
        check_blink_available()
        status["blink"] = True
    except Exception:
        pass

    try:
        check_flashentropy_available()
        status["flashentropy"] = True
    except Exception:
        pass

    return status


@dataclass(frozen=True)
class LatencyStats:
    """引擎/后端检索吞吐量与时延统计 (单位: 毫秒 ms) 与 QPS。"""

    engine: str
    n_queries: int
    batch_size: int
    wall_time_s: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    qps: float
    speedup_vs_1t: float = 1.0
    speedup_vs_mt: float = 1.0
    speedup_vs_matchms: float = 1.0
    avg_roots_pruned: float = 0.0
    avg_leaves_pruned: float = 0.0
    avg_uind_pruned: float = 0.0
    avg_scored_count: float = 0.0
    avg_pruned_ratio: float = 0.0
    raw_latencies_ms: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class IndexBuildStats:
    """离线索引构建、显存迁移与资源占用统计。"""

    engine: str
    n_spectra: int
    build_time_s: float
    memory_mb: float = 0.0
    vram_mb: float = 0.0
    upload_time_s: float = 0.0
    upload_bandwidth_mb_s: float = 0.0
    energy_savings_mb: float = 0.0


@dataclass(frozen=True)
class CorrectnessStats:
    """数学保真度与零漏检核验结果。"""

    all_zero_false_dismissals: bool = True
    total_false_dismissals: int = 0
    mean_recall_at_k: float = 1.0
    max_score_absolute_error: float = 0.0
    mean_score_absolute_error: float = 0.0
    hit_count_match_rate: float = 1.0


@dataclass(frozen=True)
class UnifiedBenchmarkReport:
    """统一多后端多引擎评测全景报告。"""

    mode: str
    n_queries: int
    batch_size: int
    library_size: int
    tolerance_da: float
    available_engines: list[str]
    index_stats: dict[str, IndexBuildStats]
    latency_stats: dict[str, LatencyStats]
    speedups_vs_1t: dict[str, float]
    speedups_vs_mt: dict[str, float]
    speedups_vs_matchms: dict[str, float]
    scaling_results: list[dict[str, Any]] = field(default_factory=list)
    accuracy_results: dict[str, Any] = field(default_factory=dict)
    correctness_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_queries": self.n_queries,
            "batch_size": self.batch_size,
            "library_size": self.library_size,
            "tolerance_da": self.tolerance_da,
            "available_engines": self.available_engines,
            "metadata": self.metadata,
            "index_stats": {
                k: {
                    "build_time_s": round(v.build_time_s, 4),
                    "memory_mb": round(v.memory_mb, 2),
                    "vram_mb": round(v.vram_mb, 2),
                    "upload_time_s": round(v.upload_time_s, 4),
                    "upload_bandwidth_mb_s": round(v.upload_bandwidth_mb_s, 1),
                    "energy_savings_mb": round(v.energy_savings_mb, 2),
                }
                for k, v in self.index_stats.items()
            },
            "latency_stats": {
                k: {
                    "wall_time_s": round(v.wall_time_s, 3),
                    "qps": round(v.qps, 1),
                    "mean_ms": round(v.mean_ms, 3),
                    "p50_ms": round(v.p50_ms, 3),
                    "p95_ms": round(v.p95_ms, 3),
                    "p99_ms": round(v.p99_ms, 3),
                    "min_ms": round(v.min_ms, 3),
                    "max_ms": round(v.max_ms, 3),
                    "std_ms": round(v.std_ms, 3),
                    "speedup_vs_1t": round(v.speedup_vs_1t, 2),
                    "speedup_vs_mt": round(v.speedup_vs_mt, 2),
                    "speedup_vs_matchms": round(v.speedup_vs_matchms, 2),
                    "avg_roots_pruned": round(v.avg_roots_pruned, 1),
                    "avg_leaves_pruned": round(v.avg_leaves_pruned, 1),
                    "avg_uind_pruned": round(v.avg_uind_pruned, 1),
                    "avg_scored_count": round(v.avg_scored_count, 1),
                    "avg_pruned_ratio": round(v.avg_pruned_ratio, 6),
                }
                for k, v in self.latency_stats.items()
            },
            "speedups_vs_1t": {k: round(v, 2) for k, v in self.speedups_vs_1t.items()},
            "speedups_vs_mt": {k: round(v, 2) for k, v in self.speedups_vs_mt.items()},
            "speedups_vs_matchms": {k: round(v, 2) for k, v in self.speedups_vs_matchms.items()},
            "scaling_results": self.scaling_results,
            "accuracy_results": self.accuracy_results,
            "correctness_results": self.correctness_results,
        }

    def format_console_table(self) -> str:
        """生成出版级多后端多引擎性能全景对比控制台表格。"""
        lines = []
        lines.append("=" * 96)
        lines.append("  JET-Forest vs BLINK vs FlashEntropy vs matchms 全景基准 (GPU / CPU-MT / CPU-1T)")
        lines.append(
            f"  检索模式: {self.mode.upper()} | 库容量: {self.library_size:,} 条 | 查询数: {self.n_queries} | 批大小: {self.batch_size} | 容差: {self.tolerance_da} Da"
        )
        if self.metadata.get("matchms_cleaning", False):
            lines.append("  matchms 数据清洗流水线: [已开启 (Industrial Mode)]")
        lines.append("=" * 96)

        # 1. 离线建索与显存驻留
        lines.append("\n[1] 离线索引 / 快照加载与显存驻留明细 (Index & Memory Footprint):")
        lines.append("-" * 96)
        hdr1 = f"{'Backend / Engine':<20} | {'Time (s)':<10} | {'RAM (MB)':<10} | {'VRAM (MB)':<10} | {'H2D BW (MB/s)':<14} | {'Energy Savings'}"
        lines.append(hdr1)
        lines.append("-" * 96)
        for eng, stat in self.index_stats.items():
            bw_str = f"{stat.upload_bandwidth_mb_s:,.1f}" if stat.upload_bandwidth_mb_s > 0 else "-"
            vram_str = f"{stat.vram_mb:,.1f}" if stat.vram_mb > 0 else "-"
            sav_str = f"{stat.energy_savings_mb:,.1f} MB" if stat.energy_savings_mb > 0 else "-"
            lines.append(
                f"{eng:<20} | {stat.build_time_s:<10.3f} | {stat.memory_mb:<10.1f} | {vram_str:<10} | {bw_str:<14} | {sav_str}"
            )
        lines.append("-" * 96)

        # 2. 吞吐量与延迟横向对比
        lines.append("\n[2] 批量吞吐量与在线检索时延对比 (Batch Throughput QPS & Online Retrieval Latency):")
        lines.append("-" * 96)
        hdr2 = f"{'Backend / Engine':<20} | {'QPS':<10} | {'vs 1T':<8} | {'vs MT':<8} | {'vs MMS':<8} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)'}"
        lines.append(hdr2)
        lines.append("-" * 96)

        for eng, lat in self.latency_stats.items():
            if eng == "matchms" and lat.batch_size > 1 and f"matchms ({lat.batch_size}T)" in self.latency_stats:
                continue
            sp_1t_str = f"{lat.speedup_vs_1t:.2f}x" if lat.speedup_vs_1t > 0 else "-"
            sp_mt_str = f"{lat.speedup_vs_mt:.2f}x" if lat.speedup_vs_mt > 0 else "-"
            sp_mms_str = f"{lat.speedup_vs_matchms:.2f}x" if lat.speedup_vs_matchms > 0 else "-"
            lines.append(
                f"{eng:<20} | {lat.qps:<10,.1f} | {sp_1t_str:<8} | {sp_mt_str:<8} | {sp_mms_str:<8} | {lat.p50_ms:<10.2f} | {lat.p95_ms:<10.2f} | {lat.p99_ms:.2f}"
            )
        lines.append("-" * 96)
        lines.append("* 注: CPU MT 的 P50/P95 包含多线程争用下的任务单次执行耗时，而均值为系统级有效服务时延 (Wall / N)；GPU 采用批处理执行，任务耗时按批大小均匀分摊。")
        if self.n_queries < 100:
            lines.append("* (注: 样本量 N < 100 时 P95/P99 分位数易受单离群点波动影响)")

        # 3. 剪枝效能拆解 (针对 JET-Forest 后端)
        jetf_backends = [k for k in self.latency_stats if k.startswith("jetf")]
        if jetf_backends:
            lines.append("\n[3] JET-Forest 多层剪枝效能分解 (Heterogeneous Pruning Breakdown):")
            lines.append("-" * 96)
            hdr3 = f"{'Backend':<20} | {'Roots Pruned':<14} | {'Leaves Pruned':<14} | {'U_ind Pruned':<14} | {'Scored Count':<14} | {'Pruned Ratio'}"
            lines.append(hdr3)
            lines.append("-" * 96)
            for eng in jetf_backends:
                lat = self.latency_stats[eng]
                p_ratio_str = f"{lat.avg_pruned_ratio * 100:.3f}%"
                lines.append(
                    f"{eng:<20} | {lat.avg_roots_pruned:<14.1f} | {lat.avg_leaves_pruned:<14.1f} | {lat.avg_uind_pruned:<14.1f} | {lat.avg_scored_count:<14.1f} | {p_ratio_str}"
                )
            lines.append("-" * 96)

        # 4. 精度与数学保真校验
        if self.correctness_results:
            lines.append("\n[4] 正确性与数学一致性核验 (Correctness & Zero False Dismissals):")
            lines.append("-" * 96)
            for k, v in self.correctness_results.items():
                if isinstance(v, dict):
                    fd_ok = "PASS (0 漏检)" if v.get("all_zero_false_dismissals", True) else "FAIL"
                    lines.append(f"  {k}:")
                    lines.append(f"    - 假阴性漏检校验: {fd_ok}")
                    lines.append(f"    - Top-K 召回率 (Recall@K vs Ground Truth): {v.get('mean_recall_at_k', 1.0) * 100:.2f}%")
                    lines.append(f"    - 最大绝对打分误差 (MAE): {v.get('max_score_absolute_error', 0.0):.2e}")
            lines.append("-" * 96)

        return "\n".join(lines)


class MultiEngineBenchmarkRunner:
    """多引擎与多执行后端统一基准测试调度器。"""

    def __init__(
        self,
        dataset: BenchmarkDataset | ForestIndex | str | Path,
        engines: Sequence[str] = ("jetf-gpu", "jetf-cpu-mt", "jetf-cpu-1t", "matchms"),
        tolerance_da: float = 0.02,
        batch_size: int = 128,
        cpu_threads: int | None = None,
        clean_matchms: bool = True,
        blink_bin_width: float = 0.001,
        blink_intensity_power: float = 0.5,
        matchms_concurrency: int = 1,
    ) -> None:
        self.tolerance_da = float(tolerance_da)
        self.batch_size = int(batch_size)
        self.cpu_threads = int(cpu_threads if cpu_threads is not None else (os.cpu_count() or 4))
        self.clean_matchms = bool(clean_matchms)
        self.blink_bin_width = float(blink_bin_width)
        self.blink_intensity_power = float(blink_intensity_power)
        self.matchms_concurrency = int(matchms_concurrency)

        # 1. 规范化加载数据集或脱机快照
        self.forest: ForestIndex
        self.dataset: BenchmarkDataset | None = None
        self.gpu_forest: GpuForestIndex | None = None
        self.gpu_mem_stats: dict[str, Any] = {}
        self.library_size: int = 0

        t0 = time.perf_counter()
        if isinstance(dataset, (str, Path)):
            p = Path(dataset)
            if not p.is_file():
                raise FileNotFoundError(f"指定的快照或数据文件不存在: {p}")
            if p.suffix == ".npz":
                self.forest = load_forest_snapshot(p)
                self.library_size = self.forest.n_spectra
                load_time = time.perf_counter() - t0
            else:
                from jetf.benchmarks.dataset import load_benchmark_dataset
                self.dataset = load_benchmark_dataset(p)
                self.forest = self.dataset.forest
                self.library_size = self.dataset.n_spectra
                load_time = time.perf_counter() - t0
        elif isinstance(dataset, ForestIndex):
            self.forest = dataset
            self.library_size = self.forest.n_spectra
            load_time = 0.001
        elif isinstance(dataset, BenchmarkDataset):
            self.dataset = dataset
            self.forest = dataset.forest
            self.library_size = dataset.n_spectra
            load_time = 0.001
        else:
            raise TypeError(f"不支持的数据集类型: {type(dataset).__name__}")

        # 2. 规范化引擎与后端列表
        avail = detect_available_engines()
        normalized_engines: list[str] = []
        for e in engines:
            e_lower = e.lower().strip()
            if e_lower == "auto":
                if avail.get("jetf-gpu", False):
                    normalized_engines.append("jetf-gpu")
                normalized_engines.extend(["jetf-cpu-mt", "jetf-cpu-1t"])
                for ext in ("blink", "flashentropy", "matchms"):
                    if avail.get(ext, False):
                        normalized_engines.append(ext)
            elif e_lower == "jetf":
                if avail.get("jetf-gpu", False):
                    normalized_engines.append("jetf-gpu")
                normalized_engines.append("jetf-cpu-mt")
            elif avail.get(e_lower, False):
                normalized_engines.append(e_lower)
            else:
                logger.warning(f"请求的引擎/后端 '{e}' 在当前环境中不可用，将被跳过。")

        self.enabled_engines = list(dict.fromkeys(normalized_engines))
        if not self.enabled_engines:
            raise ValueError(f"没有可用的引擎！请求引擎: {engines}，当前环境可用状态: {avail}")

        self.index_stats: dict[str, IndexBuildStats] = {}
        self.blink_engine: BlinkBenchmarkEngine | None = None
        self.fe_engine: FlashEntropyBenchmarkEngine | None = None
        self.matchms_lib: list[matchms.Spectrum] | None = None

        self._build_all_indexes(load_time)

    def _build_all_indexes(self, forest_load_time_s: float) -> None:
        """构建/上传各引擎所需索引及常驻显存结构。"""
        # 1. JET-Forest (CPU 索引基础)
        ram_mb = 0.0
        if hasattr(self.forest, "envelopes") and hasattr(self.forest.envelopes, "cell_index"):
            ram_mb = (
                self.forest.envelopes.cell_index.nbytes
                + self.forest.envelopes.max_peak_amplitude.nbytes
                + self.forest.postings.mass.nbytes
                + self.forest.postings.intensity.nbytes
            ) / (1024 * 1024)

        if "jetf-cpu-1t" in self.enabled_engines or "jetf-cpu-mt" in self.enabled_engines:
            for eng in ("jetf-cpu-1t", "jetf-cpu-mt"):
                if eng in self.enabled_engines:
                    self.index_stats[eng] = IndexBuildStats(
                        engine=eng,
                        n_spectra=self.library_size,
                        build_time_s=forest_load_time_s,
                        memory_mb=ram_mb,
                    )

        # 2. JET-Forest (GPU 显存驻留上传)
        if "jetf-gpu" in self.enabled_engines:
            require_cuda()
            from numba import cuda

            cuda.synchronize()
            t0 = time.perf_counter()
            self.gpu_forest = GpuForestIndex.from_forest(self.forest)
            cuda.synchronize()
            upload_time = time.perf_counter() - t0

            mem_sum = self.gpu_forest.device_memory_summary()
            total_dev_bytes = self.gpu_forest.device_memory_bytes()
            vram_mb = total_dev_bytes / (1024 * 1024)
            bandwidth = vram_mb / upload_time if upload_time > 0 else 0.0
            energy_savings = int(self.forest.postings.energy.nbytes) / (1024 * 1024)

            self.gpu_mem_stats = {
                "upload_time_s": upload_time,
                "bandwidth_mb_s": bandwidth,
                "vram_mb": vram_mb,
                "energy_savings_mb": energy_savings,
            }

            self.index_stats["jetf-gpu"] = IndexBuildStats(
                engine="jetf-gpu",
                n_spectra=self.library_size,
                build_time_s=upload_time,
                memory_mb=ram_mb,
                vram_mb=vram_mb,
                upload_time_s=upload_time,
                upload_bandwidth_mb_s=bandwidth,
                energy_savings_mb=energy_savings,
            )

        # 3. BLINK
        if "blink" in self.enabled_engines:
            t0 = time.perf_counter()
            # 若 dataset 存在使用 dataset.library；否则从 forest.postings 重构
            if self.dataset is not None:
                b_lib = self.dataset.library
            else:
                b_lib = [self.forest.postings.spectrum_at(i) for i in range(self.library_size)]
            self.blink_engine = BlinkBenchmarkEngine(
                library=b_lib,
                tolerance=self.tolerance_da,
                bin_width=self.blink_bin_width,
                intensity_power=self.blink_intensity_power,
            )
            t_blink = time.perf_counter() - t0
            self.index_stats["blink"] = IndexBuildStats(
                engine="blink",
                n_spectra=self.library_size,
                build_time_s=t_blink,
            )

        # 4. FlashEntropy
        if "flashentropy" in self.enabled_engines:
            t0 = time.perf_counter()
            if self.dataset is not None:
                fe_lib = self.dataset.library
            else:
                fe_lib = [self.forest.postings.spectrum_at(i) for i in range(self.library_size)]
            self.fe_engine = FlashEntropyBenchmarkEngine(
                library=fe_lib,
                clean_spectra=True,
            )
            t_fe = time.perf_counter() - t0
            self.index_stats["flashentropy"] = IndexBuildStats(
                engine="flashentropy",
                n_spectra=self.library_size,
                build_time_s=t_fe,
            )

        # 5. matchms
        if "matchms" in self.enabled_engines:
            t0 = time.perf_counter()
            if self.dataset is not None:
                self.matchms_lib = jetf_library_to_matchms(
                    self.dataset.library,
                    clean=self.clean_matchms,
                )
            else:
                # 从快照构建 matchms 谱列表
                from matchms import Spectrum
                m_list = []
                for i in range(self.library_size):
                    p = self.forest.postings.spectrum_at(i)
                    m = self.forest.spectra[i] if self.forest.spectra else None
                    m_list.append(
                        jetf_peaks_to_matchms(
                            p,
                            meta=m,
                            clean=self.clean_matchms,
                        )
                    )
                self.matchms_lib = m_list
            t_mms = time.perf_counter() - t0
            self.index_stats["matchms"] = IndexBuildStats(
                engine="matchms",
                n_spectra=self.library_size,
                build_time_s=t_mms,
            )

        # 兼容性别名: 若有任意 jetf 后端，保留 "jetf" 键指向首选后端
        jetf_alias = (
            self.index_stats.get("jetf-gpu")
            or self.index_stats.get("jetf-cpu-mt")
            or self.index_stats.get("jetf-cpu-1t")
        )
        if jetf_alias is not None:
            self.index_stats["jetf"] = jetf_alias

    def sample_queries(self, n_queries: int, seed: int = 2026) -> list[tuple[int, SpectrumPeaks]]:
        """从数据源或脱机快照中确定性抽样代表性查询谱。"""
        if self.dataset is not None:
            return sample_query_spectra(self.dataset.library, n_queries=n_queries, seed=seed)
        return sample_query_spectra_from_forest(self.forest, n_queries=n_queries, seed=seed)

    def run_benchmark(
        self,
        queries_info: Sequence[tuple[int, SpectrumPeaks]],
        mode: str = "open",
        top_k: int = 10,
        theta: float = 0.7,
        precursor_window_da: float = 0.02,
        batch_size: int | None = None,
        warmup_queries: int = 2,
        exclude_self: bool = False,
        concurrency: int | None = None,
    ) -> dict[str, LatencyStats]:
        """统一运行各后端与引擎的批量吞吐量与时延基准测试。"""
        eff_bs = batch_size if batch_size is not None else self.batch_size
        results: dict[str, LatencyStats] = {}
        self.last_outcomes: dict[str, list[SearchOutcome]] = {}
        n_q = len(queries_info)
        if n_q == 0:
            return results

        q_peaks_list = [q[1] for q in queries_info]
        q_metas: list[SpectrumMeta | None] = []
        for q in queries_info:
            row = q[0]
            if self.dataset is not None:
                q_metas.append(self.dataset.library.spectra[row])
            elif self.forest.spectra:
                q_metas.append(self.forest.spectra[row])
            else:
                q_metas.append(None)

        q_pmzs = [
            float(m.precursor_mz) if m and m.precursor_mz is not None and np.isfinite(m.precursor_mz) else None
            for m in q_metas
        ]

        # 构造统一 QueryConfig 列表
        cfgs: list[QueryConfig] = []
        is_topk = mode.lower() in ("open", "topk", "top_k")
        for i in range(n_q):
            p_win = (
                PrecursorWindow(mz=q_pmzs[i], tolerance_da=precursor_window_da)
                if (mode == "identity" and q_pmzs[i] is not None)
                else None
            )
            i_mode = q_metas[i].ion_mode if q_metas[i] else None
            ex_id = q_metas[i].external_id if (exclude_self and q_metas[i] and q_metas[i].external_id) else None
            cfgs.append(
                QueryConfig(
                    mode=SearchMode.TOP_K if is_topk else SearchMode.THRESHOLD,
                    k=top_k,
                    threshold=theta,
                    fragment_tolerance_da=self.tolerance_da,
                    ion_mode=i_mode,
                    precursor_window=p_win,
                    ion_mode_policy=IonModePolicy.EXACT if mode == "identity" else IonModePolicy.ANY,
                    min_matched_peaks=1,
                    exclude_spectrum_id=ex_id,
                )
            )

        # ---------------------------------------------------------------------
        # 1. JET-Forest (GPU 异构流水线)
        # ---------------------------------------------------------------------
        if "jetf-gpu" in self.enabled_engines and self.gpu_forest is not None:
            require_cuda()
            from numba import cuda

            # Warmup
            w_n = min(warmup_queries, n_q)
            if w_n > 0:
                _ = search_forest_batch_gpu(
                    q_peaks_list[:w_n],
                    self.gpu_forest,
                    config=cfgs[:w_n],
                    batch_size=w_n,
                    uind=True,
                )
                cuda.synchronize()

            gc.collect()
            cuda.synchronize()
            t0 = time.perf_counter()
            gpu_outcomes = search_forest_batch_gpu(
                q_peaks_list,
                self.gpu_forest,
                config=cfgs,
                batch_size=eff_bs,
                uind=True,
            )
            cuda.synchronize()
            self.last_outcomes["jetf-gpu"] = gpu_outcomes
            wall_time = time.perf_counter() - t0
            qps = n_q / wall_time if wall_time > 0 else 0.0

            raw_lats = [o.stats.bound_eval_time_ms + o.stats.exact_eval_time_ms for o in gpu_outcomes]
            eff_mean = (wall_time / n_q) * 1000.0

            # 统计剪枝
            r_pruned = [o.stats.pruned_by_layer.get("roots_pruned", 0) for o in gpu_outcomes]
            l_pruned = [o.stats.pruned_by_layer.get("leaves_pruned", 0) for o in gpu_outcomes]
            u_pruned = [o.stats.pruned_by_layer.get("uind_pruned", 0) for o in gpu_outcomes]
            scored = [o.stats.n_scored for o in gpu_outcomes]
            avg_scored = float(np.mean(scored)) if scored else 0.0
            avg_pruned_ratio = 1.0 - (avg_scored / self.library_size) if self.library_size > 0 else 1.0

            results["jetf-gpu"] = self._compute_latency_stats(
                engine="jetf-gpu",
                lats=raw_lats,
                wall_time_s=wall_time,
                eff_mean_ms=eff_mean,
                qps=qps,
                batch_size=eff_bs,
                avg_roots_pruned=float(np.mean(r_pruned)) if r_pruned else 0.0,
                avg_leaves_pruned=float(np.mean(l_pruned)) if l_pruned else 0.0,
                avg_uind_pruned=float(np.mean(u_pruned)) if u_pruned else 0.0,
                avg_scored_count=avg_scored,
                avg_pruned_ratio=avg_pruned_ratio,
            )

        # ---------------------------------------------------------------------
        # 2. JET-Forest (CPU 多线程批处理)
        # ---------------------------------------------------------------------
        if "jetf-cpu-mt" in self.enabled_engines:
            w_n = min(warmup_queries, n_q)
            if w_n > 0:
                _ = search_forest_batch(
                    q_peaks_list[:w_n],
                    self.forest,
                    config=cfgs[:w_n],
                    concurrency=self.cpu_threads,
                    uind=True,
                )

            gc.collect()
            t0 = time.perf_counter()
            cpu_mt_outcomes = search_forest_batch(
                q_peaks_list,
                self.forest,
                config=cfgs,
                concurrency=self.cpu_threads,
                uind=True,
            )
            self.last_outcomes["jetf-cpu-mt"] = cpu_mt_outcomes
            wall_time = time.perf_counter() - t0
            qps = n_q / wall_time if wall_time > 0 else 0.0

            raw_lats = [o.stats.bound_eval_time_ms + o.stats.exact_eval_time_ms for o in cpu_mt_outcomes]
            eff_mean = (wall_time / n_q) * 1000.0

            r_pruned = [o.stats.pruned_by_layer.get("roots_pruned", 0) for o in cpu_mt_outcomes]
            l_pruned = [o.stats.pruned_by_layer.get("leaves_pruned", 0) for o in cpu_mt_outcomes]
            u_pruned = [o.stats.pruned_by_layer.get("uind_pruned", 0) for o in cpu_mt_outcomes]
            scored = [o.stats.n_scored for o in cpu_mt_outcomes]
            avg_scored = float(np.mean(scored)) if scored else 0.0
            avg_pruned_ratio = 1.0 - (avg_scored / self.library_size) if self.library_size > 0 else 1.0

            results["jetf-cpu-mt"] = self._compute_latency_stats(
                engine="jetf-cpu-mt",
                lats=raw_lats,
                wall_time_s=wall_time,
                eff_mean_ms=eff_mean,
                qps=qps,
                batch_size=eff_bs,
                avg_roots_pruned=float(np.mean(r_pruned)) if r_pruned else 0.0,
                avg_leaves_pruned=float(np.mean(l_pruned)) if l_pruned else 0.0,
                avg_uind_pruned=float(np.mean(u_pruned)) if u_pruned else 0.0,
                avg_scored_count=avg_scored,
                avg_pruned_ratio=avg_pruned_ratio,
            )

        # ---------------------------------------------------------------------
        # 3. JET-Forest (CPU 单线程基准)
        # ---------------------------------------------------------------------
        if "jetf-cpu-1t" in self.enabled_engines:
            with adaptive_numba_threads(target_threads=1):
                w_n = min(warmup_queries, n_q)
                if w_n > 0:
                    _ = search_forest_batch(
                        q_peaks_list[:w_n],
                        self.forest,
                        config=cfgs[:w_n],
                        concurrency=1,
                        uind=True,
                    )

                gc.collect()
                t0 = time.perf_counter()
                cpu_1t_outcomes = search_forest_batch(
                    q_peaks_list,
                    self.forest,
                    config=cfgs,
                    concurrency=1,
                    uind=True,
                )
                self.last_outcomes["jetf-cpu-1t"] = cpu_1t_outcomes
                wall_time = time.perf_counter() - t0
                qps = n_q / wall_time if wall_time > 0 else 0.0

                raw_lats = [o.stats.bound_eval_time_ms + o.stats.exact_eval_time_ms for o in cpu_1t_outcomes]
                eff_mean = (wall_time / n_q) * 1000.0

                r_pruned = [o.stats.pruned_by_layer.get("roots_pruned", 0) for o in cpu_1t_outcomes]
                l_pruned = [o.stats.pruned_by_layer.get("leaves_pruned", 0) for o in cpu_1t_outcomes]
                u_pruned = [o.stats.pruned_by_layer.get("uind_pruned", 0) for o in cpu_1t_outcomes]
                scored = [o.stats.n_scored for o in cpu_1t_outcomes]
                avg_scored = float(np.mean(scored)) if scored else 0.0
                avg_pruned_ratio = 1.0 - (avg_scored / self.library_size) if self.library_size > 0 else 1.0

                results["jetf-cpu-1t"] = self._compute_latency_stats(
                    engine="jetf-cpu-1t",
                    lats=raw_lats,
                    wall_time_s=wall_time,
                    eff_mean_ms=eff_mean,
                    qps=qps,
                    batch_size=1,
                    avg_roots_pruned=float(np.mean(r_pruned)) if r_pruned else 0.0,
                    avg_leaves_pruned=float(np.mean(l_pruned)) if l_pruned else 0.0,
                    avg_uind_pruned=float(np.mean(u_pruned)) if u_pruned else 0.0,
                    avg_scored_count=avg_scored,
                    avg_pruned_ratio=avg_pruned_ratio,
                )

        # ---------------------------------------------------------------------
        # 4. BLINK
        # ---------------------------------------------------------------------
        if "blink" in self.enabled_engines and self.blink_engine is not None:
            if mode == "identity":
                logger.warning("BLINK 原生不支持前体 m/z 过滤，在 identity 模式下跳过。")
            else:
                w_n = min(warmup_queries, n_q)
                for i in range(w_n):
                    self.blink_engine.search_single(q_peaks_list[i], top_k=top_k)

                gc.collect()
                t0 = time.perf_counter()
                b_lats = []
                for i in range(n_q):
                    t_single = time.perf_counter()
                    self.blink_engine.search_single(q_peaks_list[i], top_k=top_k)
                    b_lats.append((time.perf_counter() - t_single) * 1000.0)
                wall_time = time.perf_counter() - t0
                qps = n_q / wall_time if wall_time > 0 else 0.0
                eff_mean = (wall_time / n_q) * 1000.0

                results["blink"] = self._compute_latency_stats(
                    engine="blink",
                    lats=b_lats,
                    wall_time_s=wall_time,
                    eff_mean_ms=eff_mean,
                    qps=qps,
                    batch_size=1,
                )

        # ---------------------------------------------------------------------
        # 5. FlashEntropy
        # ---------------------------------------------------------------------
        if "flashentropy" in self.enabled_engines and self.fe_engine is not None:
            w_n = min(warmup_queries, n_q)
            for i in range(w_n):
                self.fe_engine.search_single(
                    query_peaks=q_peaks_list[i],
                    precursor_mz=q_pmzs[i],
                    top_k=top_k,
                    mode=mode,
                    ms1_tolerance_da=precursor_window_da,
                    ms2_tolerance_da=self.tolerance_da,
                )

            gc.collect()
            t0 = time.perf_counter()
            fe_lats = []
            for i in range(n_q):
                t_single = time.perf_counter()
                self.fe_engine.search_single(
                    query_peaks=q_peaks_list[i],
                    precursor_mz=q_pmzs[i],
                    top_k=top_k,
                    mode=mode,
                    ms1_tolerance_da=precursor_window_da,
                    ms2_tolerance_da=self.tolerance_da,
                )
                fe_lats.append((time.perf_counter() - t_single) * 1000.0)
            wall_time = time.perf_counter() - t0
            qps = n_q / wall_time if wall_time > 0 else 0.0
            eff_mean = (wall_time / n_q) * 1000.0

            results["flashentropy"] = self._compute_latency_stats(
                engine="flashentropy",
                lats=fe_lats,
                wall_time_s=wall_time,
                eff_mean_ms=eff_mean,
                qps=qps,
                batch_size=1,
            )

        # ---------------------------------------------------------------------
        # 6. matchms (工业标准 Ground Truth)
        # ---------------------------------------------------------------------
        if "matchms" in self.enabled_engines and self.matchms_lib is not None:
            import heapq
            from matchms.similarity import CosineGreedy

            scorer = CosineGreedy(tolerance=self.tolerance_da, mz_power=0.0, intensity_power=1.0)
            m_queries = [
                jetf_peaks_to_matchms(p, m, clean=self.clean_matchms)
                for p, m in zip(q_peaks_list, q_metas)
            ]
            lib_spectra = (
                self.dataset.library.spectra
                if (self.dataset is not None and self.dataset.library.spectra is not None)
                else (self.forest.spectra if self.forest.spectra is not None else None)
            )

            def _is_candidate_eligible(q_cfg_item: QueryConfig, sm: SpectrumMeta | None) -> bool:
                if sm is None:
                    if q_cfg_item.precursor_window is not None:
                        return False
                    if q_cfg_item.ion_mode_policy == IonModePolicy.EXACT and q_cfg_item.ion_mode != IonMode.UNKNOWN:
                        return False
                    return True
                return is_eligible(q_cfg_item, sm)

            w_n = min(warmup_queries, n_q)
            for i in range(w_n):
                orig_row = queries_info[i][0]
                q_cfg = cfgs[i]
                k = q_cfg.k or top_k
                is_threshold_mode = (q_cfg.mode == SearchMode.THRESHOLD)
                thresh_val = q_cfg.threshold or 0.0
                if is_threshold_mode:
                    for row, lib_s in enumerate(self.matchms_lib):
                        if exclude_self and row == orig_row:
                            continue
                        meta = lib_spectra[row] if lib_spectra and row < len(lib_spectra) else None
                        if not _is_candidate_eligible(q_cfg, meta):
                            continue
                        res = scorer.pair(m_queries[i], lib_s)
                        if res is None:
                            continue
                        s_val = float(res["score"])
                        n_m = int(res["matches"])
                        if n_m >= q_cfg.min_matched_peaks and s_val >= thresh_val:
                            pass
                else:
                    top_k_heap: list[tuple[float, int]] = []
                    for row, lib_s in enumerate(self.matchms_lib):
                        if exclude_self and row == orig_row:
                            continue
                        meta = lib_spectra[row] if lib_spectra and row < len(lib_spectra) else None
                        if not _is_candidate_eligible(q_cfg, meta):
                            continue
                        res = scorer.pair(m_queries[i], lib_s)
                        if res is None:
                            continue
                        s_val = float(res["score"])
                        n_m = int(res["matches"])
                        if n_m >= q_cfg.min_matched_peaks:
                            if len(top_k_heap) < k:
                                heapq.heappush(top_k_heap, (s_val, row))
                            elif s_val > top_k_heap[0][0]:
                                heapq.heapreplace(top_k_heap, (s_val, row))

            mms_conc = concurrency if concurrency is not None else getattr(self, "matchms_concurrency", 1)

            def _eval_single_mms(i: int) -> float:
                orig_row = queries_info[i][0]
                q_cfg = cfgs[i]
                k = q_cfg.k or top_k
                is_threshold_mode = (q_cfg.mode == SearchMode.THRESHOLD)
                thresh_val = q_cfg.threshold or 0.0
                local_scorer = CosineGreedy(tolerance=self.tolerance_da, mz_power=0.0, intensity_power=1.0)
                t_single = time.perf_counter()
                if is_threshold_mode:
                    candidates: list[tuple[float, int]] = []
                    for row, lib_s in enumerate(self.matchms_lib):
                        if exclude_self and row == orig_row:
                            continue
                        meta = lib_spectra[row] if lib_spectra and row < len(lib_spectra) else None
                        if not _is_candidate_eligible(q_cfg, meta):
                            continue
                        res = local_scorer.pair(m_queries[i], lib_s)
                        if res is None:
                            continue
                        s_val = float(res["score"])
                        n_m = int(res["matches"])
                        if n_m >= q_cfg.min_matched_peaks and s_val >= thresh_val:
                            candidates.append((s_val, row))
                    candidates.sort(key=lambda x: -x[0])
                else:
                    top_k_heap: list[tuple[float, int]] = []
                    for row, lib_s in enumerate(self.matchms_lib):
                        if exclude_self and row == orig_row:
                            continue
                        meta = lib_spectra[row] if lib_spectra and row < len(lib_spectra) else None
                        if not _is_candidate_eligible(q_cfg, meta):
                            continue
                        res = local_scorer.pair(m_queries[i], lib_s)
                        if res is None:
                            continue
                        s_val = float(res["score"])
                        n_m = int(res["matches"])
                        if n_m >= q_cfg.min_matched_peaks:
                            if len(top_k_heap) < k:
                                heapq.heappush(top_k_heap, (s_val, row))
                            elif s_val > top_k_heap[0][0]:
                                heapq.heapreplace(top_k_heap, (s_val, row))
                    top_k_heap.sort(key=lambda x: -x[0])
                return (time.perf_counter() - t_single) * 1000.0

            gc.collect()
            t0 = time.perf_counter()
            if mms_conc > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=mms_conc) as executor:
                    mms_lats = list(executor.map(_eval_single_mms, range(n_q)))
            else:
                mms_lats = [_eval_single_mms(i) for i in range(n_q)]

            wall_time = time.perf_counter() - t0
            qps = n_q / wall_time if wall_time > 0 else 0.0
            eff_mean = (wall_time / n_q) * 1000.0

            mms_engine_name = f"matchms ({mms_conc}T)" if mms_conc > 1 else "matchms"
            mms_stat = self._compute_latency_stats(
                engine=mms_engine_name,
                lats=mms_lats,
                wall_time_s=wall_time,
                eff_mean_ms=eff_mean,
                qps=qps,
                batch_size=mms_conc if mms_conc > 1 else 1,
            )
            results["matchms"] = mms_stat
            if mms_conc > 1:
                results[mms_engine_name] = mms_stat

        # ---------------------------------------------------------------------
        # 计算加速比矩阵 (Speedup Matrix)
        # ---------------------------------------------------------------------
        time_1t = results["jetf-cpu-1t"].wall_time_s if "jetf-cpu-1t" in results else None
        time_mt = results["jetf-cpu-mt"].wall_time_s if "jetf-cpu-mt" in results else None
        time_mms = results["matchms"].wall_time_s if "matchms" in results else None

        updated_results: dict[str, LatencyStats] = {}
        for eng, st in results.items():
            sp_1t = (time_1t / st.wall_time_s) if time_1t and st.wall_time_s > 0 else 1.0
            sp_mt = (time_mt / st.wall_time_s) if time_mt and st.wall_time_s > 0 else 1.0
            sp_mms = (time_mms / st.wall_time_s) if time_mms and st.wall_time_s > 0 else 1.0
            updated_results[eng] = LatencyStats(
                engine=st.engine,
                n_queries=st.n_queries,
                batch_size=st.batch_size,
                wall_time_s=st.wall_time_s,
                mean_ms=st.mean_ms,
                p50_ms=st.p50_ms,
                p95_ms=st.p95_ms,
                p99_ms=st.p99_ms,
                min_ms=st.min_ms,
                max_ms=st.max_ms,
                std_ms=st.std_ms,
                qps=st.qps,
                speedup_vs_1t=sp_1t,
                speedup_vs_mt=sp_mt,
                speedup_vs_matchms=sp_mms,
                avg_roots_pruned=st.avg_roots_pruned,
                avg_leaves_pruned=st.avg_leaves_pruned,
                avg_uind_pruned=st.avg_uind_pruned,
                avg_scored_count=st.avg_scored_count,
                avg_pruned_ratio=st.avg_pruned_ratio,
                raw_latencies_ms=st.raw_latencies_ms,
            )

        # 兼容性别名: 若有任何 jetf 后端，保留 "jetf" 键指向首选后端
        if "jetf-gpu" in updated_results:
            updated_results["jetf"] = updated_results["jetf-gpu"]
        elif "jetf-cpu-mt" in updated_results:
            updated_results["jetf"] = updated_results["jetf-cpu-mt"]
        elif "jetf-cpu-1t" in updated_results:
            updated_results["jetf"] = updated_results["jetf-cpu-1t"]

        return updated_results

    def run_latency_benchmark(
        self,
        queries_info: Sequence[tuple[int, SpectrumPeaks]],
        mode: str = "open",
        top_k: int = 10,
        theta: float = 0.7,
        precursor_window_da: float = 0.02,
        batch_size: int | None = None,
        warmup_queries: int = 2,
        exclude_self: bool = False,
        concurrency: int | None = None,
    ) -> dict[str, LatencyStats]:
        """兼容别名，调用统一的 run_benchmark。"""
        return self.run_benchmark(
            queries_info=queries_info,
            mode=mode,
            top_k=top_k,
            theta=theta,
            precursor_window_da=precursor_window_da,
            batch_size=batch_size,
            warmup_queries=warmup_queries,
            exclude_self=exclude_self,
            concurrency=concurrency,
        )

    def run_scaling_benchmark(
        self,
        scales: Sequence[int] = (100, 500, 1000, 2000, 5000),
        n_queries: int = 30,
        mode: str = "open",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """在递增的库规模上评测各引擎的 QPS 与单次时延扩展性。"""
        scaling_records: list[dict[str, Any]] = []
        max_avail = self.library_size
        eff_scales = [s for s in scales if s <= max_avail]
        if not eff_scales:
            eff_scales = [min(max_avail, 1000)]

        queries = self.sample_queries(n_queries=n_queries)
        q_peaks = [q[1] for q in queries]
        q_rows = [q[0] for q in queries]
        q_metas = [
            self.dataset.library.spectra[r]
            if self.dataset is not None
            else (self.forest.spectra[r] if self.forest.spectra else None)
            for r in q_rows
        ]
        q_pmzs = [
            float(m.precursor_mz) if m and m.precursor_mz is not None and np.isfinite(m.precursor_mz) else None
            for m in q_metas
        ]

        for s in eff_scales:
            rec: dict[str, Any] = {"scale": s}
            if self.dataset is not None:
                sub_indices = np.arange(s, dtype=np.int64)
                sub_parsed = slice_parsed_library(self.dataset.parsed, sub_indices)
                sub_lib = preprocess_library(sub_parsed, self.dataset.library.spec)
                sub_forest = build_forest_index(sub_lib, DEFAULT_FOREST_SPEC)
            else:
                sub_forest = self.forest

            # JETF
            cfgs = [
                QueryConfig(
                    mode=SearchMode.TOP_K,
                    k=top_k,
                    fragment_tolerance_da=self.tolerance_da,
                    ion_mode=m.ion_mode if m else None,
                )
                for m in q_metas
            ]
            t0 = time.perf_counter()
            search_forest_batch(
                q_peaks,
                sub_forest,
                config=cfgs,
                concurrency=self.cpu_threads,
                uind=True,
            )
            t_used = time.perf_counter() - t0
            rec["jetf_latency_mean_ms"] = (t_used / len(queries)) * 1000.0
            rec["jetf_qps"] = len(queries) / max(1e-6, t_used)

            # BLINK
            if "blink" in self.enabled_engines and self.blink_engine is not None and mode != "identity":
                t0 = time.perf_counter()
                self.blink_engine.search_batch(q_peaks, top_k=top_k)
                t_used = time.perf_counter() - t0
                rec["blink_latency_mean_ms"] = (t_used / len(queries)) * 1000.0
                rec["blink_qps"] = len(queries) / max(1e-6, t_used)

            # FlashEntropy
            if "flashentropy" in self.enabled_engines and self.fe_engine is not None:
                t0 = time.perf_counter()
                for i in range(len(queries)):
                    self.fe_engine.search_single(
                        q_peaks[i], precursor_mz=q_pmzs[i], top_k=top_k, mode=mode
                    )
                t_used = time.perf_counter() - t0
                rec["fe_latency_mean_ms"] = (t_used / len(queries)) * 1000.0
                rec["fe_qps"] = len(queries) / max(1e-6, t_used)

            scaling_records.append(rec)

        return scaling_records

    @staticmethod
    def _compute_latency_stats(
        engine: str,
        lats: list[float],
        wall_time_s: float,
        eff_mean_ms: float,
        qps: float,
        batch_size: int,
        avg_roots_pruned: float = 0.0,
        avg_leaves_pruned: float = 0.0,
        avg_uind_pruned: float = 0.0,
        avg_scored_count: float = 0.0,
        avg_pruned_ratio: float = 0.0,
    ) -> LatencyStats:
        arr = np.asarray(lats, dtype=np.float64)
        return LatencyStats(
            engine=engine,
            n_queries=len(lats),
            batch_size=batch_size,
            wall_time_s=float(wall_time_s),
            mean_ms=float(eff_mean_ms),
            p50_ms=float(np.percentile(arr, 50)) if arr.size > 0 else 0.0,
            p95_ms=float(np.percentile(arr, 95)) if arr.size > 0 else 0.0,
            p99_ms=float(np.percentile(arr, 99)) if arr.size > 0 else 0.0,
            min_ms=float(np.min(arr)) if arr.size > 0 else 0.0,
            max_ms=float(np.max(arr)) if arr.size > 0 else 0.0,
            std_ms=float(np.std(arr)) if arr.size > 0 else 0.0,
            qps=float(qps),
            avg_roots_pruned=avg_roots_pruned,
            avg_leaves_pruned=avg_leaves_pruned,
            avg_uind_pruned=avg_uind_pruned,
            avg_scored_count=avg_scored_count,
            avg_pruned_ratio=avg_pruned_ratio,
            raw_latencies_ms=lats,
        )

    @staticmethod
    def verify_outcomes_correctness(
        ref_outcomes: list[SearchOutcome],
        test_outcomes: list[SearchOutcome],
    ) -> dict[str, Any]:
        """严格计算两组 SearchOutcome 间的真实保真度与零漏检指标。"""
        n_q = len(ref_outcomes)
        if n_q != len(test_outcomes):
            raise ValueError(f"结果数量不匹配: REF {n_q} vs TEST {len(test_outcomes)}")

        score_diffs: list[float] = []
        recalls: list[float] = []
        false_dismissals = 0
        hit_count_matches = 0

        for q_idx in range(n_q):
            ref_hits = ref_outcomes[q_idx].hits
            test_hits = test_outcomes[q_idx].hits

            if len(ref_hits) == len(test_hits):
                hit_count_matches += 1

            ref_keys = [(h.spectrum_index, h.external_id) for h in ref_hits]
            test_keys = [(h.spectrum_index, h.external_id) for h in test_hits]

            ref_key_set = set(ref_keys)
            test_key_set = set(test_keys)

            if len(ref_key_set) == 0:
                rec = 1.0
            else:
                intersection = test_key_set & ref_key_set
                rec = len(intersection) / len(ref_key_set)
                missed = len(ref_key_set - test_key_set)
                if missed > 0:
                    false_dismissals += missed
            recalls.append(rec)

            test_hit_dict = {h.spectrum_index: h.score for h in test_hits}
            for h_ref in ref_hits:
                if h_ref.spectrum_index in test_hit_dict:
                    diff = abs(float(h_ref.score) - float(test_hit_dict[h_ref.spectrum_index]))
                    score_diffs.append(diff)

        max_diff = float(np.max(score_diffs)) if score_diffs else 0.0
        mean_diff = float(np.mean(score_diffs)) if score_diffs else 0.0
        mean_rec = float(np.mean(recalls)) if recalls else 1.0
        hit_match_rate = hit_count_matches / n_q if n_q > 0 else 1.0

        return {
            "all_zero_false_dismissals": (false_dismissals == 0),
            "total_false_dismissals": false_dismissals,
            "mean_recall_at_k": mean_rec,
            "max_score_absolute_error": max_diff,
            "mean_score_absolute_error": mean_diff,
            "hit_count_match_rate": hit_match_rate,
        }

    def generate_report(
        self,
        n_queries: int = 256,
        mode: str = "open",
        top_k: int = 10,
        theta: float = 0.7,
        batch_size: int | None = None,
        run_scaling: bool = False,
        scales: Sequence[int] = (100, 500, 1000, 2000, 5000),
        concurrency: int | None = None,
    ) -> UnifiedBenchmarkReport:
        """执行全套统一多后端基准测试并生成综合报告。"""
        eff_bs = batch_size if batch_size is not None else self.batch_size
        queries = self.sample_queries(n_queries=n_queries)
        lat_stats = self.run_benchmark(
            queries,
            mode=mode,
            top_k=top_k,
            theta=theta,
            batch_size=eff_bs,
            concurrency=concurrency,
        )

        sp_1t = {k: v.speedup_vs_1t for k, v in lat_stats.items()}
        sp_mt = {k: v.speedup_vs_mt for k, v in lat_stats.items()}
        sp_mms = {k: v.speedup_vs_matchms for k, v in lat_stats.items()}

        correctness_dict: dict[str, Any] = {}
        # 若 GPU 与 CPU-1T 均已执行，动态真实对拍检验
        if hasattr(self, "last_outcomes"):
            if "jetf-gpu" in self.last_outcomes and "jetf-cpu-1t" in self.last_outcomes:
                correctness_dict["gpu_vs_cpu_1t"] = self.verify_outcomes_correctness(
                    ref_outcomes=self.last_outcomes["jetf-cpu-1t"],
                    test_outcomes=self.last_outcomes["jetf-gpu"],
                )
            if "jetf-cpu-mt" in self.last_outcomes and "jetf-cpu-1t" in self.last_outcomes:
                correctness_dict["cpu_mt_vs_cpu_1t"] = self.verify_outcomes_correctness(
                    ref_outcomes=self.last_outcomes["jetf-cpu-1t"],
                    test_outcomes=self.last_outcomes["jetf-cpu-mt"],
                )

        scaling_res: list[dict[str, Any]] = []
        if run_scaling:
            scaling_res = self.run_scaling_benchmark(
                scales=scales,
                n_queries=min(n_queries, 30),
                mode=mode,
                top_k=top_k,
            )

        metadata = {
            "matchms_cleaning": self.clean_matchms,
            "cpu_threads": self.cpu_threads,
            "gpu_memory": self.gpu_mem_stats,
        }

        return UnifiedBenchmarkReport(
            mode=mode,
            n_queries=n_queries,
            batch_size=eff_bs,
            library_size=self.library_size,
            tolerance_da=self.tolerance_da,
            available_engines=self.enabled_engines,
            index_stats=self.index_stats,
            latency_stats=lat_stats,
            speedups_vs_1t=sp_1t,
            speedups_vs_mt=sp_mt,
            speedups_vs_matchms=sp_mms,
            scaling_results=scaling_res,
            accuracy_results={},
            correctness_results=correctness_dict,
            metadata=metadata,
        )


UnifiedBenchmarkRunner = MultiEngineBenchmarkRunner

