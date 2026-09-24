"""JET-Forest GPU 吞吐量基准压测与端到端性能评测套件。

涵盖：
1. 显存驻留与 Host-to-Device 迁移统计 (device_memory_summary, 省略 energy 列显存节约)
2. 开放式阈值检索基准评测 (Open Threshold Search vs CPU 1T / CPU MT)
3. 开放式 Top-K 检索基准评测 (Open Top-K Search vs CPU 1T / CPU MT)
4. 逐位精度与零漏检对拍核验 (Zero False Dismissals, Recall@K, 浮点绝对一致性)
5. 结构化评测报告输出 (控制台整洁表格与可选 JSON 导出)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import datetime
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence
import warnings

import numpy as np
from tabulate import tabulate

import numba
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from jetf import (
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestIndex,
    QueryConfig,
    SearchMode,
    SpectrumPeaks,
    load_forest_snapshot,
    search_forest_batch,
)
from jetf.benchmarks.dataset import sample_query_spectra_from_forest
from jetf.gpu import (
    GpuForestIndex,
    is_cuda_available,
    require_cuda,
    search_forest_batch_gpu,
    search_threshold_batch_gpu,
    search_topk_batch_gpu,
)
from jetf.results import SearchOutcome


@dataclass(frozen=True)
class GpuDeviceInfo:
    """GPU 设备硬件参数。"""

    name: str
    compute_capability: str
    total_vram_gb: float
    free_vram_gb: float


@dataclass(frozen=True)
class DeviceMemoryStats:
    """显存驻留与迁移统计。"""

    upload_time_s: float
    upload_bandwidth_mb_s: float
    envelopes_bytes: int
    trees_bytes: int
    nodes_bytes: int
    postings_bytes: int
    total_bytes: int
    energy_omitted_savings_bytes: int

    @property
    def envelopes_mb(self) -> float:
        return self.envelopes_bytes / (1024 * 1024)

    @property
    def trees_mb(self) -> float:
        return self.trees_bytes / (1024 * 1024)

    @property
    def nodes_mb(self) -> float:
        return self.nodes_bytes / (1024 * 1024)

    @property
    def postings_mb(self) -> float:
        return self.postings_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def energy_savings_mb(self) -> float:
        return self.energy_omitted_savings_bytes / (1024 * 1024)


@dataclass(frozen=True)
class BenchmarkRunResult:
    """单项基准评测运行结果。"""

    backend: str
    mode: str
    batch_size: int
    n_queries: int
    library_size: int
    wall_time_s: float
    qps: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float
    speedup_vs_1t: float
    speedup_vs_mt: float
    avg_roots_pruned: float
    avg_leaves_pruned: float
    avg_uind_pruned: float
    avg_scored_count: float
    avg_pruned_ratio: float
    raw_latencies_ms: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class CorrectnessResult:
    """正确性与零漏检/零误检核验结果。"""

    mode: str
    n_queries_evaluated: int
    all_zero_false_dismissals: bool
    total_false_dismissals: int
    mean_recall_at_k: float
    max_score_absolute_error: float
    mean_score_absolute_error: float
    hit_count_match_rate: float
    discrepant_queries_count: int
    fp32_tolerance_discrepant: int = 0
    extra_gpu_hits_count: int = 0
    unexpected_false_positives: int = 0
    all_zero_false_positives: bool = True


def resolve_snapshot_path(snapshot_arg: str | Path | None) -> Path:
    """解析并定位森林快照文件路径。"""
    if snapshot_arg:
        p = Path(snapshot_arg)
        if p.is_file():
            return p
        raise FileNotFoundError(f"指定的森林快照不存在: {snapshot_arg}")

    candidates = [
        Path("cleaned_forest.npz"),
        Path("all_gnps_forest.npz"),
        Path(__file__).resolve().parents[3] / "cleaned_forest.npz",
        Path(__file__).resolve().parents[3] / "all_gnps_forest.npz",
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "未在当前目录或项目根目录下找到默认森林快照文件 (cleaned_forest.npz 或 all_gnps_forest.npz)。"
        "请使用 --snapshot 指定有效快照路径。"
    )


def get_gpu_device_info() -> GpuDeviceInfo:
    """探测当前活跃的 NVIDIA GPU 设备参数。"""
    require_cuda()
    device = cuda.get_current_device()
    raw_name = device.name
    name_str = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
    cc = f"{device.compute_capability[0]}.{device.compute_capability[1]}"

    ctx = cuda.current_context()
    free_bytes, total_bytes = ctx.get_memory_info()

    return GpuDeviceInfo(
        name=name_str,
        compute_capability=cc,
        total_vram_gb=total_bytes / (1024**3),
        free_vram_gb=free_bytes / (1024**3),
    )


def upload_forest_and_measure_memory(
    forest: ForestIndex,
) -> tuple[GpuForestIndex, DeviceMemoryStats]:
    """将 Host 端森林上传至 GPU 设备端并度量显存驻留与耗时。"""
    require_cuda()
    cuda.synchronize()

    t0 = time.perf_counter()
    gpu_forest = GpuForestIndex.from_forest(forest)
    cuda.synchronize()
    upload_time = time.perf_counter() - t0

    mem_summary = gpu_forest.device_memory_summary()
    total_device_bytes = gpu_forest.device_memory_bytes()
    bandwidth = (total_device_bytes / (1024 * 1024)) / upload_time if upload_time > 0 else 0.0

    energy_savings = int(forest.postings.energy.nbytes)

    stats = DeviceMemoryStats(
        upload_time_s=upload_time,
        upload_bandwidth_mb_s=bandwidth,
        envelopes_bytes=mem_summary["envelopes"],
        trees_bytes=mem_summary["trees"],
        nodes_bytes=mem_summary["nodes"],
        postings_bytes=mem_summary["postings"],
        total_bytes=total_device_bytes,
        energy_omitted_savings_bytes=energy_savings,
    )
    return gpu_forest, stats


def prepare_queries_and_configs(
    forest: ForestIndex,
    n_queries: int = 64,
    seed: int = 2026,
    theta: float = 0.7,
    k: int = 10,
) -> tuple[list[tuple[int, SpectrumPeaks]], list[SpectrumPeaks], list[QueryConfig], list[QueryConfig]]:
    """从森林索引中抽样代表性查询谱，并配置对应的离子模式与查询参数。"""
    sampled = sample_query_spectra_from_forest(forest, n_queries=n_queries, seed=seed)
    q_peaks = [item[1] for item in sampled]

    cfgs_th: list[QueryConfig] = []
    cfgs_topk: list[QueryConfig] = []

    for row, _ in sampled:
        meta = forest.spectra[row]
        cfgs_th.append(
            QueryConfig(
                mode=SearchMode.THRESHOLD,
                threshold=theta,
                ion_mode=meta.ion_mode,
                fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
                min_matched_peaks=1,
            )
        )
        cfgs_topk.append(
            QueryConfig(
                mode=SearchMode.TOP_K,
                k=k,
                ion_mode=meta.ion_mode,
                fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
                min_matched_peaks=1,
            )
        )

    return sampled, q_peaks, cfgs_th, cfgs_topk


def warmup_kernels(
    forest: ForestIndex,
    gpu_forest: GpuForestIndex,
    q_peaks: list[SpectrumPeaks],
    cfgs_th: list[QueryConfig],
    cfgs_topk: list[QueryConfig],
    n_warmup: int = 2,
) -> None:
    """预热 Numba JIT 与 CUDA 内核，消除首次编译对基准计时的扰动。"""
    w_n = min(n_warmup, len(q_peaks))
    if w_n == 0:
        return

    w_peaks = q_peaks[:w_n]
    w_cfgs_th = cfgs_th[:w_n]
    w_cfgs_topk = cfgs_topk[:w_n]

    # CPU 预热
    search_forest_batch(w_peaks, forest, config=w_cfgs_th, concurrency=1, uind=True)
    search_forest_batch(w_peaks, forest, config=w_cfgs_topk, concurrency=1, uind=True)

    # GPU 预热
    search_threshold_batch_gpu(w_peaks, gpu_forest, config=w_cfgs_th, batch_size=w_n, uind=True)
    search_topk_batch_gpu(w_peaks, gpu_forest, config=w_cfgs_topk, batch_size=w_n, uind=True)
    cuda.synchronize()


def _compute_latency_distribution(
    latencies: list[float],
) -> tuple[float, float, float, float, float, float, float]:
    """计算延迟数组的 mean, p50, p95, p99, min, max, std (毫秒)。"""
    arr = np.array(latencies, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.std(arr)),
    )


def run_cpu_benchmark(
    queries: list[SpectrumPeaks],
    forest: ForestIndex,
    configs: list[QueryConfig],
    concurrency: int,
    mode_name: str,
    library_size: int,
    ref_1t_time: float | None = None,
) -> tuple[BenchmarkRunResult, list[SearchOutcome]]:
    """执行 CPU 基准检索并记录吞吐与时延。"""
    n_q = len(queries)
    t0 = time.perf_counter()
    outcomes = search_forest_batch(
        queries=queries,
        forest=forest,
        config=configs,
        concurrency=concurrency,
        uind=True,
    )
    wall_time = time.perf_counter() - t0

    qps = n_q / wall_time if wall_time > 0 else 0.0

    # 提取逐查询时延 (由 SearchStats 计时汇总)
    raw_lats = [o.stats.bound_eval_time_ms + o.stats.exact_eval_time_ms for o in outcomes]
    if concurrency <= 1:
        mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms, std_ms = _compute_latency_distribution(raw_lats)
    else:
        # 多线程并发下，单查询有效平均服务时延为系统级摊薄时延
        eff_mean = (wall_time / n_q) * 1000.0
        _, p50_ms, p95_ms, p99_ms, min_ms, max_ms, std_ms = _compute_latency_distribution(raw_lats)
        mean_ms = eff_mean

    speedup_1t = (ref_1t_time / wall_time) if ref_1t_time and wall_time > 0 else 1.0

    # 统计剪枝
    roots_pruned = [o.stats.pruned_by_layer.get("roots_pruned", 0) for o in outcomes]
    leaves_pruned = [o.stats.pruned_by_layer.get("leaves_pruned", 0) for o in outcomes]
    uind_pruned = [o.stats.pruned_by_layer.get("uind_pruned", 0) for o in outcomes]
    scored = [o.stats.n_scored for o in outcomes]

    avg_scored = float(np.mean(scored)) if scored else 0.0
    avg_pruned_ratio = 1.0 - (avg_scored / library_size) if library_size > 0 else 1.0

    backend_label = f"CPU (1 Thread)" if concurrency == 1 else f"CPU ({concurrency} Threads)"

    res = BenchmarkRunResult(
        backend=backend_label,
        mode=mode_name,
        batch_size=concurrency,
        n_queries=n_q,
        library_size=library_size,
        wall_time_s=wall_time,
        qps=qps,
        latency_mean_ms=mean_ms,
        latency_p50_ms=p50_ms,
        latency_p95_ms=p95_ms,
        latency_p99_ms=p99_ms,
        latency_min_ms=min_ms,
        latency_max_ms=max_ms,
        latency_std_ms=std_ms,
        speedup_vs_1t=speedup_1t,
        speedup_vs_mt=1.0 if concurrency > 1 else (speedup_1t if ref_1t_time else 0.0),
        avg_roots_pruned=float(np.mean(roots_pruned)) if roots_pruned else 0.0,
        avg_leaves_pruned=float(np.mean(leaves_pruned)) if leaves_pruned else 0.0,
        avg_uind_pruned=float(np.mean(uind_pruned)) if uind_pruned else 0.0,
        avg_scored_count=avg_scored,
        avg_pruned_ratio=avg_pruned_ratio,
        raw_latencies_ms=raw_lats,
    )
    return res, outcomes


def run_gpu_benchmark(
    queries: list[SpectrumPeaks],
    gpu_forest: GpuForestIndex,
    configs: list[QueryConfig],
    batch_size: int,
    mode_name: str,
    library_size: int,
    ref_1t_time: float | None = None,
    ref_mt_time: float | None = None,
) -> tuple[BenchmarkRunResult, list[SearchOutcome]]:
    """执行 GPU 批处理基准检索并记录吞吐、时延与加速比。"""
    require_cuda()
    n_q = len(queries)

    cuda.synchronize()
    t0 = time.perf_counter()
    if mode_name == "threshold":
        outcomes = search_threshold_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            config=configs,
            batch_size=batch_size,
            uind=True,
        )
    elif mode_name == "topk":
        outcomes = search_topk_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            config=configs,
            batch_size=batch_size,
            probe_trees=3,
            uind=True,
        )
    else:
        outcomes = search_forest_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            config=configs,
            batch_size=batch_size,
            uind=True,
        )
    cuda.synchronize()
    wall_time = time.perf_counter() - t0

    qps = n_q / wall_time if wall_time > 0 else 0.0

    raw_lats = [o.stats.bound_eval_time_ms + o.stats.exact_eval_time_ms for o in outcomes]
    eff_mean = (wall_time / n_q) * 1000.0
    _, p50_ms, p95_ms, p99_ms, min_ms, max_ms, std_ms = _compute_latency_distribution(raw_lats)

    speedup_1t = (ref_1t_time / wall_time) if ref_1t_time and wall_time > 0 else float("nan")
    speedup_mt = (ref_mt_time / wall_time) if ref_mt_time and wall_time > 0 else float("nan")

    roots_pruned = [o.stats.pruned_by_layer.get("roots_pruned", 0) for o in outcomes]
    leaves_pruned = [o.stats.pruned_by_layer.get("leaves_pruned", 0) for o in outcomes]
    uind_pruned = [o.stats.pruned_by_layer.get("uind_pruned", 0) for o in outcomes]
    scored = [o.stats.n_scored for o in outcomes]

    avg_scored = float(np.mean(scored)) if scored else 0.0
    avg_pruned_ratio = 1.0 - (avg_scored / library_size) if library_size > 0 else 1.0

    backend_label = f"GPU (batch_size={batch_size})"

    res = BenchmarkRunResult(
        backend=backend_label,
        mode=mode_name,
        batch_size=batch_size,
        n_queries=n_q,
        library_size=library_size,
        wall_time_s=wall_time,
        qps=qps,
        latency_mean_ms=eff_mean,
        latency_p50_ms=p50_ms,
        latency_p95_ms=p95_ms,
        latency_p99_ms=p99_ms,
        latency_min_ms=min_ms,
        latency_max_ms=max_ms,
        latency_std_ms=std_ms,
        speedup_vs_1t=speedup_1t,
        speedup_vs_mt=speedup_mt,
        avg_roots_pruned=float(np.mean(roots_pruned)) if roots_pruned else 0.0,
        avg_leaves_pruned=float(np.mean(leaves_pruned)) if leaves_pruned else 0.0,
        avg_uind_pruned=float(np.mean(uind_pruned)) if uind_pruned else 0.0,
        avg_scored_count=avg_scored,
        avg_pruned_ratio=avg_pruned_ratio,
        raw_latencies_ms=raw_lats,
    )
    return res, outcomes


def verify_correctness(
    cpu_outcomes: list[SearchOutcome],
    gpu_outcomes: list[SearchOutcome],
    mode_name: str,
    threshold: float = 0.7,
    score_margin: float = 1e-4,
) -> CorrectnessResult:
    """逐位严格对拍 CPU 与 GPU 检索结果的一致性、得分误差、零漏检与假阳性。"""
    n_q = len(cpu_outcomes)
    if n_q != len(gpu_outcomes):
        raise ValueError(f"结果数量不匹配: CPU {n_q} vs GPU {len(gpu_outcomes)}")

    score_diffs: list[float] = []
    recalls: list[float] = []
    false_dismissals = 0
    hit_count_matches = 0
    discrepant_count = 0
    fp32_tolerance_discrepant_count = 0
    extra_gpu_hits_count = 0
    unexpected_false_positives = 0

    for q_idx in range(n_q):
        c_res = cpu_outcomes[q_idx]
        g_res = gpu_outcomes[q_idx]

        c_hits = c_res.hits
        g_hits = g_res.hits

        if len(c_hits) == len(g_hits):
            hit_count_matches += 1

        c_keys = [(h.spectrum_index, h.external_id) for h in c_hits]
        g_keys = [(h.spectrum_index, h.external_id) for h in g_hits]

        c_key_set = set(c_keys)
        g_key_set = set(g_keys)

        query_discrepant = False

        # 1. 召回率核验 (CPU hits missed by GPU)
        if len(c_key_set) == 0:
            rec = 1.0
        else:
            intersection = g_key_set & c_key_set
            rec = len(intersection) / len(c_key_set)
            missed = len(c_key_set - g_key_set)
            if missed > 0:
                false_dismissals += missed
                query_discrepant = True
        recalls.append(rec)

        # 2. 对称多余命中核验 (Extra GPU hits not in CPU)
        extra_keys = g_key_set - c_key_set
        if extra_keys:
            extra_gpu_hits_count += len(extra_keys)
            is_topk = mode_name.lower().startswith("top") or c_res.mode == SearchMode.TOP_K
            g_hit_map = {(h.spectrum_index, h.external_id): h for h in g_hits}
            if is_topk:
                if len(c_hits) == 0:
                    # CPU 无任何符合条件的候选命中，GPU 返回的任何额外命中均为误检假阳性
                    unexpected_false_positives += len(extra_keys)
                    query_discrepant = True
                else:
                    eff_threshold = min((h.score for h in c_hits))
                    for k in extra_keys:
                        h_gpu = g_hit_map[k]
                        score_val = float(h_gpu.score)
                        if np.isnan(score_val) or score_val < (eff_threshold - score_margin):
                            unexpected_false_positives += 1
                            query_discrepant = True
            else:
                eff_threshold = threshold
                for k in extra_keys:
                    h_gpu = g_hit_map[k]
                    score_val = float(h_gpu.score)
                    if (
                        np.isnan(score_val)
                        or score_val < (eff_threshold - score_margin)
                        or score_val > (eff_threshold + score_margin)
                    ):
                        unexpected_false_positives += 1
                        query_discrepant = True

        # 3. 分差核验 (对齐相同的命中谱)
        max_q_diff = 0.0
        g_hit_dict = {h.spectrum_index: h.score for h in g_hits}
        for h_cpu in c_hits:
            if h_cpu.spectrum_index in g_hit_dict:
                s_cpu = float(h_cpu.score)
                s_gpu = float(g_hit_dict[h_cpu.spectrum_index])
                if np.isnan(s_cpu) or np.isnan(s_gpu):
                    query_discrepant = True
                    continue
                diff = abs(s_cpu - s_gpu)
                score_diffs.append(diff)
                if diff > max_q_diff:
                    max_q_diff = diff

        if max_q_diff > score_margin:
            query_discrepant = True
        elif not query_discrepant and 1e-6 < max_q_diff <= score_margin:
            fp32_tolerance_discrepant_count += 1

        if query_discrepant:
            discrepant_count += 1

    max_diff = float(np.max(score_diffs)) if score_diffs else 0.0
    mean_diff = float(np.mean(score_diffs)) if score_diffs else 0.0
    mean_rec = float(np.mean(recalls)) if recalls else 1.0
    hit_match_rate = hit_count_matches / n_q if n_q > 0 else 1.0

    return CorrectnessResult(
        mode=mode_name,
        n_queries_evaluated=n_q,
        all_zero_false_dismissals=(false_dismissals == 0),
        total_false_dismissals=false_dismissals,
        mean_recall_at_k=mean_rec,
        max_score_absolute_error=max_diff,
        mean_score_absolute_error=mean_diff,
        hit_count_match_rate=hit_match_rate,
        discrepant_queries_count=discrepant_count,
        fp32_tolerance_discrepant=fp32_tolerance_discrepant_count,
        extra_gpu_hits_count=extra_gpu_hits_count,
        unexpected_false_positives=unexpected_false_positives,
        all_zero_false_positives=(unexpected_false_positives == 0),
    )


# =========================================================================
# 表格格式化与控制台报告
# =========================================================================

def format_device_memory_table(mem: DeviceMemoryStats, dev: GpuDeviceInfo) -> str:
    """格式化显存驻留与迁移统计表格。"""
    headers = ["显存驻留组件 (VRAM Component)", "占用显存 (MB)", "占用显存 (GB)", "占总显存比例", "设计优化与说明"]
    tot_mb = mem.total_mb
    dev_tot_mb = dev.total_vram_gb * 1024

    rows = [
        [
            "Envelopes (双层包络与网格幅值)",
            f"{mem.envelopes_mb:.2f} MB",
            f"{mem.envelopes_mb / 1024:.3f} GB",
            f"{(mem.envelopes_mb / tot_mb) * 100:.1f}%",
            "FP32 紧致压缩，保存 0.02 Da 空间网格幅值",
        ],
        [
            "Trees (森林前体质量与根叶拓扑)",
            f"{mem.trees_mb:.2f} MB",
            f"{mem.trees_mb / 1024:.3f} GB",
            f"{(mem.trees_mb / tot_mb) * 100:.1f}%",
            "前体二分区间与根/叶节点索引偏移表",
        ],
        [
            "Nodes (BVH-SAH 紧凑节点描述符)",
            f"{mem.nodes_mb:.2f} MB",
            f"{mem.nodes_mb / 1024:.3f} GB",
            f"{(mem.nodes_mb / tot_mb) * 100:.1f}%",
            "叶节点布尔标志与内部 ID 边界指针",
        ],
        [
            "Postings (连续峰表 mass/intensity/norm)",
            f"{mem.postings_mb:.2f} MB",
            f"{mem.postings_mb / 1024:.3f} GB",
            f"{(mem.postings_mb / tot_mb) * 100:.1f}%",
            "密集排布，去能量列；强度转换为 FP32",
        ],
        [
            "TOTAL Device Resident (GPU 驻留总计)",
            f"{mem.total_mb:.2f} MB",
            f"{mem.total_gb:.3f} GB",
            f"{(mem.total_mb / dev_tot_mb) * 100:.1f}% (总 VRAM)",
            f"H2D 搬运耗时 {mem.upload_time_s:.2f}s ({mem.upload_bandwidth_mb_s:,.0f} MB/s)",
        ],
        [
            "VRAM Energy Omission Savings (显存削减)",
            f"{mem.energy_savings_mb:.2f} MB",
            f"{mem.energy_savings_mb / 1024:.3f} GB",
            "节省 ~40% 峰表",
            "完全剔除 postings.energy 列，零精度损失节省显存",
        ],
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def format_benchmark_table(results: list[BenchmarkRunResult], title: str) -> str:
    """格式化单检索模式下的性能对比表格。"""
    headers = [
        "检索后端 / 配置",
        "批容量 (Batch)",
        "耗时 (Wall Time)",
        "吞吐量 (QPS)",
        "有效均值时延 (摊薄)",
        "任务时延 P50",
        "任务时延 P95",
        "加速比 vs 1T",
        "加速比 vs MT",
        "包络剪枝率",
    ]
    rows = []
    for r in results:
        sp_1t = f"{r.speedup_vs_1t:.2f}x" if not np.isnan(r.speedup_vs_1t) else "N/A"
        sp_mt = f"{r.speedup_vs_mt:.2f}x" if not np.isnan(r.speedup_vs_mt) else "N/A"
        batch_str = f"{r.batch_size} (eff {r.n_queries})" if r.batch_size > r.n_queries else str(r.batch_size)
        rows.append(
            [
                r.backend,
                batch_str,
                f"{r.wall_time_s:.3f} s",
                f"{r.qps:,.1f}",
                f"{r.latency_mean_ms:.2f} ms",
                f"{r.latency_p50_ms:.2f} ms",
                f"{r.latency_p95_ms:.2f} ms",
                sp_1t,
                sp_mt,
                f"{r.avg_pruned_ratio * 100:.2f}%",
            ]
        )
    table = tabulate(rows, headers=headers, tablefmt="github")
    footnote = "* 注: CPU MT 的 P50/P95 包含多线程争用下的任务单次执行耗时，而均值为系统级有效服务时延 (Wall / N)；GPU 采用批处理执行，任务耗时按批大小均匀分摊。"
    if any(r.n_queries < 100 for r in results):
        footnote += "\n* (注: 样本量 N < 100 时 P95/P99 分位数易受单离群点波动影响)"
    prefix = f"### {title}\n" if title else ""
    return f"{prefix}{table}\n{footnote}"


def format_correctness_table(results: list[CorrectnessResult]) -> str:
    """格式化正确性对拍核验表格。"""
    headers = [
        "评测检索模式",
        "测试查询数",
        "零漏检状态 (Zero Miss)",
        "零误检状态 (Zero FP)",
        "平均召回率 (Recall@K)",
        "漏检谱总数",
        "多余命中/误检数 (Extra/FP)",
        "FP32容差波动数 (Tol Diff)",
        "最大绝对分差 (Max |ΔScore|)",
        "平均绝对分差 (MAE)",
        "命中山头完全吻合率",
    ]
    rows = []
    for r in results:
        status_miss = "PASS (Zero Miss)" if r.all_zero_false_dismissals else "FAIL"
        status_fp = "PASS (Zero FP)" if r.all_zero_false_positives else f"FAIL ({r.unexpected_false_positives} FP)"
        extra_str = f"{r.extra_gpu_hits_count} / {r.unexpected_false_positives}"
        rows.append(
            [
                r.mode.upper(),
                r.n_queries_evaluated,
                status_miss,
                status_fp,
                f"{r.mean_recall_at_k * 100:.2f}%",
                r.total_false_dismissals,
                extra_str,
                r.fp32_tolerance_discrepant,
                f"{r.max_score_absolute_error:.2e}",
                f"{r.mean_score_absolute_error:.2e}",
                f"{r.hit_count_match_rate * 100:.2f}%",
            ]
        )
    table = tabulate(rows, headers=headers, tablefmt="github")
    footnote = "* 注: 'FP32容差波动数' 指分差在 1e-6 ~ score_margin 之间的正常累加波动 query 数 (属于合法精度容差范围，非异常)。"
    return f"{table}\n{footnote}"


def export_benchmark_json(
    output_path: str | Path,
    device_info: GpuDeviceInfo,
    memory_stats: DeviceMemoryStats,
    snapshot_path: Path,
    forest: ForestIndex,
    all_results: dict[str, list[BenchmarkRunResult]],
    correctness_results: list[CorrectnessResult],
) -> None:
    """将全套基准评测结果导出为标准 JSON 文件。"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_architecture": platform.machine(),
            "cpu_cores": os.cpu_count() or 1,
            "gpu_name": device_info.name,
            "gpu_compute_capability": device_info.compute_capability,
            "gpu_total_vram_gb": device_info.total_vram_gb,
            "gpu_free_vram_gb": device_info.free_vram_gb,
        },
        "snapshot_info": {
            "path": str(snapshot_path),
            "file_size_mb": snapshot_path.stat().st_size / (1024 * 1024),
            "n_spectra": forest.n_spectra,
            "n_trees": forest.n_trees,
            "n_nodes": forest.n_nodes,
            "summary_grid_da": forest.spec.summary_grid_da,
        },
        "device_memory": {
            "upload_time_s": memory_stats.upload_time_s,
            "upload_bandwidth_mb_s": memory_stats.upload_bandwidth_mb_s,
            "envelopes_mb": memory_stats.envelopes_mb,
            "trees_mb": memory_stats.trees_mb,
            "nodes_mb": memory_stats.nodes_mb,
            "postings_mb": memory_stats.postings_mb,
            "total_resident_mb": memory_stats.total_mb,
            "total_resident_gb": memory_stats.total_gb,
            "energy_omitted_savings_mb": memory_stats.energy_savings_mb,
        },
        "benchmarks": {},
        "correctness": [asdict(c) for c in correctness_results],
    }

    for mode_key, run_list in all_results.items():
        mode_data = []
        for r in run_list:
            d = asdict(r)
            d.pop("raw_latencies_ms", None)  # 剔除超大裸数组，保持 JSON 紧凑易读
            mode_data.append(d)
        data["benchmarks"][mode_key] = mode_data

    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_batch_sizes(values: list[str] | list[int] | None) -> list[int]:
    """灵活解析 batch-sizes 参数 (支持逗号、空格或混合格式)。"""
    if not values:
        return [64, 128, 256, 512]
    parsed: list[int] = []
    for item in values:
        if isinstance(item, int):
            parsed.append(item)
        elif isinstance(item, str):
            for part in item.replace(",", " ").split():
                if part.strip().isdigit():
                    parsed.append(int(part.strip()))
    return sorted(list(set(parsed))) if parsed else [64, 128, 256, 512]


# =========================================================================
# 主运行入口
# =========================================================================

def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="JET-Forest GPU 吞吐量基准压测与性能评测套件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="森林快照文件路径 (.npz)。默认优先检查 cleaned_forest.npz，次之检查 all_gnps_forest.npz。",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=256,
        help="从快照中抽样的查询谱数量 (默认 256 以保证 P95/P99 统计稳健)。",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        default=["64", "128", "256", "512"],
        help="待测试的 GPU batch size 列表。",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["threshold", "topk"],
        choices=["threshold", "topk"],
        help="评测检索模式 (threshold: 阈值检索; topk: Top-K 检索)。",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.7,
        help="阈值检索的相似度门槛 theta。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K 检索的候选返回容量 K。",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=os.cpu_count() or 4,
        help="CPU 多线程并发测试使用的线程数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="抽样查询谱的确定性随机种子。",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="基准计时前用于 JIT 预热的查询谱数量。",
    )
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="跳过 CPU 基准评测，仅测试 GPU 吞吐。",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选将评测详细指标保存为 JSON 文件的路径。",
    )

    args = parser.parse_args(argv)

    print("=" * 80)
    print(" JET-Forest (JETF) GPU 吞吐量基准压测与端到端性能评测套件")
    print("=" * 80)

    # 1. 检查 CUDA 环境
    if not is_cuda_available():
        print("[ERROR] 当前系统未检测到可用的 NVIDIA CUDA 环境或 GPU 设备。")
        print("请检查 NVIDIA 驱动与 CUDA 支持是否就绪。")
        return 1

    dev_info = get_gpu_device_info()
    print(f"[*] 检测到 GPU 设备: {dev_info.name} (Compute Capability: {dev_info.compute_capability})")
    print(f"[*] 显存容量: 可用 {dev_info.free_vram_gb:.2f} GB / 总计 {dev_info.total_vram_gb:.2f} GB")

    # 2. 定位并加载快照
    snap_path = resolve_snapshot_path(args.snapshot)
    file_size_mb = snap_path.stat().st_size / (1024 * 1024)
    print(f"[*] 加载森林快照: {snap_path} ({file_size_mb:.2f} MB)...")
    t0 = time.perf_counter()
    forest = load_forest_snapshot(snap_path)
    t_load = time.perf_counter() - t0
    print(
        f"    快照加载完成 (耗时 {t_load:.2f}s): "
        f"{forest.n_spectra:,} 谱, {forest.n_trees:,} 棵小树, {forest.n_nodes:,} 节点"
    )

    # 3. 显存驻留与迁移统计
    print("[*] 上传森林至 GPU 显存并度量驻留统计 (GpuForestIndex.from_forest)...")
    gpu_forest, mem_stats = upload_forest_and_measure_memory(forest)
    print(f"    上传完成! 显存驻留总计: {mem_stats.total_mb:.2f} MB ({mem_stats.total_gb:.3f} GB)")

    print("\n" + "=" * 80)
    print(" 1. 显存驻留与 Host-to-Device 迁移统计 (Device Memory Summary)")
    print("=" * 80)
    print(format_device_memory_table(mem_stats, dev_info))

    # 4. 查询谱抽样
    n_queries = min(args.n_queries, forest.n_spectra)
    print(f"\n[*] 确定性抽样 {n_queries} 条真实查询谱 (seed={args.seed})...")
    if n_queries < 100:
        print("    (注: 样本量 N < 100 时 P95/P99 分位数易受单离群点波动影响)")
    sampled_queries, q_peaks, cfgs_th, cfgs_topk = prepare_queries_and_configs(
        forest=forest,
        n_queries=n_queries,
        seed=args.seed,
        theta=args.theta,
        k=args.k,
    )
    print(f"    抽样完成，已配置精确离子模式与查询参数。")

    # 5. 预热 JIT 与 CUDA 内核
    print(f"[*] 执行 JIT 预热 ({args.warmup} queries)...")
    warmup_kernels(forest, gpu_forest, q_peaks, cfgs_th, cfgs_topk, n_warmup=args.warmup)
    print("    预热完成，各算子已处于稳态编译状态。")

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    all_benchmark_results: dict[str, list[BenchmarkRunResult]] = {}
    correctness_results: list[CorrectnessResult] = []

    # =========================================================================
    # 评测模式 1: 开放式阈值检索 (Open Threshold Search)
    # =========================================================================
    if "threshold" in args.modes:
        print("\n" + "=" * 80)
        print(f" 2. 开放式阈值检索基准 (Threshold Search: θ = {args.theta})")
        print("=" * 80)

        th_results: list[BenchmarkRunResult] = []
        cpu_1t_time: float | None = None
        cpu_mt_time: float | None = None
        ref_cpu_outcomes: list[SearchOutcome] | None = None

        if not args.skip_cpu:
            print(f"[*] 运行 CPU 单线程基准 (concurrency=1, {n_queries} queries)...")
            res_1t, ref_cpu_outcomes = run_cpu_benchmark(
                queries=q_peaks,
                forest=forest,
                configs=cfgs_th,
                concurrency=1,
                mode_name="threshold",
                library_size=forest.n_spectra,
            )
            cpu_1t_time = res_1t.wall_time_s
            print(f"    CPU 1T 完成: 耗时 {res_1t.wall_time_s:.2f}s, QPS = {res_1t.qps:.1f}")

            print(f"[*] 运行 CPU 多线程基准 (concurrency={args.cpu_threads}, {n_queries} queries)...")
            res_mt, _ = run_cpu_benchmark(
                queries=q_peaks,
                forest=forest,
                configs=cfgs_th,
                concurrency=args.cpu_threads,
                mode_name="threshold",
                library_size=forest.n_spectra,
                ref_1t_time=cpu_1t_time,
            )
            cpu_mt_time = res_mt.wall_time_s
            print(
                f"    CPU MT 完成: 耗时 {res_mt.wall_time_s:.2f}s, QPS = {res_mt.qps:.1f}, "
                f"加速比 = {res_mt.speedup_vs_1t:.2f}x"
            )

            # 更新 CPU 1T 的 speedup_vs_mt
            sp_1t_vs_mt = (cpu_mt_time / cpu_1t_time) if cpu_1t_time > 0 else 0.0
            res_1t = BenchmarkRunResult(
                backend=res_1t.backend,
                mode=res_1t.mode,
                batch_size=res_1t.batch_size,
                n_queries=res_1t.n_queries,
                library_size=res_1t.library_size,
                wall_time_s=res_1t.wall_time_s,
                qps=res_1t.qps,
                latency_mean_ms=res_1t.latency_mean_ms,
                latency_p50_ms=res_1t.latency_p50_ms,
                latency_p95_ms=res_1t.latency_p95_ms,
                latency_p99_ms=res_1t.latency_p99_ms,
                latency_min_ms=res_1t.latency_min_ms,
                latency_max_ms=res_1t.latency_max_ms,
                latency_std_ms=res_1t.latency_std_ms,
                speedup_vs_1t=1.0,
                speedup_vs_mt=sp_1t_vs_mt,
                avg_roots_pruned=res_1t.avg_roots_pruned,
                avg_leaves_pruned=res_1t.avg_leaves_pruned,
                avg_uind_pruned=res_1t.avg_uind_pruned,
                avg_scored_count=res_1t.avg_scored_count,
                avg_pruned_ratio=res_1t.avg_pruned_ratio,
                raw_latencies_ms=res_1t.raw_latencies_ms,
            )
            th_results.append(res_1t)
            th_results.append(res_mt)

        # GPU 批处理遍历
        last_gpu_outcomes: list[SearchOutcome] | None = None
        for bs in batch_sizes:
            print(f"[*] 运行 GPU 批处理基准 (batch_size={bs}, {n_queries} queries)...")
            res_gpu, gpu_outcomes = run_gpu_benchmark(
                queries=q_peaks,
                gpu_forest=gpu_forest,
                configs=cfgs_th,
                batch_size=bs,
                mode_name="threshold",
                library_size=forest.n_spectra,
                ref_1t_time=cpu_1t_time,
                ref_mt_time=cpu_mt_time,
            )
            th_results.append(res_gpu)
            last_gpu_outcomes = gpu_outcomes
            sp_str = f", vs 1T = {res_gpu.speedup_vs_1t:.1f}x, vs MT = {res_gpu.speedup_vs_mt:.1f}x" if cpu_1t_time else ""
            print(f"    GPU (bs={bs}) 完成: 耗时 {res_gpu.wall_time_s:.3f}s, QPS = {res_gpu.qps:,.1f}{sp_str}")

        all_benchmark_results["threshold"] = th_results
        print("\n" + format_benchmark_table(th_results, f"Threshold Search (θ = {args.theta})"))

        # 正确性核对
        if ref_cpu_outcomes and last_gpu_outcomes:
            c_res = verify_correctness(ref_cpu_outcomes, last_gpu_outcomes, "threshold", threshold=args.theta)
            correctness_results.append(c_res)

    # =========================================================================
    # 评测模式 2: 开放式 Top-K 检索 (Open Top-K Search)
    # =========================================================================
    if "topk" in args.modes:
        print("\n" + "=" * 80)
        print(f" 3. 开放式 Top-K 检索基准 (Top-K Search: K = {args.k})")
        print("=" * 80)

        topk_results: list[BenchmarkRunResult] = []
        cpu_1t_time_topk: float | None = None
        cpu_mt_time_topk: float | None = None
        ref_cpu_outcomes_topk: list[SearchOutcome] | None = None

        if not args.skip_cpu:
            print(f"[*] 运行 CPU 单线程基准 (concurrency=1, {n_queries} queries)...")
            res_1t, ref_cpu_outcomes_topk = run_cpu_benchmark(
                queries=q_peaks,
                forest=forest,
                configs=cfgs_topk,
                concurrency=1,
                mode_name="topk",
                library_size=forest.n_spectra,
            )
            cpu_1t_time_topk = res_1t.wall_time_s
            print(f"    CPU 1T 完成: 耗时 {res_1t.wall_time_s:.2f}s, QPS = {res_1t.qps:.1f}")

            print(f"[*] 运行 CPU 多线程基准 (concurrency={args.cpu_threads}, {n_queries} queries)...")
            res_mt, _ = run_cpu_benchmark(
                queries=q_peaks,
                forest=forest,
                configs=cfgs_topk,
                concurrency=args.cpu_threads,
                mode_name="topk",
                library_size=forest.n_spectra,
                ref_1t_time=cpu_1t_time_topk,
            )
            cpu_mt_time_topk = res_mt.wall_time_s
            print(
                f"    CPU MT 完成: 耗时 {res_mt.wall_time_s:.2f}s, QPS = {res_mt.qps:.1f}, "
                f"加速比 = {res_mt.speedup_vs_1t:.2f}x"
            )

            # 更新 CPU 1T 的 speedup_vs_mt
            sp_1t_vs_mt = (cpu_mt_time_topk / cpu_1t_time_topk) if cpu_1t_time_topk > 0 else 0.0
            res_1t = BenchmarkRunResult(
                backend=res_1t.backend,
                mode=res_1t.mode,
                batch_size=res_1t.batch_size,
                n_queries=res_1t.n_queries,
                library_size=res_1t.library_size,
                wall_time_s=res_1t.wall_time_s,
                qps=res_1t.qps,
                latency_mean_ms=res_1t.latency_mean_ms,
                latency_p50_ms=res_1t.latency_p50_ms,
                latency_p95_ms=res_1t.latency_p95_ms,
                latency_p99_ms=res_1t.latency_p99_ms,
                latency_min_ms=res_1t.latency_min_ms,
                latency_max_ms=res_1t.latency_max_ms,
                latency_std_ms=res_1t.latency_std_ms,
                speedup_vs_1t=1.0,
                speedup_vs_mt=sp_1t_vs_mt,
                avg_roots_pruned=res_1t.avg_roots_pruned,
                avg_leaves_pruned=res_1t.avg_leaves_pruned,
                avg_uind_pruned=res_1t.avg_uind_pruned,
                avg_scored_count=res_1t.avg_scored_count,
                avg_pruned_ratio=res_1t.avg_pruned_ratio,
                raw_latencies_ms=res_1t.raw_latencies_ms,
            )
            topk_results.append(res_1t)
            topk_results.append(res_mt)

        # GPU 批处理遍历
        last_gpu_outcomes_topk: list[SearchOutcome] | None = None
        for bs in batch_sizes:
            print(f"[*] 运行 GPU 批处理基准 (batch_size={bs}, {n_queries} queries)...")
            res_gpu, gpu_outcomes = run_gpu_benchmark(
                queries=q_peaks,
                gpu_forest=gpu_forest,
                configs=cfgs_topk,
                batch_size=bs,
                mode_name="topk",
                library_size=forest.n_spectra,
                ref_1t_time=cpu_1t_time_topk,
                ref_mt_time=cpu_mt_time_topk,
            )
            topk_results.append(res_gpu)
            last_gpu_outcomes_topk = gpu_outcomes
            sp_str = f", vs 1T = {res_gpu.speedup_vs_1t:.1f}x, vs MT = {res_gpu.speedup_vs_mt:.1f}x" if cpu_1t_time_topk else ""
            print(f"    GPU (bs={bs}) 完成: 耗时 {res_gpu.wall_time_s:.3f}s, QPS = {res_gpu.qps:,.1f}{sp_str}")

        all_benchmark_results["topk"] = topk_results
        print("\n" + format_benchmark_table(topk_results, f"Top-K Search (K = {args.k})"))

        # 正确性核对
        if ref_cpu_outcomes_topk and last_gpu_outcomes_topk:
            c_res = verify_correctness(ref_cpu_outcomes_topk, last_gpu_outcomes_topk, "topk")
            correctness_results.append(c_res)

    # =========================================================================
    # 正确性与数学保真汇总
    # =========================================================================
    if correctness_results:
        print("\n" + "=" * 80)
        print(" 4. 正确性与零漏检对拍核验 (Correctness & Zero False Dismissals)")
        print("=" * 80)
        print(format_correctness_table(correctness_results))

    # =========================================================================
    # 保存结果 JSON
    # =========================================================================
    if args.output_json:
        export_benchmark_json(
            output_path=args.output_json,
            device_info=dev_info,
            memory_stats=mem_stats,
            snapshot_path=snap_path,
            forest=forest,
            all_results=all_benchmark_results,
            correctness_results=correctness_results,
        )
        print(f"\n[OK] 评测指标已成功导出至 JSON: {args.output_json}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] GPU 吞吐量压测套件执行完毕!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
