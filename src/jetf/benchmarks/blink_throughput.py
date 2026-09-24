"""JET-Forest 与 BLINK 吞吐量与库规模伸缩性评测模块。

涵盖：
1. 库规模伸缩性评测 (N = [100, 500, 1000, 2000, 5000, 10000])
2. BLINK vs JETF 1-to-N 检索时延与 QPS 对比
3. 对数线性模型外推 (Log-Log Extrapolation to 50k, 100k, 500k, 2M)
4. 真实 200 万级 (all_gnps_forest.npz) JETF 性能测定与对 BLINK 的实际加速比
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from jetf.benchmarks.blink_adapter import BlinkBenchmarkEngine
from jetf.benchmarks.dataset import (
    BenchmarkDataset,
    sample_query_spectra,
    sample_query_spectra_from_forest,
    slice_parsed_library,
)
from jetf.builder import build_forest_index
from jetf.preprocessing import preprocess_library
from jetf.query import IonModePolicy, QueryConfig, SearchMode
from jetf.search import search_forest
from jetf.serialization import load_forest_snapshot
from jetf.structure import DEFAULT_FOREST_SPEC, ForestIndex
from jetf.types import SpectrumPeaks


@dataclass(frozen=True)
class ScalingBenchmarkResult:
    """不同库规模下的检索时延与吞吐量评测结果。"""

    scales: list[int]
    blink_latency_mean_ms: list[float]
    blink_latency_p50_ms: list[float]
    blink_latency_p95_ms: list[float]
    blink_qps: list[float]
    blink_disc_mean_ms: list[float]
    blink_score_mean_ms: list[float]
    jetf_latency_mean_ms: list[float]
    jetf_latency_p50_ms: list[float]
    jetf_latency_p95_ms: list[float]
    jetf_qps: list[float]
    speedup_mean: list[float]


@dataclass(frozen=True)
class ExtrapolationResult:
    """BLINK 线性外推模型与大规模检索预测结果。"""

    measured_scales: list[int]
    measured_blink_latencies_ms: list[float]
    target_scales: list[int]
    blink_extrapolated_latency_ms: list[float]
    blink_extrapolated_qps: list[float]
    alpha: float                  # 对数线性截距: log10(T_ms) = alpha + beta * log10(N)
    beta: float                   # 复杂度阶数指数 (理论上 O(N) -> beta ≈ 1.0)
    r_squared: float              # 拟合优度 R^2
    jetf_2m_measured_latency_ms: float | None = None
    jetf_2m_measured_qps: float | None = None
    jetf_speedup_at_2m: float | None = None


def benchmark_blink_scaling(
    dataset: BenchmarkDataset,
    scales: Sequence[int] = (100, 500, 1000, 2000, 5000, 10000),
    n_queries: int = 50,
    top_k: int = 10,
    tolerance: float = 0.02,
    bin_width: float = 0.001,
) -> ScalingBenchmarkResult:
    """在递增的库规模上测量 BLINK 与 JETF 的 1-to-N 检索时延与 QPS。"""
    max_available = dataset.n_spectra
    effective_scales = [int(s) for s in scales if s <= max_available]
    if not effective_scales:
        effective_scales = [min(max_available, 1000)]

    # 抽样测试查询（多抽取 5 条作为独立预热谱，避免预热混入计时统计）
    queries_info = sample_query_spectra(dataset.library, n_queries=n_queries + 5)
    if len(queries_info) > 5:
        warmup_peaks = [q[1] for q in queries_info[:5]]
        formal_peaks = [q[1] for q in queries_info[5: 5 + n_queries]]
    else:
        warmup_peaks = [q[1] for q in queries_info]
        formal_peaks = [q[1] for q in queries_info]

    b_lat_mean: list[float] = []
    b_lat_p50: list[float] = []
    b_lat_p95: list[float] = []
    b_qps_list: list[float] = []
    b_disc_mean: list[float] = []
    b_score_mean: list[float] = []

    j_lat_mean: list[float] = []
    j_lat_p50: list[float] = []
    j_lat_p95: list[float] = []
    j_qps_list: list[float] = []
    speedup_list: list[float] = []

    q_cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=top_k,
        fragment_tolerance_da=tolerance,
        min_matched_peaks=1,
        ion_mode_policy=IonModePolicy.ANY,
    )

    for n in effective_scales:
        # 切分规模为 n 的子库
        sub_indices = np.arange(n, dtype=np.int64)
        sub_parsed = slice_parsed_library(dataset.parsed, sub_indices)
        sub_lib = preprocess_library(sub_parsed, dataset.library.spec)
        sub_forest = build_forest_index(sub_lib, DEFAULT_FOREST_SPEC)

        # 构建 BLINK 引擎
        engine = BlinkBenchmarkEngine(
            sub_lib, tolerance=tolerance, bin_width=bin_width, intensity_power=0.5
        )

        # 预热（使用专属预热谱，不计入后续正式计时）
        for q in warmup_peaks:
            engine.search_single(q, top_k=top_k)
            search_forest(q, sub_forest, config=q_cfg, library=sub_lib)

        # 1. 测量 BLINK (仅对正式查询计时)
        b_lats: list[float] = []
        b_discs: list[float] = []
        b_scores: list[float] = []
        for q in formal_peaks:
            res = engine.search_single(q, top_k=top_k)
            b_lats.append(res.total_time_s * 1000.0)
            b_discs.append(res.discretize_time_s * 1000.0)
            b_scores.append(res.score_time_s * 1000.0)

        # 2. 测量 JETF (仅对正式查询计时)
        j_lats: list[float] = []
        for q in formal_peaks:
            t0 = time.perf_counter()
            search_forest(q, sub_forest, config=q_cfg, library=sub_lib)
            j_lats.append((time.perf_counter() - t0) * 1000.0)

        b_mean = float(np.mean(b_lats)) if b_lats else 0.0
        j_mean = float(np.mean(j_lats)) if j_lats else 0.0

        b_lat_mean.append(b_mean)
        b_lat_p50.append(float(np.median(b_lats)) if b_lats else 0.0)
        b_lat_p95.append(float(np.percentile(b_lats, 95)) if b_lats else 0.0)
        b_qps_list.append(1000.0 / b_mean if b_mean > 0 else 0.0)
        b_disc_mean.append(float(np.mean(b_discs)) if b_discs else 0.0)
        b_score_mean.append(float(np.mean(b_scores)) if b_scores else 0.0)

        j_lat_mean.append(j_mean)
        j_lat_p50.append(float(np.median(j_lats)) if j_lats else 0.0)
        j_lat_p95.append(float(np.percentile(j_lats, 95)) if j_lats else 0.0)
        j_qps_list.append(1000.0 / j_mean if j_mean > 0 else 0.0)

        speedup_list.append(b_mean / j_mean if j_mean > 0 else 1.0)

    return ScalingBenchmarkResult(
        scales=effective_scales,
        blink_latency_mean_ms=b_lat_mean,
        blink_latency_p50_ms=b_lat_p50,
        blink_latency_p95_ms=b_lat_p95,
        blink_qps=b_qps_list,
        blink_disc_mean_ms=b_disc_mean,
        blink_score_mean_ms=b_score_mean,
        jetf_latency_mean_ms=j_lat_mean,
        jetf_latency_p50_ms=j_lat_p50,
        jetf_latency_p95_ms=j_lat_p95,
        jetf_qps=j_qps_list,
        speedup_mean=speedup_list,
    )


def measure_jetf_snapshot_throughput(
    snapshot_path: str | Path | ForestIndex,
    n_queries: int = 50,
    top_k: int = 10,
    tolerance_da: float = 0.02,
    seed: int = 2026,
) -> tuple[float, float]:
    """在全量 200 万快照上直接实测 JET-Forest 的平均检索时延 (ms) 与 QPS。"""
    if isinstance(snapshot_path, ForestIndex):
        forest = snapshot_path
    else:
        p = Path(snapshot_path)
        if not p.is_file():
            raise FileNotFoundError(f"未找到指定的快照文件: {p}")
        forest = load_forest_snapshot(p)

    sampled = sample_query_spectra_from_forest(forest, n_queries=n_queries + 5, seed=seed)
    if len(sampled) > 5:
        warmup_queries = [q[1] for q in sampled[:5]]
        formal_queries = [q[1] for q in sampled[5: 5 + n_queries]]
    else:
        warmup_queries = [q[1] for q in sampled]
        formal_queries = [q[1] for q in sampled]

    cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=top_k,
        fragment_tolerance_da=tolerance_da,
        min_matched_peaks=1,
        ion_mode_policy=IonModePolicy.ANY,
    )

    # 预热 (使用专属预热谱，不混入后续正式计时)
    for q in warmup_queries:
        search_forest(q, forest, config=cfg)

    latencies_ms: list[float] = []
    for q in formal_queries:
        t0 = time.perf_counter()
        search_forest(q, forest, config=cfg)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    qps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return mean_ms, qps


def extrapolate_blink_throughput(
    measured_scales: Sequence[int],
    measured_latencies_ms: Sequence[float],
    target_scales: Sequence[int] = (50000, 100000, 500000, 2003310),
    jetf_2m_latency_ms: float | None = None,
) -> ExtrapolationResult:
    """基于测量的库规模时延，使用对数线性回归 log10(Time) = alpha + beta * log10(N) 外推更大库规模的时延与 QPS。"""
    scales_arr = np.asarray(measured_scales, dtype=np.float64)
    lats_arr = np.asarray(measured_latencies_ms, dtype=np.float64)
    if scales_arr.shape != lats_arr.shape:
        raise ValueError("measured_scales 与 measured_latencies_ms 长度必须一致")

    valid_mask = (
        np.isfinite(scales_arr)
        & np.isfinite(lats_arr)
        & (scales_arr > 0)
        & (lats_arr > 0)
    )
    if np.sum(valid_mask) < 2:
        raise ValueError("对数线性回归外推至少需要 2 个具有正实数规模与耗时的有效测量点")

    valid_scales = scales_arr[valid_mask]
    valid_lats = lats_arr[valid_mask]

    x = np.log10(valid_scales)
    y = np.log10(valid_lats)

    # 线性多项式拟合: y = beta * x + alpha
    poly_coeffs = np.polyfit(x, y, deg=1)
    beta = float(poly_coeffs[0])
    alpha = float(poly_coeffs[1])

    # 决定系数 R^2
    y_pred = np.polyval(poly_coeffs, x)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r2 = float(np.clip(1.0 - (ss_res / ss_tot), 0.0, 1.0)) if ss_tot > 1e-12 else 1.0

    targets = [int(s) for s in target_scales]
    extrap_latencies: list[float] = []
    extrap_qps: list[float] = []

    for s in targets:
        pred_log = float(np.polyval(poly_coeffs, np.log10(s)))
        pred_ms = float(10.0**pred_log)
        extrap_latencies.append(pred_ms)
        extrap_qps.append(1000.0 / pred_ms if pred_ms > 0 else 0.0)

    jetf_speedup = None
    jetf_2m_qps = None
    if jetf_2m_latency_ms is not None and jetf_2m_latency_ms > 0:
        jetf_2m_qps = 1000.0 / jetf_2m_latency_ms
        blink_2m_ms = float(10.0 ** (alpha + beta * np.log10(2003310)))
        jetf_speedup = float(blink_2m_ms / jetf_2m_latency_ms)

    return ExtrapolationResult(
        measured_scales=[int(s) for s in valid_scales],
        measured_blink_latencies_ms=[float(lat) for lat in valid_lats],
        target_scales=targets,
        blink_extrapolated_latency_ms=extrap_latencies,
        blink_extrapolated_qps=extrap_qps,
        alpha=alpha,
        beta=beta,
        r_squared=r2,
        jetf_2m_measured_latency_ms=jetf_2m_latency_ms,
        jetf_2m_measured_qps=jetf_2m_qps,
        jetf_speedup_at_2m=jetf_speedup,
    )
