"""JET-Forest 与 matchms 吞吐量与耗时基准评测模块。

涵盖：
1. 单对谱纯算子微基准 (Pairwise Kernel Microbenchmark)
2. 端到端 1-to-N 库检索宏基准 (1-to-N Library Retrieval Macrobenchmark)
3. 库规模伸缩性评测 (Scalability Benchmark)
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Sequence

import numpy as np

from jetf.benchmarks.adapter import check_matchms_available, jetf_peaks_to_matchms
from jetf.benchmarks.dataset import BenchmarkDataset, load_benchmark_dataset, sample_query_spectra
from jetf.bounds import adaptive_numba_threads
from jetf.query import IonModePolicy, QueryConfig, SearchMode, is_eligible
from jetf.scoring import score_greedy_cosine
from jetf.search import search_forest
from jetf.structure import ForestIndex
from jetf.types import DEFAULT_FRAGMENT_TOLERANCE_DA, IonMode, SpectrumPeaks


@dataclass(frozen=True)
class PairwiseThroughputResult:
    """逐对谱评分算子吞吐量与耗时统计。"""

    n_pairs: int
    jetf_pairs_per_sec: float
    matchms_pairs_per_sec: float
    jetf_avg_time_us: float
    matchms_avg_time_us: float
    speedup: float


@dataclass(frozen=True)
class RetrievalThroughputResult:
    """1-to-N 库检索宏观吞吐量与时延统计。

    指标口径与语义说明:
    - jetf_qps / matchms_qps: 系统级每秒处理查询数 (QPS = n_queries / wall_duration_s)。
    - jetf_latency_mean_ms: 系统级有效服务均值时延 (摊薄时延, wall_duration_s / n_queries * 1000)。
      注: 多线程并发 (concurrency > 1) 下，均值为系统级有效服务时延；单任务单次耗时由 P50/P95/P99 体现。
    - jetf_latency_p50_ms / p95_ms / p99_ms: 任务时延分位数 (包含线程争用与调度开销)。
    - speedup: 均值加速比 (matchms_mean / jetf_mean)。
    - throughput_speedup: 系统级吞吐加速比 (jetf_qps / matchms_qps)。
    """

    mode_name: str
    n_queries: int
    library_size: int
    jetf_qps: float
    matchms_qps: float
    jetf_latency_mean_ms: float
    jetf_latency_p50_ms: float
    jetf_latency_p95_ms: float
    jetf_latency_p99_ms: float
    matchms_latency_mean_ms: float
    matchms_latency_p50_ms: float
    matchms_latency_p95_ms: float
    matchms_latency_p99_ms: float
    speedup: float
    avg_scored_ratio: float
    avg_pruned_ratio: float
    jetf_latency_std_ms: float = 0.0
    matchms_latency_std_ms: float = 0.0
    concurrency: int = 1
    throughput_speedup: float = 0.0


def benchmark_pairwise_throughput(
    pairs: Sequence[tuple[SpectrumPeaks, SpectrumPeaks]],
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
    warmup: int = 20,
    n_repeats: int = 3,
) -> PairwiseThroughputResult:
    """测量逐对谱评分内核的每秒吞吐量与微秒耗时。"""
    check_matchms_available()
    from matchms.similarity import CosineGreedy

    scorer = CosineGreedy(tolerance=tolerance_da, mz_power=0.0, intensity_power=1.0)
    n = len(pairs)
    if n == 0:
        raise ValueError("pairs 列表不能为空")

    # 预准备 matchms 格式数据
    m_pairs = [(jetf_peaks_to_matchms(p1), jetf_peaks_to_matchms(p2)) for p1, p2 in pairs]

    # 1. 预热 (使 Numba JIT 编译完成)
    for i in range(min(warmup, n)):
        score_greedy_cosine(pairs[i][0], pairs[i][1], tolerance_da=tolerance_da)
        scorer.pair(m_pairs[i][0], m_pairs[i][1])

    # 2. 测量 JET-Forest (多轮重复，取中位数降低抖动干扰)
    jetf_times = []
    for _ in range(max(1, n_repeats)):
        t0 = time.perf_counter()
        for p1, p2 in pairs:
            score_greedy_cosine(p1, p2, tolerance_da=tolerance_da)
        jetf_times.append(time.perf_counter() - t0)

    # 3. 测量 matchms
    matchms_times = []
    for _ in range(max(1, n_repeats)):
        t0 = time.perf_counter()
        for s1, s2 in m_pairs:
            scorer.pair(s1, s2)
        matchms_times.append(time.perf_counter() - t0)

    t_jetf_rep = float(np.median(jetf_times))
    t_matchms_rep = float(np.median(matchms_times))

    jetf_avg_us = (t_jetf_rep / n) * 1e6
    matchms_avg_us = (t_matchms_rep / n) * 1e6
    jetf_rate = n / t_jetf_rep if t_jetf_rep > 0 else 0.0
    matchms_rate = n / t_matchms_rep if t_matchms_rep > 0 else 0.0
    speedup = matchms_avg_us / jetf_avg_us if jetf_avg_us > 0 else 0.0

    return PairwiseThroughputResult(
        n_pairs=n,
        jetf_pairs_per_sec=jetf_rate,
        matchms_pairs_per_sec=matchms_rate,
        jetf_avg_time_us=jetf_avg_us,
        matchms_avg_time_us=matchms_avg_us,
        speedup=speedup,
    )


def benchmark_retrieval_throughput(
    dataset: BenchmarkDataset | ForestIndex,
    queries: Sequence[tuple[int, SpectrumPeaks]] | Sequence[tuple[int, SpectrumPeaks, QueryConfig]],
    config: QueryConfig | Sequence[QueryConfig] | None = None,
    mode_name: str = "custom",
    warmup: int = 3,
    skip_matchms: bool = False,
    concurrency: int = 1,
) -> RetrievalThroughputResult:
    """评测 1-to-N 库检索的 QPS 与时延指标。支持脱机 ForestIndex 快照与可选跳过 matchms 穷举。"""
    import warnings

    if not queries:
        raise ValueError("queries 列表不能为空")

    if isinstance(dataset, ForestIndex):
        forest = dataset
        library = None
        lib_size = forest.n_spectra
        lib_spectra = forest.spectra
        partitions = forest.partitions
    else:
        forest = dataset.forest
        library = dataset.library
        lib_size = library.n_spectra
        lib_spectra = library.spectra
        partitions = forest.partitions

    has_unknown_partition = any(p.ion_mode == IonMode.UNKNOWN for p in partitions)

    parsed_queries: list[tuple[int, SpectrumPeaks, QueryConfig]] = []
    for idx, item in enumerate(queries):
        if len(item) == 3:
            query_row, q_peaks, q_cfg = item  # type: ignore[misc]
        elif isinstance(config, Sequence):
            query_row, q_peaks = item[:2]
            q_cfg = config[idx]
        elif config is not None:
            query_row, q_peaks = item[:2]
            q_cfg = config
        else:
            raise ValueError("未为查询提供有效的 QueryConfig")

        if (
            q_cfg.ion_mode == IonMode.UNKNOWN
            and q_cfg.ion_mode_policy == IonModePolicy.INCLUDE_UNKNOWN
            and not has_unknown_partition
        ):
            warnings.warn(
                f"查询 {query_row} 使用了默认 UNKNOWN 离子模式，而库中无 UNKNOWN 分区，将检索不到任何结果。",
                UserWarning,
                stacklevel=2,
            )

        parsed_queries.append((query_row, q_peaks, q_cfg))

    # 预热 JIT、包络缓存与解释器循环
    for i in range(min(warmup, len(parsed_queries))):
        search_forest(parsed_queries[i][1], forest, library, parsed_queries[i][2])

    # 1. 测量 JET-Forest 检索耗时与剪枝统计
    jetf_times_ms: list[float] = []
    scored_counts: list[int] = []

    effective_concurrency = max(1, min(concurrency, len(parsed_queries)))

    t_wall_start = time.perf_counter()
    with adaptive_numba_threads(effective_concurrency) as target_inner:
        if effective_concurrency <= 1:
            for _, q, q_cfg in parsed_queries:
                t0 = time.perf_counter()
                outcome = search_forest(q, forest, library, q_cfg)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                jetf_times_ms.append(elapsed_ms)
                scored_counts.append(outcome.stats.n_scored)
        else:
            from concurrent.futures import ThreadPoolExecutor

            def _init_worker(inner_threads: int | None) -> None:
                if inner_threads is not None:
                    try:
                        import numba
                        numba.set_num_threads(inner_threads)
                    except Exception:
                        pass

            def _eval_single(item: tuple[int, SpectrumPeaks, QueryConfig]) -> tuple[float, int]:
                _, q, q_cfg = item
                t0 = time.perf_counter()
                outcome = search_forest(q, forest, library, q_cfg, parallel_root_bounds=False)
                elapsed = (time.perf_counter() - t0) * 1000.0
                return elapsed, outcome.stats.n_scored

            with ThreadPoolExecutor(
                max_workers=effective_concurrency,
                initializer=_init_worker,
                initargs=(target_inner,),
            ) as executor:
                eval_results = list(executor.map(_eval_single, parsed_queries))

            jetf_times_ms = [res[0] for res in eval_results]
            scored_counts = [res[1] for res in eval_results]

    wall_duration_s = time.perf_counter() - t_wall_start

    # 2. 测量 matchms 检索耗时 (可选)
    matchms_times_ms: list[float] = []

    if not skip_matchms:
        check_matchms_available()
        from matchms.similarity import CosineGreedy

        if lib_size > 50000:
            warnings.warn(
                f"当前库规模为 {lib_size:,} 条谱，全库 matchms 穷举计算将耗费大量内存与时间。"
                "建议在百万级全库宏观吞吐量测试时启用 skip_matchms=True。",
                UserWarning,
                stacklevel=2,
            )

        scorers: dict[float, CosineGreedy] = {}

        def _get_scorer(tau: float) -> CosineGreedy:
            if tau not in scorers:
                scorers[tau] = CosineGreedy(tolerance=tau, mz_power=0.0, intensity_power=1.0)
            return scorers[tau]

        if library is not None:
            matchms_lib = [
                jetf_peaks_to_matchms(library.peaks.spectrum_at(r), meta=library.spectra[r])
                for r in range(lib_size)
            ]
        else:
            empty_peaks = SpectrumPeaks._create_unchecked(
                mass=np.empty(0, dtype=np.float64),
                intensity=np.empty(0, dtype=np.float64),
                energy=np.empty(0, dtype=np.float64),
                peak_id=np.empty(0, dtype=np.int64),
                norm=0.0,
            )
            matchms_lib = []
            for r in range(lib_size):
                iid = int(forest.row_to_internal[r])
                sp = forest.postings.spectrum_at(iid) if iid >= 0 else empty_peaks
                meta = lib_spectra[r] if lib_spectra else None
                matchms_lib.append(jetf_peaks_to_matchms(sp, meta=meta))

        m_queries = [jetf_peaks_to_matchms(q) for _, q, _ in parsed_queries]

        if lib_size > 0:
            w_scorer = _get_scorer(parsed_queries[0][2].fragment_tolerance_da)
            for r_warm in range(min(20, lib_size)):
                w_scorer.pair(m_queries[0], matchms_lib[r_warm])

        for idx, m_q in enumerate(m_queries):
            q_cfg = parsed_queries[idx][2]
            k = q_cfg.k or 10
            is_threshold_mode = (q_cfg.mode == SearchMode.THRESHOLD)
            thresh_val = q_cfg.threshold or 0.0
            scorer = _get_scorer(q_cfg.fragment_tolerance_da)
            t0 = time.perf_counter()
            if is_threshold_mode:
                candidates: list[tuple[float, int]] = []
                for row, meta in enumerate(lib_spectra):
                    if not is_eligible(q_cfg, meta):
                        continue
                    lib_s = matchms_lib[row]
                    res = scorer.pair(m_q, lib_s)
                    s_val = float(res["score"])
                    n_m = int(res["matches"])
                    if n_m >= q_cfg.min_matched_peaks and s_val >= thresh_val:
                        candidates.append((s_val, row))
                candidates.sort(key=lambda x: -x[0])
            else:
                top_k_heap: list[tuple[float, int]] = []
                for row, meta in enumerate(lib_spectra):
                    if not is_eligible(q_cfg, meta):
                        continue
                    lib_s = matchms_lib[row]
                    res = scorer.pair(m_q, lib_s)
                    s_val = float(res["score"])
                    n_m = int(res["matches"])
                    if n_m >= q_cfg.min_matched_peaks:
                        if len(top_k_heap) < k:
                            heapq.heappush(top_k_heap, (s_val, row))
                        elif s_val > top_k_heap[0][0]:
                            heapq.heapreplace(top_k_heap, (s_val, row))
                top_k_heap.sort(key=lambda x: -x[0])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            matchms_times_ms.append(elapsed_ms)

    jetf_arr = np.array(jetf_times_ms, dtype=np.float64)
    # 多线程并发下，均值按系统级有效服务时延 (Wall / N) 摊薄计算；单线程下与单次均值一致
    if effective_concurrency > 1:
        jetf_mean = (wall_duration_s / len(parsed_queries)) * 1000.0
    else:
        jetf_mean = float(np.mean(jetf_arr)) if jetf_arr.size > 0 else 0.0
    jetf_std = float(np.std(jetf_arr)) if jetf_arr.size > 0 else 0.0
    # 系统级有效 QPS 统计（考虑多核并发墙上时延）
    jetf_qps = len(parsed_queries) / wall_duration_s if wall_duration_s > 0 else (1000.0 / jetf_mean if jetf_mean > 0 else 0.0)

    throughput_speedup = 0.0
    if not skip_matchms and matchms_times_ms:
        mms_arr = np.array(matchms_times_ms, dtype=np.float64)
        mms_mean = float(np.mean(mms_arr))
        mms_std = float(np.std(mms_arr))
        mms_qps = 1000.0 / mms_mean if mms_mean > 0 else 0.0
        mms_p50 = float(np.percentile(mms_arr, 50))
        mms_p95 = float(np.percentile(mms_arr, 95))
        mms_p99 = float(np.percentile(mms_arr, 99))
        speedup = mms_mean / jetf_mean if jetf_mean > 0 else 1.0
        if mms_qps > 0:
            throughput_speedup = jetf_qps / mms_qps
    else:
        mms_mean = 0.0
        mms_std = 0.0
        mms_qps = 0.0
        mms_p50 = 0.0
        mms_p95 = 0.0
        mms_p99 = 0.0
        speedup = float('nan')  # matchms 被跳过时无加速比
        throughput_speedup = float('nan')

    avg_scored = float(np.mean(scored_counts))
    avg_scored_ratio = avg_scored / lib_size if lib_size > 0 else 1.0
    avg_pruned_ratio = 1.0 - avg_scored_ratio

    return RetrievalThroughputResult(
        mode_name=mode_name,
        n_queries=len(queries),
        library_size=lib_size,
        jetf_qps=jetf_qps,
        matchms_qps=mms_qps,
        jetf_latency_mean_ms=jetf_mean,
        jetf_latency_p50_ms=float(np.percentile(jetf_arr, 50)) if jetf_arr.size > 0 else 0.0,
        jetf_latency_p95_ms=float(np.percentile(jetf_arr, 95)) if jetf_arr.size > 0 else 0.0,
        jetf_latency_p99_ms=float(np.percentile(jetf_arr, 99)) if jetf_arr.size > 0 else 0.0,
        matchms_latency_mean_ms=mms_mean,
        matchms_latency_p50_ms=mms_p50,
        matchms_latency_p95_ms=mms_p95,
        matchms_latency_p99_ms=mms_p99,
        speedup=speedup,
        avg_scored_ratio=avg_scored_ratio,
        avg_pruned_ratio=avg_pruned_ratio,
        jetf_latency_std_ms=jetf_std,
        matchms_latency_std_ms=mms_std,
        concurrency=concurrency,
        throughput_speedup=throughput_speedup,
    )


def benchmark_scalability(
    mgf_path: str | Path | None = None,
    library_sizes: Sequence[int] = (250, 500, 1000, 2000),
    n_queries: int = 20,
    config: QueryConfig | None = None,
    mode_name: str = "open_top_10",
) -> list[RetrievalThroughputResult]:
    """评测随参考库谱数增长时的开放检索吞吐量与加速比伸缩性。"""
    results: list[RetrievalThroughputResult] = []

    for size in library_sizes:
        ds = load_benchmark_dataset(mgf_path=mgf_path, library_size=size)
        queries = sample_query_spectra(ds.library, n_queries=n_queries)

        if config is None:
            # 默认配置为开放检索 Top-10
            cfg = QueryConfig(
                mode=SearchMode.TOP_K,
                k=10,
                ion_mode=ds.library.spectra[0].ion_mode,
            )
        else:
            cfg = config

        res = benchmark_retrieval_throughput(ds, queries, cfg, mode_name=f"{mode_name} (N={size})")
        results.append(res)

    return results
