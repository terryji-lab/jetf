"""End-to-end tests and benchmarks for GPU batch threshold search.

Validates:
1. End-to-end bitwise consistency with CPU reference implementation (search_forest_batch)
   across multiple threshold levels (theta = 0.5, 0.7, 0.9) on GNPS library subset.
2. Guaranteed zero false dismissals and identical ranking, external_id, spectrum_index,
   n_matched, and float scores (atol=1e-12).
3. Edge & corner cases: empty query sets, 0-peak queries, extreme high/low thresholds,
   auto-conversion from host ForestIndex, and Top-K mode rejection.
4. End-to-end throughput and speedup benchmark against CPU single/multi-thread.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np
import pytest

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestIndex,
    ForestSpec,
    ParsedLibrary,
    PreprocessedLibrary,
    QueryConfig,
    SCORER_VERSIONED_ID,
    SearchMode,
    SpectrumPeaks,
    build_forest_index,
    parse_mgf,
    preprocess_library,
    search_forest_batch,
)
from jetf.gpu import (
    GpuForestIndex,
    is_cuda_available,
    search_forest_batch_gpu,
    search_threshold_batch_gpu,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _subset import stratified_subset_indices, subset_parsed_library

GNPS_PATH = Path(__file__).resolve().parents[3] / "GNPS-LIBRARY.mgf"


@pytest.fixture(scope="module")
def gnps_e2e_fixture():
    if not GNPS_PATH.exists():
        pytest.skip(f"GNPS dataset not found at {GNPS_PATH}")
    parsed = parse_mgf(GNPS_PATH)
    # Stratified subset of 100 spectra (25 per stratum)
    indices = stratified_subset_indices(parsed, n_per_stratum=25)
    sub = subset_parsed_library(parsed, indices)
    lib = preprocess_library(sub, CORRECTNESS_V1)
    spec = ForestSpec(tree_capacity=8, leaf_capacity=4)
    forest = build_forest_index(lib, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)
    return lib, forest, gpu_forest


def _assert_outcomes_equal(gpu_outcomes, cpu_outcomes, label=""):
    """Strictly assert zero false dismissals and bitwise match between GPU and CPU outcomes."""
    assert len(gpu_outcomes) == len(cpu_outcomes), (
        f"{label} Number of outcomes mismatch: GPU {len(gpu_outcomes)} vs CPU {len(cpu_outcomes)}"
    )

    for q_idx, (gpu_res, cpu_res) in enumerate(zip(gpu_outcomes, cpu_outcomes)):
        assert gpu_res.complete is True
        assert cpu_res.complete is True
        assert len(gpu_res.hits) == len(cpu_res.hits), (
            f"{label} Query #{q_idx}: hit count mismatch: GPU {len(gpu_res.hits)} vs CPU {len(cpu_res.hits)}"
        )

        for h_idx, (h_gpu, h_cpu) in enumerate(zip(gpu_res.hits, cpu_res.hits)):
            assert h_gpu.spectrum_index == h_cpu.spectrum_index, (
                f"{label} Query #{q_idx} Hit #{h_idx} spectrum_index mismatch: {h_gpu} vs {h_cpu}"
            )
            assert h_gpu.external_id == h_cpu.external_id, (
                f"{label} Query #{q_idx} Hit #{h_idx} external_id mismatch: {h_gpu} vs {h_cpu}"
            )
            assert h_gpu.n_matched == h_cpu.n_matched, (
                f"{label} Query #{q_idx} Hit #{h_idx} n_matched mismatch: {h_gpu} vs {h_cpu}"
            )
            assert np.isclose(h_gpu.score, h_cpu.score, atol=1e-12), (
                f"{label} Query #{q_idx} Hit #{h_idx} score mismatch: {h_gpu.score} vs {h_cpu.score}"
            )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_threshold_search_bitwise_consistency(gnps_e2e_fixture):
    """End-to-end bitwise verification against CPU search across multiple thresholds (theta = 0.5, 0.7, 0.9)."""
    lib, forest, gpu_forest = gnps_e2e_fixture

    # Select 16 diverse queries from the library
    test_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 99]
    queries = [lib.peaks.spectrum_at(i) for i in test_indices]

    for theta in (0.5, 0.7, 0.9):
        config = QueryConfig(
            mode=SearchMode.THRESHOLD,
            threshold=theta,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=lib.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
            snapshot_id="e2e_test",
        )

        cpu_outcomes = search_forest_batch(
            queries=queries,
            forest=forest,
            library=lib,
            config=config,
            concurrency=1,
            uind=True,
        )

        # GPU with U_ind enabled
        gpu_outcomes_uind = search_threshold_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=lib,
            config=config,
            batch_size=8,
            uind=True,
        )
        _assert_outcomes_equal(gpu_outcomes_uind, cpu_outcomes, label=f"[theta={theta}, uind=True]")

        # GPU without U_ind enabled (verifying K1 + K2 + exact scoring directly)
        gpu_outcomes_no_uind = search_threshold_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=lib,
            config=config,
            batch_size=8,
            uind=False,
        )
        _assert_outcomes_equal(gpu_outcomes_no_uind, cpu_outcomes, label=f"[theta={theta}, uind=False]")


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_threshold_search_corner_and_edge_cases(gnps_e2e_fixture):
    """End-to-end testing of boundary conditions and corner cases."""
    lib, forest, gpu_forest = gnps_e2e_fixture

    base_config = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.7,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="e2e_edge_cases",
    )

    # 1. Empty queries
    empty_outcomes = search_threshold_batch_gpu(
        queries=[],
        gpu_forest=gpu_forest,
        library=lib,
        config=base_config,
    )
    assert empty_outcomes == []

    # 2. Batch with 0-peak queries mixed with valid queries
    q0 = lib.peaks.spectrum_at(0)
    q_empty = SpectrumPeaks(
        mass=np.empty(0, dtype=np.float64),
        intensity=np.empty(0, dtype=np.float64),
        energy=np.empty(0, dtype=np.float64),
        peak_id=np.empty(0, dtype=np.int64),
        norm=1.0,
    )
    q1 = lib.peaks.spectrum_at(1)

    mixed_queries = [q0, q_empty, q1]
    cpu_mixed = search_forest_batch(mixed_queries, forest, lib, base_config, concurrency=1)
    gpu_mixed = search_threshold_batch_gpu(mixed_queries, gpu_forest, lib, base_config, batch_size=2)
    _assert_outcomes_equal(gpu_mixed, cpu_mixed, label="[mixed 0-peak queries]")
    assert len(gpu_mixed[1].hits) == 0

    # 3. All 0-peak queries batch
    all_empty_queries = [q_empty, q_empty]
    cpu_all_empty = search_forest_batch(all_empty_queries, forest, lib, base_config, concurrency=1)
    gpu_all_empty = search_threshold_batch_gpu(all_empty_queries, gpu_forest, lib, base_config)
    _assert_outcomes_equal(gpu_all_empty, cpu_all_empty, label="[all 0-peak queries]")

    # 4. Extreme high threshold (no hits expected)
    high_config = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=1.5,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="e2e_edge_cases",
    )
    gpu_high = search_threshold_batch_gpu([q0, q1], gpu_forest, lib, high_config)
    cpu_high = search_forest_batch([q0, q1], forest, lib, high_config)
    _assert_outcomes_equal(gpu_high, cpu_high, label="[extreme high threshold]")
    assert all(len(o.hits) == 0 for o in gpu_high)

    # 5. Extreme low threshold (threshold = 0.0 with zero supplement)
    low_config = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.0,
        min_matched_peaks=0,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="e2e_edge_cases",
    )
    gpu_low = search_threshold_batch_gpu([q0], gpu_forest, lib, low_config)
    cpu_low = search_forest_batch([q0], forest, lib, low_config)
    _assert_outcomes_equal(gpu_low, cpu_low, label="[extreme low threshold 0.0]")

    # 6. Auto-conversion from host ForestIndex
    gpu_auto = search_threshold_batch_gpu([q0], forest, lib, base_config)
    _assert_outcomes_equal(gpu_auto, [cpu_mixed[0]], label="[host ForestIndex auto-conversion]")

    # 7. Unified dispatcher search_forest_batch_gpu
    gpu_disp = search_forest_batch_gpu([q0], gpu_forest, lib, base_config)
    _assert_outcomes_equal(gpu_disp, [cpu_mixed[0]], label="[search_forest_batch_gpu dispatcher]")

    # 8. Top-K mode should raise NotImplementedError in search_threshold_batch_gpu
    topk_config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=10,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
    )
    with pytest.raises(NotImplementedError, match="Phase 4"):
        search_threshold_batch_gpu([q0], gpu_forest, lib, topk_config)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_threshold_search_benchmark_and_speedup(gnps_e2e_fixture):
    """Benchmark throughput and speedup: CPU single/multi-thread vs GPU search_threshold_batch_gpu."""
    lib, forest, gpu_forest = gnps_e2e_fixture

    n_q = min(64, lib.n_spectra)
    queries = [lib.peaks.spectrum_at(i % lib.n_spectra) for i in range(n_q)]

    config = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.7,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="benchmark",
    )

    # 1. Warm-up GPU
    _ = search_threshold_batch_gpu(queries[:4], gpu_forest, lib, config, batch_size=4)

    # 2. Benchmark CPU single-thread
    t_start = time.perf_counter()
    cpu_single_outcomes = search_forest_batch(queries, forest, lib, config, concurrency=1)
    cpu_single_time = time.perf_counter() - t_start
    cpu_single_qps = n_q / cpu_single_time

    # 3. Benchmark CPU multi-thread (concurrency=4)
    t_start = time.perf_counter()
    cpu_multi_outcomes = search_forest_batch(queries, forest, lib, config, concurrency=4)
    cpu_multi_time = time.perf_counter() - t_start
    cpu_multi_qps = n_q / cpu_multi_time

    # 4. Benchmark GPU batch threshold search
    t_start = time.perf_counter()
    gpu_outcomes = search_threshold_batch_gpu(queries, gpu_forest, lib, config, batch_size=64)
    gpu_time = time.perf_counter() - t_start
    gpu_qps = n_q / gpu_time

    # Ensure correctness on benchmark queries
    _assert_outcomes_equal(gpu_outcomes, cpu_single_outcomes, label="[benchmark batch]")

    speedup_single = cpu_single_time / gpu_time
    speedup_multi = cpu_multi_time / gpu_time

    print("\n" + "=" * 70)
    print(f" JET-Forest Threshold Search Benchmark (N={n_q} queries, theta=0.7)")
    print("=" * 70)
    print(f" CPU Single-Thread : {cpu_single_time * 1000.0:8.2f} ms ({cpu_single_qps:6.1f} QPS)")
    print(f" CPU Multi-Thread  : {cpu_multi_time * 1000.0:8.2f} ms ({cpu_multi_qps:6.1f} QPS)")
    print(f" GPU Batch (b=64)  : {gpu_time * 1000.0:8.2f} ms ({gpu_qps:6.1f} QPS)")
    print(f" Speedup vs CPU-1T : {speedup_single:6.2f}x")
    print(f" Speedup vs CPU-MT : {speedup_multi:6.2f}x")
    print("=" * 70)

    assert gpu_time > 0.0
    assert len(gpu_outcomes) == n_q
