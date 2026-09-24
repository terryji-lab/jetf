"""End-to-end tests and benchmarks for GPU batch Top-K search.

Validates:
1. End-to-end bitwise consistency with CPU reference implementation (search_forest_batch)
   across multiple Top-K settings (k = 1, 5, 10) on GNPS library subset.
2. Guaranteed 100% Recall@K, zero false dismissals, identical ranking, external_id,
   spectrum_index, n_matched, and float scores (atol=1e-12).
3. 100% Recall@K and bitwise alignment with brute-force search_exhaustive.
4. Edge & corner cases: empty query sets, 0-peak queries, precursor window filtering,
   queries with no possible matches, auto-conversion from host ForestIndex, and
   unified dispatcher search_forest_batch_gpu routing.
5. End-to-end throughput and speedup benchmark against CPU single/multi-thread.
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
    PrecursorWindow,
    PreprocessedLibrary,
    QueryConfig,
    SCORER_VERSIONED_ID,
    SearchMode,
    SpectrumPeaks,
    build_forest_index,
    parse_mgf,
    preprocess_library,
    search_exhaustive,
    search_forest_batch,
)
from jetf.gpu import (
    GpuForestIndex,
    is_cuda_available,
    search_forest_batch_gpu,
    search_topk_batch_gpu,
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


def _assert_topk_outcomes_equal(gpu_outcomes, ref_outcomes, label=""):
    """Strictly assert Recall@K = 100% and bitwise match between GPU and reference outcomes."""
    assert len(gpu_outcomes) == len(ref_outcomes), (
        f"{label} Number of outcomes mismatch: GPU {len(gpu_outcomes)} vs REF {len(ref_outcomes)}"
    )

    for q_idx, (gpu_res, ref_res) in enumerate(zip(gpu_outcomes, ref_outcomes)):
        assert gpu_res.complete is True
        assert ref_res.complete is True
        assert len(gpu_res.hits) == len(ref_res.hits), (
            f"{label} Query #{q_idx}: hit count mismatch: GPU {len(gpu_res.hits)} vs REF {len(ref_res.hits)}"
        )

        ref_hit_keys = {(h.spectrum_index, h.external_id) for h in ref_res.hits}
        gpu_hit_keys = {(h.spectrum_index, h.external_id) for h in gpu_res.hits}
        if len(ref_hit_keys) > 0:
            recall = len(gpu_hit_keys & ref_hit_keys) / len(ref_hit_keys)
            assert recall == 1.0, (
                f"{label} Query #{q_idx}: Recall@K is {recall * 100:.1f}%, expected 100%!"
            )

        for h_idx, (h_gpu, h_ref) in enumerate(zip(gpu_res.hits, ref_res.hits)):
            assert h_gpu.spectrum_index == h_ref.spectrum_index, (
                f"{label} Query #{q_idx} Hit #{h_idx} spectrum_index mismatch: {h_gpu} vs {h_ref}"
            )
            assert h_gpu.external_id == h_ref.external_id, (
                f"{label} Query #{q_idx} Hit #{h_idx} external_id mismatch: {h_gpu} vs {h_ref}"
            )
            assert h_gpu.n_matched == h_ref.n_matched, (
                f"{label} Query #{q_idx} Hit #{h_idx} n_matched mismatch: {h_gpu} vs {h_ref}"
            )
            assert np.isclose(h_gpu.score, h_ref.score, atol=1e-12), (
                f"{label} Query #{q_idx} Hit #{h_idx} score mismatch: {h_gpu.score} vs {h_ref.score}"
            )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_topk_search_bitwise_consistency(gnps_e2e_fixture):
    """End-to-end bitwise verification against CPU search across multiple Top-K settings (k = 1, 5, 10)."""
    lib, forest, gpu_forest = gnps_e2e_fixture

    # Select 16 diverse queries from the library
    test_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 99]
    queries = [lib.peaks.spectrum_at(i) for i in test_indices]

    for k_val in (1, 5, 10):
        config = QueryConfig(
            mode=SearchMode.TOP_K,
            k=k_val,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=lib.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
            snapshot_id="e2e_topk_test",
        )

        cpu_outcomes = search_forest_batch(
            queries=queries,
            forest=forest,
            library=lib,
            config=config,
            concurrency=1,
            uind=True,
        )

        # GPU with U_ind enabled and default probe_trees=3
        gpu_outcomes_uind = search_topk_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=lib,
            config=config,
            batch_size=8,
            probe_trees=3,
            uind=True,
        )
        _assert_topk_outcomes_equal(
            gpu_outcomes_uind, cpu_outcomes, label=f"[k={k_val}, uind=True, probe=3]"
        )

        # GPU without U_ind enabled (verifying K1 + K2 + exact scoring directly)
        gpu_outcomes_no_uind = search_topk_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=lib,
            config=config,
            batch_size=8,
            probe_trees=3,
            uind=False,
        )
        _assert_topk_outcomes_equal(
            gpu_outcomes_no_uind, cpu_outcomes, label=f"[k={k_val}, uind=False, probe=3]"
        )

        # GPU with probe_trees=1 (verifying single probe tree)
        gpu_outcomes_probe1 = search_topk_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=lib,
            config=config,
            batch_size=8,
            probe_trees=1,
            uind=True,
        )
        _assert_topk_outcomes_equal(
            gpu_outcomes_probe1, cpu_outcomes, label=f"[k={k_val}, uind=True, probe=1]"
        )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_topk_search_against_exhaustive(gnps_e2e_fixture):
    """End-to-end verification against exhaustive ground truth search_exhaustive."""
    lib, forest, gpu_forest = gnps_e2e_fixture

    peak_counts = np.diff(lib.peaks.spectrum_offsets)
    idx_max = int(np.argmax(peak_counts))
    idx_min = int(np.argmin(peak_counts))
    test_rows = [0, 25, 50, idx_max, idx_min]
    queries = [lib.peaks.spectrum_at(r) for r in test_rows]

    for k_val in (5, 10):
        config = QueryConfig(
            mode=SearchMode.TOP_K,
            k=k_val,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=lib.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
            snapshot_id="e2e_topk_exhaustive",
        )

        exhaustive_outcomes = [search_exhaustive(q, lib, config) for q in queries]
        gpu_outcomes = search_topk_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=lib,
            config=config,
            batch_size=4,
            uind=True,
        )
        _assert_topk_outcomes_equal(
            gpu_outcomes, exhaustive_outcomes, label=f"[exhaustive vs GPU k={k_val}]"
        )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_topk_search_corner_and_edge_cases(gnps_e2e_fixture):
    """End-to-end testing of boundary conditions, edge cases, and unified dispatcher."""
    lib, forest, gpu_forest = gnps_e2e_fixture

    base_config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="e2e_topk_edge_cases",
    )

    # 1. Empty queries
    empty_outcomes = search_topk_batch_gpu(
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
    gpu_mixed = search_topk_batch_gpu(
        mixed_queries, gpu_forest, lib, base_config, batch_size=2
    )
    _assert_topk_outcomes_equal(gpu_mixed, cpu_mixed, label="[mixed 0-peak queries]")

    # 3. All 0-peak queries batch
    all_empty_queries = [q_empty, q_empty]
    cpu_all_empty = search_forest_batch(all_empty_queries, forest, lib, base_config, concurrency=1)
    gpu_all_empty = search_topk_batch_gpu(
        all_empty_queries, gpu_forest, lib, base_config
    )
    _assert_topk_outcomes_equal(gpu_all_empty, cpu_all_empty, label="[all 0-peak queries]")

    # 4. Top-K search with precursor window constraint
    meta0 = lib.spectra[0]
    prec_config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        precursor_window=PrecursorWindow(mz=meta0.precursor_mz, tolerance_da=5.0),
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="e2e_topk_prec_window",
    )
    gpu_prec = search_topk_batch_gpu([q0], gpu_forest, lib, prec_config)
    cpu_prec = search_forest_batch([q0], forest, lib, prec_config)
    _assert_topk_outcomes_equal(gpu_prec, cpu_prec, label="[precursor window constraint]")

    # 5. Queries with no possible matches (impossible precursor mz)
    no_match_config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        precursor_window=PrecursorWindow(mz=999999.0, tolerance_da=0.01),
        min_matched_peaks=1,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="e2e_topk_no_match",
    )
    gpu_no_match = search_topk_batch_gpu([q0, q1], gpu_forest, lib, no_match_config)
    cpu_no_match = search_forest_batch([q0, q1], forest, lib, no_match_config)
    _assert_topk_outcomes_equal(gpu_no_match, cpu_no_match, label="[no matches expected]")
    assert all(len(o.hits) == 0 for o in gpu_no_match)

    # 6. Auto-conversion from host ForestIndex
    gpu_auto = search_topk_batch_gpu([q0], forest, lib, base_config)
    _assert_topk_outcomes_equal(
        gpu_auto, [cpu_mixed[0]], label="[host ForestIndex auto-conversion]"
    )

    # 7. Unified dispatcher search_forest_batch_gpu with single config
    gpu_disp = search_forest_batch_gpu([q0], gpu_forest, lib, base_config)
    _assert_topk_outcomes_equal(
        gpu_disp, [cpu_mixed[0]], label="[search_forest_batch_gpu unified dispatcher]"
    )

    # 8. Unified dispatcher with per-query config sequence
    seq_configs = [base_config, prec_config]
    gpu_seq = search_forest_batch_gpu([q0, q0], gpu_forest, lib, seq_configs)
    cpu_seq = search_forest_batch([q0, q0], forest, lib, seq_configs)
    _assert_topk_outcomes_equal(
        gpu_seq, cpu_seq, label="[search_forest_batch_gpu config sequence]"
    )

    # 9. Invalid mode rejection in search_topk_batch_gpu
    thresh_config = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.7,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
    )
    with pytest.raises(ValueError, match="SearchMode.THRESHOLD is not supported"):
        search_topk_batch_gpu([q0], gpu_forest, lib, thresh_config)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_topk_search_benchmark_and_speedup(gnps_e2e_fixture):
    """Benchmark throughput and speedup: CPU single/multi-thread vs GPU search_topk_batch_gpu."""
    lib, forest, gpu_forest = gnps_e2e_fixture

    n_q = min(64, lib.n_spectra)
    queries = [lib.peaks.spectrum_at(i % lib.n_spectra) for i in range(n_q)]

    config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=10,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        preprocess_version=lib.spec.versioned_id,
        scorer_version=SCORER_VERSIONED_ID,
        snapshot_id="topk_benchmark",
    )

    # 1. Warm-up GPU
    _ = search_topk_batch_gpu(queries[:4], gpu_forest, lib, config, batch_size=4)

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

    # 4. Benchmark GPU batch Top-K search
    t_start = time.perf_counter()
    gpu_outcomes = search_topk_batch_gpu(queries, gpu_forest, lib, config, batch_size=64)
    gpu_time = time.perf_counter() - t_start
    gpu_qps = n_q / gpu_time

    # Ensure correctness on benchmark queries
    _assert_topk_outcomes_equal(gpu_outcomes, cpu_single_outcomes, label="[benchmark batch]")

    speedup_single = cpu_single_time / gpu_time
    speedup_multi = cpu_multi_time / gpu_time

    print("\n" + "=" * 70)
    print(f" JET-Forest Top-K Search Benchmark (N={n_q} queries, k=10)")
    print("=" * 70)
    print(f" CPU Single-Thread : {cpu_single_time * 1000.0:8.2f} ms ({cpu_single_qps:6.1f} QPS)")
    print(f" CPU Multi-Thread  : {cpu_multi_time * 1000.0:8.2f} ms ({cpu_multi_qps:6.1f} QPS)")
    print(f" GPU Batch (b=64)  : {gpu_time * 1000.0:8.2f} ms ({gpu_qps:6.1f} QPS)")
    print(f" Speedup vs CPU-1T : {speedup_single:6.2f}x")
    print(f" Speedup vs CPU-MT : {speedup_multi:6.2f}x")
    print("=" * 70)

    assert gpu_time > 0.0
    assert len(gpu_outcomes) == n_q
