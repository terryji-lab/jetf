"""Unit tests for GPU-Host asynchronous overlap pipeline and precursor bucketing concurrency.

Validates Phase 2 / P1 features:
1. Precursor window adaptive bucketing (presort_by_precursor_mz) and exact inverse restoration.
2. End-to-end bitwise consistency between single-stream and double-buffering overlap pipeline
   (search_threshold_batch_gpu and search_topk_batch_gpu).
3. Narrow precursor window union_trees pruning efficiency across batches.
"""

from __future__ import annotations

import random
from pathlib import Path
import sys
import numpy as np
import pytest
from numba import cuda

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestIndex,
    ForestSpec,
    IonMode,
    IonModePolicy,
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
    search_forest_batch,
)
from jetf.gpu import (
    GpuForestIndex,
    is_cuda_available,
    presort_by_precursor_mz,
    search_threshold_batch_gpu,
    search_topk_batch_gpu,
)
from jetf.gpu.search import _find_eligible_trees

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _subset import stratified_subset_indices, subset_parsed_library
from tests.unit.gpu.test_device import make_synthetic_library

GNPS_PATH = Path(__file__).resolve().parents[3] / "GNPS-LIBRARY.mgf"


@pytest.fixture(scope="module")
def multi_tree_forest_fixture():
    """Forest fixture with multiple trees across distinct precursor m/z ranges."""
    if GNPS_PATH.exists():
        parsed = parse_mgf(GNPS_PATH)
        indices = stratified_subset_indices(parsed, n_per_stratum=25)
        sub = subset_parsed_library(parsed, indices)
        lib = preprocess_library(sub, CORRECTNESS_V1)
        spec = ForestSpec(tree_capacity=8, leaf_capacity=4)
        forest = build_forest_index(lib, spec)
        gpu_forest = GpuForestIndex.from_forest(forest)
        return lib, forest, gpu_forest
    else:
        parsed = make_synthetic_library(n_spectra=64)
        lib = preprocess_library(parsed, CORRECTNESS_V1)
        spec = ForestSpec(tree_capacity=8, leaf_capacity=4)
        forest = build_forest_index(lib, spec)
        gpu_forest = GpuForestIndex.from_forest(forest)
        return lib, forest, gpu_forest


def _assert_outcomes_identical(outcomes_a, outcomes_b, label=""):
    """Strict assertion that two outcome lists match 100% bitwise."""
    assert len(outcomes_a) == len(outcomes_b), (
        f"{label} Length mismatch: {len(outcomes_a)} vs {len(outcomes_b)}"
    )
    for q_idx, (oa, ob) in enumerate(zip(outcomes_a, outcomes_b)):
        assert oa.complete is True and ob.complete is True
        assert len(oa.hits) == len(ob.hits), (
            f"{label} Query #{q_idx} hits length mismatch: {len(oa.hits)} vs {len(ob.hits)}"
        )
        for h_idx, (ha, hb) in enumerate(zip(oa.hits, ob.hits)):
            assert ha.spectrum_index == hb.spectrum_index, (
                f"{label} Query #{q_idx} Hit #{h_idx} spectrum_index mismatch: {ha.spectrum_index} vs {hb.spectrum_index}"
            )
            assert ha.external_id == hb.external_id, (
                f"{label} Query #{q_idx} Hit #{h_idx} external_id mismatch: {ha.external_id} vs {hb.external_id}"
            )
            assert ha.n_matched == hb.n_matched, (
                f"{label} Query #{q_idx} Hit #{h_idx} n_matched mismatch: {ha.n_matched} vs {hb.n_matched}"
            )
            assert np.isclose(ha.score, hb.score, atol=1e-12), (
                f"{label} Query #{q_idx} Hit #{h_idx} score mismatch: {ha.score} vs {hb.score}"
            )


# ==============================================================================
# Test 1: Precursor Window Presort Bucketing and Order Preservation
# ==============================================================================

def test_presort_by_precursor_mz_unit():
    """Unit test for presort_by_precursor_mz mapping and exact inverse reconstruction."""
    # Create synthetic mock queries
    q0 = SpectrumPeaks(mass=np.array([100.0]), intensity=np.array([10.0]), energy=np.array([1.0]), peak_id=np.array([0]), norm=1.0)
    q1 = SpectrumPeaks(mass=np.array([200.0]), intensity=np.array([20.0]), energy=np.array([1.0]), peak_id=np.array([1]), norm=1.0)
    q2 = SpectrumPeaks(mass=np.array([300.0]), intensity=np.array([30.0]), energy=np.array([1.0]), peak_id=np.array([2]), norm=1.0)
    q3 = SpectrumPeaks(mass=np.array([400.0]), intensity=np.array([40.0]), energy=np.array([1.0]), peak_id=np.array([3]), norm=1.0)
    q4 = SpectrumPeaks(mass=np.array([500.0]), intensity=np.array([50.0]), energy=np.array([1.0]), peak_id=np.array([4]), norm=1.0)

    # Configs with out-of-order precursor windows and one None window
    c0 = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.7, precursor_window=PrecursorWindow(mz=500.0, tolerance_da=10.0))  # center: 500
    c1 = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.7, precursor_window=PrecursorWindow(mz=100.0, tolerance_da=10.0))   # center: 100
    c2 = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.7, precursor_window=None)                             # center: inf
    c3 = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.7, precursor_window=PrecursorWindow(mz=200.0, tolerance_da=10.0))  # center: 200
    c4 = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.7, precursor_window=PrecursorWindow(mz=300.0, tolerance_da=10.0))  # center: 300

    queries = [q0, q1, q2, q3, q4]
    configs = [c0, c1, c2, c3, c4]

    sorted_queries, sorted_configs, inv_order = presort_by_precursor_mz(queries, configs)

    assert inv_order is not None
    assert sorted_configs[0] == c1
    assert sorted_configs[1] == c3
    assert sorted_configs[2] == c4
    assert sorted_configs[3] == c0
    assert sorted_configs[4] == c2

    # Verify inverse restoration
    restored_queries = [sorted_queries[inv_order[i]] for i in range(len(queries))]
    restored_configs = [sorted_configs[inv_order[i]] for i in range(len(configs))]

    for orig_q, rest_q in zip(queries, restored_queries):
        np.testing.assert_array_equal(orig_q.mass, rest_q.mass)
    for orig_c, rest_c in zip(configs, restored_configs):
        assert orig_c == rest_c

    # When no queries have precursor windows:
    configs_no_pw = [QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.7, precursor_window=None) for _ in range(3)]
    sq, sc, inv = presort_by_precursor_mz(queries[:3], configs_no_pw)
    assert inv is None
    assert sq == queries[:3]
    assert sc == configs_no_pw


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_precursor_presort_end_to_end_order_preservation(multi_tree_forest_fixture):
    """End-to-end verification that precursor presorting preserves exact query-outcome mapping."""
    lib, forest, gpu_forest = multi_tree_forest_fixture

    n_test = min(20, lib.n_spectra)
    rng = random.Random(42)
    sample_indices = list(range(n_test))
    rng.shuffle(sample_indices)

    queries: list[SpectrumPeaks] = []
    configs: list[QueryConfig] = []
    topk_configs: list[QueryConfig] = []

    for idx in sample_indices:
        q = lib.peaks.spectrum_at(idx)
        meta = lib.spectra[idx]
        pmz = meta.precursor_mz
        queries.append(q)

        # Narrow precursor window around true precursor mz
        pw = PrecursorWindow(mz=pmz, tolerance_da=5.0)
        configs.append(
            QueryConfig(
                mode=SearchMode.THRESHOLD,
                threshold=0.6,
                precursor_window=pw,
                ion_mode=meta.ion_mode,
                ion_mode_policy=IonModePolicy.EXACT,
                fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
                preprocess_version=lib.spec.versioned_id,
                scorer_version=SCORER_VERSIONED_ID,
            )
        )
        topk_configs.append(
            QueryConfig(
                mode=SearchMode.TOP_K,
                k=5,
                precursor_window=pw,
                ion_mode=meta.ion_mode,
                ion_mode_policy=IonModePolicy.EXACT,
                fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
                preprocess_version=lib.spec.versioned_id,
                scorer_version=SCORER_VERSIONED_ID,
            )
        )

    # 1. Run CPU search as gold-standard reference (preserves input order)
    cpu_threshold_outcomes = search_forest_batch(
        queries, forest, lib, config=configs, concurrency=1
    )

    # 2. Run GPU threshold search with small batch_size=4 to force multi-batch presort bucketing
    gpu_threshold_outcomes = search_threshold_batch_gpu(
        queries, gpu_forest, lib, config=configs, batch_size=4
    )

    # Assert 100% bitwise query-for-query identity
    _assert_outcomes_identical(
        gpu_threshold_outcomes, cpu_threshold_outcomes, label="[Presort Order Threshold]"
    )

    # 3. Run GPU Top-K search with small batch_size=4
    cpu_topk_outcomes = search_forest_batch(
        queries, forest, lib, config=topk_configs, concurrency=1
    )
    gpu_topk_outcomes = search_topk_batch_gpu(
        queries, gpu_forest, lib, config=topk_configs, batch_size=4
    )
    _assert_outcomes_identical(
        gpu_topk_outcomes, cpu_topk_outcomes, label="[Presort Order Top-K]"
    )


# ==============================================================================
# Test 2: Multi-Batch Pipeline Concurrency Consistency (Single vs Double Stream)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_pipeline_concurrency_single_vs_double_stream(multi_tree_forest_fixture):
    """Assert 100% bitwise consistency between single-stream and double-buffering overlap pipeline."""
    lib, forest, gpu_forest = multi_tree_forest_fixture

    n_test = min(24, lib.n_spectra)
    queries = [lib.peaks.spectrum_at(i) for i in range(n_test)]

    # Threshold configs
    threshold_configs = [
        QueryConfig(
            mode=SearchMode.THRESHOLD,
            threshold=0.6,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=lib.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
        )
        for _ in range(n_test)
    ]

    # Batch size = 4 produces 6 batches, heavily exercising the ping-pong double-buffering pipeline
    batch_size = 4

    # Run Threshold with double-buffering overlap pipeline (stream=None)
    outcomes_double_thresh = search_threshold_batch_gpu(
        queries, gpu_forest, lib, config=threshold_configs, batch_size=batch_size, stream=None
    )

    # Run Threshold with explicit single CUDA stream (serialized execution)
    single_stream = cuda.stream()
    outcomes_single_thresh = search_threshold_batch_gpu(
        queries, gpu_forest, lib, config=threshold_configs, batch_size=batch_size, stream=single_stream
    )

    # Bitwise comparison
    _assert_outcomes_identical(
        outcomes_double_thresh, outcomes_single_thresh, label="[Threshold Single vs Double Stream]"
    )

    # Top-K configs
    topk_configs = [
        QueryConfig(
            mode=SearchMode.TOP_K,
            k=5,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=lib.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
        )
        for _ in range(n_test)
    ]

    # Run Top-K with double-buffering overlap pipeline (stream=None)
    outcomes_double_topk = search_topk_batch_gpu(
        queries, gpu_forest, lib, config=topk_configs, batch_size=batch_size, stream=None
    )

    # Run Top-K with explicit single CUDA stream
    single_stream_topk = cuda.stream()
    outcomes_single_topk = search_topk_batch_gpu(
        queries, gpu_forest, lib, config=topk_configs, batch_size=batch_size, stream=single_stream_topk
    )

    # Bitwise comparison
    _assert_outcomes_identical(
        outcomes_double_topk, outcomes_single_topk, label="[Top-K Single vs Double Stream]"
    )


# ==============================================================================
# Test 3: Narrow Precursor Window union_trees Pruning Efficiency
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_precursor_bucketing_union_trees_pruning_efficiency(multi_tree_forest_fixture):
    """Verify that presort bucketing significantly reduces total union_trees evaluated across batches."""
    lib, forest, gpu_forest = multi_tree_forest_fixture

    # Find spectra spanning a wide range of precursor m/z
    all_pmzs = np.array([m.precursor_mz for m in lib.spectra])
    sorted_spec_indices = np.argsort(all_pmzs)

    # Select 8 spectra with low precursor m/z and 8 spectra with high precursor m/z
    low_indices = sorted_spec_indices[:8]
    high_indices = sorted_spec_indices[-8:]

    low_pmz = float(all_pmzs[low_indices[-1]])
    high_pmz = float(all_pmzs[high_indices[0]])

    # Make sure we have a clear distinction between low and high precursor m/z
    if high_pmz - low_pmz < 50.0:
        pytest.skip("Not enough precursor m/z divergence in dataset to demonstrate tree separation")

    interleaved_indices = []
    for l_idx, h_idx in zip(low_indices, high_indices):
        interleaved_indices.extend([l_idx, h_idx])

    interleaved_queries = [lib.peaks.spectrum_at(i) for i in interleaved_indices]
    interleaved_configs = [
        QueryConfig(
            mode=SearchMode.THRESHOLD,
            threshold=0.7,
            precursor_window=PrecursorWindow(
                mz=lib.spectra[i].precursor_mz,
                tolerance_da=5.0,
            ),
            ion_mode_policy=IonModePolicy.ANY,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=lib.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
        )
        for i in interleaved_indices
    ]

    batch_size = 4
    n_queries = len(interleaved_queries)

    # 1. Compute union_trees for unbucketed (interleaved) batches
    unbucketed_total_union_trees = 0
    for b_start in range(0, n_queries, batch_size):
        b_end = min(b_start + batch_size, n_queries)
        b_cfgs = interleaved_configs[b_start:b_end]
        trees_for_batch = []
        for cfg in b_cfgs:
            trees_for_batch.extend(_find_eligible_trees(forest, cfg))
        union_trees = np.unique(np.array(trees_for_batch, dtype=np.int64))
        unbucketed_total_union_trees += len(union_trees)

    # 2. Compute union_trees for bucketed (presorted) batches
    sorted_queries, sorted_configs, inv_order = presort_by_precursor_mz(
        interleaved_queries, interleaved_configs
    )
    bucketed_total_union_trees = 0
    for b_start in range(0, n_queries, batch_size):
        b_end = min(b_start + batch_size, n_queries)
        b_cfgs = sorted_configs[b_start:b_end]
        trees_for_batch = []
        for cfg in b_cfgs:
            trees_for_batch.extend(_find_eligible_trees(forest, cfg))
        union_trees = np.unique(np.array(trees_for_batch, dtype=np.int64))
        bucketed_total_union_trees += len(union_trees)

    # Bucketing must reduce total candidate union trees
    assert bucketed_total_union_trees < unbucketed_total_union_trees, (
        f"Bucketed union trees ({bucketed_total_union_trees}) should be strictly less than "
        f"unbucketed ({unbucketed_total_union_trees})"
    )

    reduction_pct = (1.0 - bucketed_total_union_trees / unbucketed_total_union_trees) * 100.0
    print(
        f"\n[Pruning Efficiency] Unbucketed: {unbucketed_total_union_trees} trees, "
        f"Bucketed: {bucketed_total_union_trees} trees, Reduction: {reduction_pct:.1f}%"
    )
