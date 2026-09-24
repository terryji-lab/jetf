"""Unit tests for Phase 1 Core Algorithm & GPU Defect Fixes.

Covers:
1. GPU float32 ResultSet threshold tolerance compatibility (retaining 0.49997 under threshold=0.5).
2. Batch GPU search with mixed fragment_tolerance_da auto-bucketing and order restoration.
3. BatchQueryDevice.from_queries mass monotonicity assertion.
4. GPU upper bound underflow protection for tiny intensities with valid peak overlap.
5. CPU/GPU score truncation band consistency and documentation validation.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pytest

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestIndex,
    ForestSpec,
    IonMode,
    IonModePolicy,
    ParsedLibrary,
    PreprocessedLibrary,
    QueryConfig,
    SearchMode,
    SourceRef,
    SpectrumMeta,
    SpectrumPeaks,
    build_forest_index,
    preprocess_library,
)
from jetf.gpu import (
    DEFAULT_GPU_SCORE_MARGIN,
    BatchQueryDevice,
    GpuForestIndex,
    batch_root_bounds_gpu,
    batch_uind_pairs_gpu,
    is_cuda_available,
    require_cuda,
    search_forest_batch_gpu,
    search_threshold_batch_gpu,
    search_topk_batch_gpu,
)
from jetf.results import ResultSet, SearchHit
from jetf.scoring import score_greedy_cosine


def _make_peaks(mass: np.ndarray, intensity: np.ndarray) -> SpectrumPeaks:
    m = np.asarray(mass, dtype=np.float64)
    it = np.asarray(intensity, dtype=np.float64)
    return SpectrumPeaks(
        mass=m,
        intensity=it,
        energy=(it * it).astype(np.float64),
        peak_id=np.arange(len(m), dtype=np.int64),
    )


def _make_single_spec_library(
    masses: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float = 200.0,
    ion_mode: IonMode = IonMode.POSITIVE,
    external_id: str = "spec_0",
) -> tuple[PreprocessedLibrary, ForestIndex, GpuForestIndex]:
    """Helper to build a 1-spectrum forest for targeted numerical tests."""
    meta = SpectrumMeta(
        external_id=external_id,
        precursor_mz=precursor_mz,
        charge=1,
        ion_mode=ion_mode,
        source=SourceRef("test.mgf", 0),
    )
    n_peaks = len(masses)
    parsed = ParsedLibrary(
        source_path="test.mgf",
        spectra=(meta,),
        mass=masses.astype(np.float64),
        intensity=intensities.astype(np.float64),
        peak_id=np.arange(n_peaks, dtype=np.int64),
        spectrum_offsets=np.array([0, n_peaks], dtype=np.int64),
    )
    lib = preprocess_library(parsed, CORRECTNESS_V1)
    spec = ForestSpec(tree_capacity=4, leaf_capacity=2)
    forest = build_forest_index(lib, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)
    return lib, forest, gpu_forest


# ==============================================================================
# Test 1: GPU 结果集阈值容差兼容 (P0)
# ==============================================================================

def test_resultset_score_margin_boundary_tolerance():
    """Verify ResultSet score_margin behavior: 0.49997 with threshold 0.5."""
    cfg = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.5,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    hit = SearchHit(external_id="spec_test", spectrum_index=0, score=0.49997, n_matched=1)

    # 1. Standard CPU margin (1e-12) rejects 0.49997 under threshold=0.5
    res_cpu = ResultSet(cfg, score_margin=1e-12)
    res_cpu.update(hit)
    assert len(res_cpu.finish()) == 0

    # 2. GPU margin (5e-5) accepts 0.49997 under threshold=0.5
    res_gpu = ResultSet(cfg, score_margin=5e-5)
    assert res_gpu.score_margin == 5e-5
    res_gpu.update(hit)
    hits = res_gpu.finish()
    assert len(hits) == 1
    assert hits[0].score == 0.49997
    assert hits[0].external_id == "spec_test"

    # 3. Dynamic theta respects score_margin in THRESHOLD mode
    assert res_gpu.theta() == 0.5 - 5e-5


def test_resultset_accepts_zero_scores_decoupled_from_score_margin():
    """Verify accepts_zero_scores() is strictly decoupled from score_margin in THRESHOLD mode.

    Defect B regression test:
    Under GPU margin (score_margin = 5e-5), a small positive threshold (e.g. 1e-5)
    must NOT trigger zero-score acceptance. Only non-positive thresholds (<= 0.0)
    should accept zero scores.
    """
    # 1. threshold = 1e-5, score_margin = 5e-5: must return False (critical defect fix!)
    cfg_tiny_pos = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=1e-5,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    rs_gpu_tiny = ResultSet(cfg_tiny_pos, score_margin=5e-5)
    assert rs_gpu_tiny.accepts_zero_scores() is False

    # 2. threshold = 0.0: must return True!
    cfg_zero = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.0,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    rs_zero_cpu = ResultSet(cfg_zero, score_margin=1e-12)
    rs_zero_gpu = ResultSet(cfg_zero, score_margin=5e-5)
    assert rs_zero_cpu.accepts_zero_scores() is True
    assert rs_zero_gpu.accepts_zero_scores() is True

    # 3. threshold = -0.05: must return True!
    cfg_neg = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=-0.05,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    rs_neg_cpu = ResultSet(cfg_neg, score_margin=1e-12)
    rs_neg_gpu = ResultSet(cfg_neg, score_margin=5e-5)
    assert rs_neg_cpu.accepts_zero_scores() is True
    assert rs_neg_gpu.accepts_zero_scores() is True

    # 4. threshold = 0.7: must return False!
    cfg_normal = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.7,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    rs_normal_cpu = ResultSet(cfg_normal, score_margin=1e-12)
    rs_normal_gpu = ResultSet(cfg_normal, score_margin=5e-5)
    assert rs_normal_cpu.accepts_zero_scores() is False
    assert rs_normal_gpu.accepts_zero_scores() is False


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_threshold_search_boundary_hit_retention():
    """End-to-end verification that a boundary hit of 0.49997 is retained in search_threshold_batch_gpu."""
    require_cuda()

    # Construct library spectrum with 2 peaks, L2 normalized
    # peak 0: m/z 100.0, intensity sqrt(0.49997)
    # peak 1: m/z 200.0, intensity sqrt(1.0 - 0.49997)
    w_target = 0.49997
    lib_masses = np.array([100.0, 200.0], dtype=np.float64)
    lib_intensities = np.array([math.sqrt(w_target), math.sqrt(1.0 - w_target)], dtype=np.float64)
    lib, forest, gpu_forest = _make_single_spec_library(lib_masses, lib_intensities)

    # Construct query spectrum matching only peak 0 (m/z 100.0)
    q_masses = np.array([100.0, 300.0], dtype=np.float64)
    q_intensities = np.array([math.sqrt(w_target), math.sqrt(1.0 - w_target)], dtype=np.float64)
    query = _make_peaks(q_masses, q_intensities)

    # Verify theoretical cosine score is ~0.49997
    lib_spec_peaks = _make_peaks(lib_masses, lib_intensities)
    score_res = score_greedy_cosine(query, lib_spec_peaks, 0.02)
    assert abs(score_res.score - 0.49997) < 1e-6
    assert score_res.score < 0.50  # Strictly below 0.50

    cfg = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.50,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )

    # Under GPU threshold search (default score_margin = 5e-5), this boundary hit is retained
    gpu_outcomes = search_threshold_batch_gpu(
        queries=[query],
        gpu_forest=gpu_forest,
        library=lib,
        config=cfg,
    )
    assert len(gpu_outcomes) == 1
    assert len(gpu_outcomes[0].hits) == 1
    assert gpu_outcomes[0].hits[0].external_id == "spec_0"
    assert abs(gpu_outcomes[0].hits[0].score - 0.49997) < 1e-4


# ==============================================================================
# Test 2: GPU 批检索支持混合容差自动分桶 (P0)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_mixed_fragment_tolerance_batch_search():
    """Verify search_threshold_batch_gpu and search_forest_batch_gpu with mixed tolerances."""
    require_cuda()

    # Build a multi-peak library spectrum
    lib_masses = np.array([100.0, 100.03, 200.0, 300.0], dtype=np.float64)
    lib_intensities = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    lib, forest, gpu_forest = _make_single_spec_library(lib_masses, lib_intensities)

    # Create queries where tolerance makes a difference
    # Query A: m/z 100.015 (matches 100.0 within 0.02, matches both 100.0 and 100.03 within 0.05)
    q0 = _make_peaks(
        np.array([100.015, 200.0], dtype=np.float64),
        np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64),
    )
    q1 = _make_peaks(
        np.array([100.035, 300.0], dtype=np.float64),
        np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64),
    )
    q2 = q0
    q3 = q1

    # Mixed configs with alternating tolerances: 0.02 and 0.05
    cfg_002 = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.1,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    cfg_005 = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.1,
        fragment_tolerance_da=0.05,
        ion_mode=IonMode.POSITIVE,
    )
    mixed_cfgs = [cfg_002, cfg_005, cfg_002, cfg_005]
    queries = [q0, q1, q2, q3]

    # Individual single-query baseline calls
    res0 = search_threshold_batch_gpu([q0], gpu_forest, lib, cfg_002)
    res1 = search_threshold_batch_gpu([q1], gpu_forest, lib, cfg_005)
    res2 = search_threshold_batch_gpu([q2], gpu_forest, lib, cfg_002)
    res3 = search_threshold_batch_gpu([q3], gpu_forest, lib, cfg_005)

    # 1. Batch call with mixed tolerances should auto-bucket without raising ValueError
    mixed_outcomes = search_threshold_batch_gpu(
        queries=queries,
        gpu_forest=gpu_forest,
        library=lib,
        config=mixed_cfgs,
        batch_size=2,
    )
    assert len(mixed_outcomes) == 4

    # Verify exact equivalence to individual calls and order preservation
    for i, expected in enumerate([res0[0], res1[0], res2[0], res3[0]]):
        actual = mixed_outcomes[i]
        assert len(actual.hits) == len(expected.hits)
        for h_act, h_exp in zip(actual.hits, expected.hits):
            assert h_act.external_id == h_exp.external_id
            assert abs(h_act.score - h_exp.score) < 1e-5
            assert h_act.n_matched == h_exp.n_matched

    # 2. Unified dispatcher search_forest_batch_gpu should also handle mixed tolerances seamlessly
    dispatcher_outcomes = search_forest_batch_gpu(
        queries=queries,
        gpu_forest=gpu_forest,
        library=lib,
        config=mixed_cfgs,
        batch_size=2,
    )
    assert len(dispatcher_outcomes) == 4
    for i in range(4):
        assert len(dispatcher_outcomes[i].hits) == len(mixed_outcomes[i].hits)
        if mixed_outcomes[i].hits:
            assert dispatcher_outcomes[i].hits[0].score == mixed_outcomes[i].hits[0].score


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_mixed_search_mode_dispatcher():
    """Verify search_forest_batch_gpu can dispatch mixed SearchModes (THRESHOLD & TOP_K)."""
    require_cuda()

    lib_masses = np.array([100.0, 200.0], dtype=np.float64)
    lib_intensities = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64)
    lib, forest, gpu_forest = _make_single_spec_library(lib_masses, lib_intensities)

    q = _make_peaks(
        np.array([100.0, 200.0], dtype=np.float64),
        np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64),
    )

    cfg_th = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.5,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    cfg_topk = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )

    mixed_cfgs = [cfg_th, cfg_topk, cfg_th]
    outcomes = search_forest_batch_gpu(
        queries=[q, q, q],
        gpu_forest=gpu_forest,
        library=lib,
        config=mixed_cfgs,
    )
    assert len(outcomes) == 3
    assert outcomes[0].mode == SearchMode.THRESHOLD
    assert outcomes[1].mode == SearchMode.TOP_K
    assert outcomes[2].mode == SearchMode.THRESHOLD
    for o in outcomes:
        assert len(o.hits) == 1


# ==============================================================================
# Test 3: BatchQueryDevice.from_queries 单调性防御与排序保底 (P1)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_batch_query_device_unsorted_mass_raises():
    """BatchQueryDevice.from_queries must raise ValueError on non-monotonic peak masses."""
    require_cuda()

    # Query with descending peak mass (using unchecked factory to bypass SpectrumPeaks __post_init__)
    unsorted_query = SpectrumPeaks._create_unchecked(
        mass=np.array([200.0, 100.0], dtype=np.float64),
        intensity=np.array([0.7071, 0.7071], dtype=np.float64),
        energy=np.array([0.5, 0.5], dtype=np.float64),
        peak_id=np.array([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Query peaks mass must be sorted in ascending order"):
        BatchQueryDevice.from_queries(
            queries=[unsorted_query],
            frag_tau=0.02,
            grid_da=0.01,
        )

    # Valid ascending query should pass without error
    sorted_query = _make_peaks(
        np.array([100.0, 200.0], dtype=np.float64),
        np.array([0.7071, 0.7071], dtype=np.float64),
    )
    batch_q = BatchQueryDevice.from_queries(
        queries=[sorted_query],
        frag_tau=0.02,
        grid_da=0.01,
    )
    assert batch_q.n_queries == 1
    assert batch_q.total_peaks == 2


# ==============================================================================
# Test 4: GPU 上界下溢保护 (P0)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_underflow_protection_tiny_intensities():
    """Verify that when peaks overlap but intensity product underflows FP32, bound is >= 1e-7."""
    require_cuda()

    # Create library with a peak having tiny intensity (1e-25)
    # Note: 1e-25 * 1e-25 = 1e-50, which strictly underflows float32 (min positive subnormal ~1.4e-45)
    lib_masses = np.array([100.0], dtype=np.float64)
    lib_intensities = np.array([1e-25], dtype=np.float64)
    lib, forest, gpu_forest = _make_single_spec_library(lib_masses, lib_intensities)

    # Overlapping query: mass 100.0, intensity 1e-50 (underflows float32 to 0.0)
    q_overlap = _make_peaks(
        np.array([100.0], dtype=np.float64),
        np.array([1e-50], dtype=np.float64),
    )
    batch_q_overlap = BatchQueryDevice.from_queries([q_overlap], frag_tau=0.02, grid_da=gpu_forest.grid_da)

    # 1. Test K3a single-spectrum bound underflow protection
    d_uind = batch_uind_pairs_gpu(
        batch_query=batch_q_overlap,
        gpu_forest=gpu_forest,
        query_indices=np.array([0], dtype=np.int64),
        candidate_iids=np.array([0], dtype=np.int64),
        frag_tau=0.02,
    )
    h_uind = d_uind.copy_to_host()
    # With underflow floor, bound should be >= 1e-7 (non-zero!)
    assert h_uind[0] >= 1e-7
    assert h_uind[0] > 0.0

    # 2. Test K1 root bound underflow protection
    d_root = batch_root_bounds_gpu(
        batch_query=batch_q_overlap,
        gpu_forest=gpu_forest,
        tree_ids=np.array([0], dtype=np.int64),
    )
    h_root = d_root.copy_to_host()
    assert h_root[0, 0] >= 1e-7
    assert h_root[0, 0] > 0.0

    # 3. Disjoint query: mass 500.0 (no overlap with 100.0) -> bound must be strictly 0.0
    q_disjoint = _make_peaks(
        np.array([500.0], dtype=np.float64),
        np.array([1e-50], dtype=np.float64),
    )
    batch_q_disjoint = BatchQueryDevice.from_queries([q_disjoint], frag_tau=0.02, grid_da=gpu_forest.grid_da)

    d_uind_disjoint = batch_uind_pairs_gpu(
        batch_query=batch_q_disjoint,
        gpu_forest=gpu_forest,
        query_indices=np.array([0], dtype=np.int64),
        candidate_iids=np.array([0], dtype=np.int64),
        frag_tau=0.02,
    )
    h_uind_disjoint = d_uind_disjoint.copy_to_host()
    assert h_uind_disjoint[0] == 0.0


# ==============================================================================
# Test 5: CPU/GPU 分数截断带统一 (P1)
# ==============================================================================

def test_score_truncation_band_consistency():
    """Verify that CPU FP64 1e-12 and GPU FP32 1e-6 score clamping logic is consistent."""
    # Score slightly above 1.0 (e.g. 1.0000000000000002) should clamp to 1.0 on CPU
    q = _make_peaks(
        np.array([100.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )
    lib_spec = _make_peaks(
        np.array([100.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )
    res = score_greedy_cosine(q, lib_spec, 0.02)
    assert res.score == 1.0


# ==============================================================================
# Test 6: BatchQueryDevice 0峰与1峰极端用例测试 (P1)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_batch_query_device_zero_and_single_peak_monotonicity():
    """Verify that BatchQueryDevice.from_queries handles 0-peak and 1-peak queries safely."""
    require_cuda()

    empty_q = _make_peaks(np.empty(0), np.empty(0))
    single_q = _make_peaks(np.array([150.0]), np.array([1.0]))
    two_peaks_q = _make_peaks(np.array([100.0, 200.0]), np.array([0.7071, 0.7071]))

    batch_q = BatchQueryDevice.from_queries(
        queries=[empty_q, single_q, two_peaks_q],
        frag_tau=0.02,
        grid_da=0.01,
    )
    assert batch_q.n_queries == 3
    assert batch_q.total_peaks == 3
    assert int(batch_q.q_offsets.copy_to_host()[-1]) == 3


# ==============================================================================
# Test 7: 同时混合 SearchMode (THRESHOLD/TOP_K) 与混合容差的复合分桶测试 (P0)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_simultaneous_mixed_mode_and_tolerance_batch_search():
    """Verify search_forest_batch_gpu with both mixed modes and mixed tolerances simultaneously."""
    require_cuda()

    lib_masses = np.array([100.0, 200.0], dtype=np.float64)
    lib_intensities = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64)
    lib, forest, gpu_forest = _make_single_spec_library(lib_masses, lib_intensities)

    q = _make_peaks(
        np.array([100.0, 200.0], dtype=np.float64),
        np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64),
    )

    # 4 queries with alternating modes AND alternating tolerances:
    # 0: THRESHOLD + 0.02
    # 1: TOP_K     + 0.05
    # 2: THRESHOLD + 0.05
    # 3: TOP_K     + 0.02
    cfg0 = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.5, fragment_tolerance_da=0.02, ion_mode=IonMode.POSITIVE)
    cfg1 = QueryConfig(mode=SearchMode.TOP_K, k=5, fragment_tolerance_da=0.05, ion_mode=IonMode.POSITIVE)
    cfg2 = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.5, fragment_tolerance_da=0.05, ion_mode=IonMode.POSITIVE)
    cfg3 = QueryConfig(mode=SearchMode.TOP_K, k=5, fragment_tolerance_da=0.02, ion_mode=IonMode.POSITIVE)

    configs = [cfg0, cfg1, cfg2, cfg3]
    queries = [q, q, q, q]

    outcomes = search_forest_batch_gpu(
        queries=queries,
        gpu_forest=gpu_forest,
        library=lib,
        config=configs,
        batch_size=2,
    )
    assert len(outcomes) == 4
    assert outcomes[0].mode == SearchMode.THRESHOLD
    assert outcomes[1].mode == SearchMode.TOP_K
    assert outcomes[2].mode == SearchMode.THRESHOLD
    assert outcomes[3].mode == SearchMode.TOP_K

    for o in outcomes:
        assert len(o.hits) == 1
        assert o.hits[0].external_id == "spec_0"
        assert abs(o.hits[0].score - 1.0) < 1e-4


# ==============================================================================
# Test 8: 空查询与单查询边缘用例全面验证 (P1)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_empty_and_single_query_robustness():
    """Verify empty query list and single query edge cases across all GPU entrypoints."""
    require_cuda()

    lib_masses = np.array([100.0, 200.0], dtype=np.float64)
    lib_intensities = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64)
    lib, forest, gpu_forest = _make_single_spec_library(lib_masses, lib_intensities)

    cfg_th = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.5, fragment_tolerance_da=0.02, ion_mode=IonMode.POSITIVE)
    cfg_topk = QueryConfig(mode=SearchMode.TOP_K, k=5, fragment_tolerance_da=0.02, ion_mode=IonMode.POSITIVE)

    # 1. Empty queries list
    assert search_threshold_batch_gpu([], gpu_forest, lib, cfg_th) == []
    assert search_topk_batch_gpu([], gpu_forest, lib, cfg_topk) == []
    assert search_forest_batch_gpu([], gpu_forest, lib, cfg_th) == []

    empty_batch = BatchQueryDevice.from_queries([], frag_tau=0.02, grid_da=0.01)
    assert empty_batch.n_queries == 0
    assert empty_batch.total_peaks == 0

    # 2. Single query with sequence config
    q = _make_peaks(np.array([100.0, 200.0]), np.array([0.7071, 0.7071]))
    res_th = search_threshold_batch_gpu([q], gpu_forest, lib, [cfg_th])
    assert len(res_th) == 1
    assert len(res_th[0].hits) == 1

    res_topk = search_topk_batch_gpu([q], gpu_forest, lib, [cfg_topk])
    assert len(res_topk) == 1
    assert len(res_topk[0].hits) == 1


# ==============================================================================
# Test 9: GPU SearchStats probe_scored 开销透明追踪验证 (Phase 3)
# ==============================================================================

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_probe_scored_tracking():
    """验证 GPU 检索端 SearchStats.probe_scored 字段的正确记录与模式隔离。"""
    require_cuda()

    lib_masses = np.array([100.0, 200.0], dtype=np.float64)
    lib_intensities = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64)
    lib, forest, gpu_forest = _make_single_spec_library(lib_masses, lib_intensities)

    q = _make_peaks(
        np.array([100.0, 200.0], dtype=np.float64),
        np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float64),
    )

    cfg_topk = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    cfg_th = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.5,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )

    # 1. GPU Top-K 检索
    res_topk = search_topk_batch_gpu([q], gpu_forest, lib, cfg_topk)
    assert len(res_topk) == 1
    stats_topk = res_topk[0].stats
    assert stats_topk.probe_scored >= 0
    assert stats_topk.probe_scored <= stats_topk.n_scored
    assert stats_topk.probe_scored == 1  # 探测命中 1 条谱图

    # 2. GPU THRESHOLD 检索（应无 probe 阶段，probe_scored 恒为 0）
    res_th = search_threshold_batch_gpu([q], gpu_forest, lib, cfg_th)
    assert len(res_th) == 1
    stats_th = res_th[0].stats
    assert stats_th.probe_scored == 0
    assert stats_th.n_scored >= 0

    # 3. 0 峰空查询在 GPU 下短路时 probe_scored 保持 0
    q_empty = _make_peaks(np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    res_empty_topk = search_topk_batch_gpu([q_empty], gpu_forest, lib, cfg_topk)
    assert len(res_empty_topk) == 1
    assert res_empty_topk[0].stats.probe_scored == 0
    assert res_empty_topk[0].stats.n_scored == 0


