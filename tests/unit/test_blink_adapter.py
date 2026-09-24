"""BLINK 适配器与基准测试引擎单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from jetf.benchmarks.blink_adapter import (
    BlinkBenchmarkEngine,
    blink_to_jetf_peaks,
    check_blink_available,
    jetf_peaks_to_blink,
    score_blink_pair,
)
from jetf.types import SpectrumPeaks


@pytest.fixture(autouse=True)
def ensure_blink():
    check_blink_available()


def make_empty_peaks() -> SpectrumPeaks:
    return SpectrumPeaks._create_unchecked(
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.int64),
        0.0,
    )


def make_test_peaks(mz_list: list[float], int_list: list[float]) -> SpectrumPeaks:
    mass = np.array(mz_list, dtype=np.float64)
    raw_intensity = np.array(int_list, dtype=np.float64)
    norm = float(np.linalg.norm(raw_intensity))
    normalized = raw_intensity / norm if norm > 0 else np.zeros_like(raw_intensity)
    energy = normalized**2
    peak_id = np.arange(len(mass), dtype=np.int64)
    return SpectrumPeaks._create_unchecked(mass, normalized, energy, peak_id, norm)


def test_conversion_roundtrip():
    """测试 SpectrumPeaks 与 BLINK 2D 数组之间的转换正确性。"""
    p = make_test_peaks([100.0, 200.0, 300.0], [10.0, 50.0, 100.0])
    mzi = jetf_peaks_to_blink(p, use_raw_intensity=True)
    assert mzi.shape == (2, 3)
    np.testing.assert_allclose(mzi[0], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(mzi[1], [10.0, 50.0, 100.0])

    # 转换回 SpectrumPeaks
    p2 = blink_to_jetf_peaks(mzi)
    assert len(p2.mass) == 3
    np.testing.assert_allclose(p2.mass, p.mass)
    np.testing.assert_allclose(p2.intensity, p.intensity)


def test_conversion_empty():
    """测试空谱转换。"""
    empty_peaks = make_empty_peaks()
    mzi = jetf_peaks_to_blink(empty_peaks)
    assert mzi.shape == (2, 0)

    p_back = blink_to_jetf_peaks(mzi)
    assert len(p_back.mass) == 0


def test_score_blink_pair_basic():
    """测试常规谱对打分及区间。"""
    p1 = make_test_peaks([100.0, 200.0], [1.0, 2.0])
    p2 = make_test_peaks([100.005, 200.0], [1.0, 2.0])

    score, n_matched = score_blink_pair(p1, p2, tolerance=0.01, bin_width=0.001)
    assert 0.0 <= score <= 1.0
    assert n_matched == 2
    assert score > 0.999


def test_score_blink_pair_consistency_with_native():
    """测试适配器打分与原生 BLINK 直接打分完全一致。"""
    from blink import blink as bk

    m1 = np.array([[105.1, 205.2, 305.3], [10.0, 30.0, 60.0]], dtype=np.float64)
    m2 = np.array([[105.105, 205.2, 400.0], [10.0, 30.0, 20.0]], dtype=np.float64)

    # 原生打分
    disc = bk.discretize_spectra(
        [m1], [m2], [0.0], [0.0], tolerance=0.01, bin_width=0.001, intensity_power=0.5
    )
    native_res = bk.score_sparse_spectra(disc)
    native_score = float(native_res["mzi"].toarray()[0, 0])
    native_matches = int(round(float(native_res["mzc"].toarray()[0, 0])))

    # 适配器打分
    adapter_score, adapter_matches = score_blink_pair(m1, m2, tolerance=0.01, bin_width=0.001)
    assert abs(adapter_score - native_score) < 1e-9
    assert adapter_matches == native_matches


def test_score_blink_pair_empty_and_zero():
    """测试空谱或全零谱打分不崩溃并返回 0。"""
    p_normal = make_test_peaks([100.0, 200.0], [1.0, 2.0])
    p_empty = make_empty_peaks()
    p_zero = make_test_peaks([100.0, 200.0], [0.0, 0.0])

    s1, c1 = score_blink_pair(p_normal, p_empty)
    assert s1 == 0.0 and c1 == 0

    s2, c2 = score_blink_pair(p_empty, p_normal)
    assert s2 == 0.0 and c2 == 0

    s3, c3 = score_blink_pair(p_zero, p_normal)
    assert s3 == 0.0 and c3 == 0


def test_benchmark_engine_search_single():
    """测试 BlinkBenchmarkEngine 的 1-to-N 检索正确性与降序排列。"""
    lib_peaks = [
        make_test_peaks([100.0, 200.0], [1.0, 2.0]),        # 完美匹配 query
        make_test_peaks([100.005, 200.0], [1.0, 1.0]),     # 部分匹配
        make_test_peaks([500.0, 600.0], [1.0, 1.0]),        # 无匹配
        make_empty_peaks(),                                 # 空谱
    ]
    engine = BlinkBenchmarkEngine(lib_peaks, tolerance=0.01, bin_width=0.001)

    query = make_test_peaks([100.0, 200.0], [1.0, 2.0])
    res = engine.search_single(query, top_k=3)

    assert len(res.indices) == 3
    assert len(res.scores) == 3
    assert len(res.counts) == 3

    # 最高分应为 index 0，得分约 1.0
    assert res.indices[0] == 0
    assert res.scores[0] > 0.999
    assert res.counts[0] == 2

    # 得分应单调递减
    assert np.all(res.scores[:-1] >= res.scores[1:])

    # 耗时统计合法性
    assert res.discretize_time_s >= 0.0
    assert res.score_time_s >= 0.0
    assert res.total_time_s >= res.score_time_s

    # 元组解构测试
    scores, indices = res
    np.testing.assert_array_equal(scores, res.scores)
    np.testing.assert_array_equal(indices, res.indices)


def test_benchmark_engine_score_batch():
    """测试 BlinkBenchmarkEngine 批量检索与单条检索结果一致性。"""
    lib_peaks = [
        make_test_peaks([100.0, 200.0], [1.0, 2.0]),
        make_test_peaks([150.0, 250.0], [3.0, 4.0]),
        make_test_peaks([300.0, 400.0], [1.0, 1.0]),
    ]
    engine = BlinkBenchmarkEngine(lib_peaks, tolerance=0.01, bin_width=0.001)

    queries = [
        make_test_peaks([100.0, 200.0], [1.0, 2.0]),
        make_test_peaks([150.0, 250.0], [3.0, 4.0]),
    ]

    batch_res = engine.score_batch(queries, top_k=2)
    assert batch_res.n_queries == 2
    assert len(batch_res.indices) == 2

    # 单条对比
    single_0 = engine.search_single(queries[0], top_k=2)
    single_1 = engine.search_single(queries[1], top_k=2)

    np.testing.assert_array_equal(batch_res.indices[0], single_0.indices)
    np.testing.assert_allclose(batch_res.scores[0], single_0.scores)

    np.testing.assert_array_equal(batch_res.indices[1], single_1.indices)
    np.testing.assert_allclose(batch_res.scores[1], single_1.scores)


def test_blink_to_jetf_peaks_1d_and_invalid_shapes():
    """测试 1D 数组、非法维度与包含 NaN/Inf 的数据转换安全性。"""
    # 1D 数组不应引发 IndexError
    p_1d = blink_to_jetf_peaks(np.array([100.0, 1.0]))
    assert len(p_1d.mass) == 0

    # 3x2 形状
    p_3row = blink_to_jetf_peaks(np.ones((3, 2)))
    assert len(p_3row.mass) == 0

    # 包含 NaN
    p_nan = blink_to_jetf_peaks(np.array([[100.0, np.nan], [1.0, 2.0]]))
    assert len(p_nan.mass) == 1
    assert p_nan.mass[0] == 100.0


def test_score_blink_pair_edge_cases():
    """测试逐对打分算子在 1D 数组、NaN、Inf 及负数下的鲁棒性。"""
    p_normal = make_test_peaks([100.0, 200.0], [1.0, 2.0])

    # 1D 输入
    s, c = score_blink_pair(np.array([100.0, 1.0]), p_normal)
    assert s == 0.0 and c == 0

    # NaN / Inf 输入
    m_nan = np.array([[100.0, np.nan], [1.0, 2.0]])
    s, c = score_blink_pair(m_nan, p_normal)
    assert s == 0.0 and c == 0

    m_inf = np.array([[100.0, 200.0], [1.0, np.inf]])
    s, c = score_blink_pair(p_normal, m_inf)
    assert s == 0.0 and c == 0


def test_benchmark_engine_empty_and_zero_library():
    """测试 BlinkBenchmarkEngine 对空库与全零库的检索表现。"""
    query = make_test_peaks([100.0, 200.0], [1.0, 2.0])

    # 1. 彻底空库
    engine_empty = BlinkBenchmarkEngine([], tolerance=0.01, bin_width=0.001)
    res_empty = engine_empty.search_single(query, top_k=5)
    assert len(res_empty.indices) == 0
    assert len(res_empty.scores) == 0

    batch_empty = engine_empty.score_batch([query], top_k=5)
    assert len(batch_empty.indices[0]) == 0

    # 2. 全由空谱构成的库
    engine_zero = BlinkBenchmarkEngine([make_empty_peaks(), make_empty_peaks()])
    res_zero = engine_zero.search_single(query, top_k=5)
    assert len(res_zero.indices) == 2
    assert np.all(res_zero.scores == 0.0)


def test_benchmark_engine_extreme_mz_and_disjoint():
    """测试超大 m/z、负 m/z 与完全互不相交的谱检索。"""
    lib_peaks = [
        make_test_peaks([100.0, 200.0], [1.0, 2.0]),
        make_test_peaks([150.0, 250.0], [1.0, 1.0]),
    ]
    engine = BlinkBenchmarkEngine(lib_peaks, tolerance=0.01, bin_width=0.001)

    # 1. 互不相交谱
    q_disjoint = make_test_peaks([800.0, 900.0], [1.0, 1.0])
    res_disjoint = engine.search_single(q_disjoint, top_k=2)
    assert len(res_disjoint.indices) == 2
    assert np.all(res_disjoint.scores == 0.0)
    assert np.all(res_disjoint.counts == 0)

    # 2. 超大 m/z (超过有效 max_bin)
    q_huge = np.array([[8000.0, 9000.0], [1.0, 2.0]], dtype=np.float64)
    res_huge = engine.search_single(q_huge, top_k=2)
    assert len(res_huge.indices) == 2
    assert np.all(res_huge.scores == 0.0)

    # 3. 负 m/z
    q_neg = np.array([[-10.0, -5.0], [1.0, 1.0]], dtype=np.float64)
    res_neg = engine.search_single(q_neg, top_k=2)
    assert len(res_neg.indices) == 2
    assert np.all(res_neg.scores == 0.0)


def test_benchmark_engine_topk_larger_than_library():
    """测试 Top-K 远大于库容量时的处理。"""
    lib_peaks = [
        make_test_peaks([100.0, 200.0], [1.0, 2.0]),
        make_test_peaks([100.0, 300.0], [1.0, 1.0]),
    ]
    engine = BlinkBenchmarkEngine(lib_peaks, tolerance=0.01, bin_width=0.001)
    query = make_test_peaks([100.0, 200.0], [1.0, 2.0])

    res = engine.search_single(query, top_k=50)
    assert len(res.indices) == 2
    assert len(res.scores) == 2
    assert res.indices[0] == 0


def test_benchmark_engine_batch_with_mixed_queries():
    """测试批量查询中混合空谱、包含非有限值与正常谱的鲁棒性。"""
    lib_peaks = [
        make_test_peaks([100.0, 200.0], [1.0, 2.0]),
    ]
    engine = BlinkBenchmarkEngine(lib_peaks, tolerance=0.01, bin_width=0.001)

    queries = [
        make_test_peaks([100.0, 200.0], [1.0, 2.0]),            # 正常匹配
        make_empty_peaks(),                                      # 空谱
        np.array([[100.0, np.nan], [1.0, 2.0]]),                 # NaN 谱
        np.array([[8000.0], [1.0]]),                             # 超界谱
    ]

    batch_res = engine.score_batch(queries, top_k=1)
    assert batch_res.n_queries == 4
    assert batch_res.scores[0][0] > 0.999
    assert batch_res.scores[1][0] == 0.0
    assert batch_res.scores[2][0] == 0.0
    assert batch_res.scores[3][0] == 0.0
