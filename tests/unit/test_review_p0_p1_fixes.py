"""单元测试：验证 P0/P1 修复项与规范化整改。

包含：
1. 量化偏置边缘用例 (m_lib = 100.0 - 10^-13, m_q = 99.98 - 10^-13, 容差 0.02 Da)，断言 0 漏检；
2. 全 0 强度谱建库，验证列等长校验、微块对齐与无越界；
3. 阈值检索临界分一致性 [threshold - 10^-12, threshold - 10^-13]；
4. 0 峰空谱快速短路正确性 (nodes_visited == 0)。
"""

from __future__ import annotations

import numpy as np
import pytest

from jetf import (
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    IonMode,
    ParsedLibrary,
    PrecursorWindow,
    QueryConfig,
    SearchMode,
    SourceRef,
    SpectrumMeta,
    SpectrumPeaks,
    build_forest_index,
    preprocess_library,
    preprocess_query,
    score_greedy_cosine,
    search_exhaustive,
    search_forest,
)
from jetf.preprocessing import LibraryPeaks
from jetf.bounds import window_cells
from jetf.results import ResultSet, SearchHit


def test_quantization_bias_edge_case():
    """1. 量化偏置边缘用例：验证极限距离下无假阴性漏检。

    m_lib = 100.0 - 1e-13
    m_q = 99.98 - 1e-13
    容差 0.02 Da。
    若无 +1e-12 偏置，window_cells 计算的 upper 为 4999，而建库 cell 为 5000，会导致假阴性漏检。
    修复后 lower 与 upper 加上 +1e-12，upper 正确包含 5000。
    """
    grid_da = 0.02
    tolerance_da = 0.02
    m_lib = 100.0 - 1e-13
    m_q = 99.98 - 1e-13

    # 1.1 验证 window_cells 行为
    q_mass = np.array([m_q], dtype=np.float64)
    c_low, c_high = window_cells(q_mass, tolerance_da, grid_da)
    expected_lib_cell = int(np.floor((m_lib + 1e-12) / grid_da))
    assert expected_lib_cell == 5000
    assert c_high[0] >= expected_lib_cell, f"c_high={c_high[0]} 应 >= expected_lib_cell={expected_lib_cell}"

    # 1.2 端到端建库与检索测试
    metas = [
        SpectrumMeta(
            external_id="edge_lib_0",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("edge.mgf", 0),
            raw_metadata={},
        ),
        SpectrumMeta(
            external_id="other_lib_1",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("edge.mgf", 1),
            raw_metadata={},
        ),
    ]

    # edge_lib_0 在 m_lib 处有单峰
    m0 = np.array([m_lib], dtype=np.float64)
    it0 = np.array([100.0], dtype=np.float64)
    pid0 = np.array([0], dtype=np.int64)

    # other_lib_1 在远离区域有单峰
    m1 = np.array([300.0], dtype=np.float64)
    it1 = np.array([50.0], dtype=np.float64)
    pid1 = np.array([0], dtype=np.int64)

    parsed = ParsedLibrary(
        source_path="edge.mgf",
        spectra=tuple(metas),
        mass=np.concatenate([m0, m1]),
        intensity=np.concatenate([it0, it1]),
        peak_id=np.concatenate([pid0, pid1]),
        spectrum_offsets=np.array([0, 1, 2], dtype=np.int64),
    )
    lib = preprocess_library(parsed)
    forest = build_forest_index(lib)

    raw_query = SpectrumPeaks._create_unchecked(
        mass=np.array([m_q], dtype=np.float64),
        intensity=np.array([10.0], dtype=np.float64),
        energy=np.array([100.0], dtype=np.float64),
        peak_id=np.array([0], dtype=np.int64),
        norm=10.0,
    )
    query = preprocess_query(raw_query)

    cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=10,
        fragment_tolerance_da=tolerance_da,
        ion_mode=IonMode.POSITIVE,
    )

    res_forest = search_forest(query, forest, lib, cfg)
    res_exh = search_exhaustive(query, lib, cfg)

    assert len(res_forest.hits) == len(res_exh.hits) == 1
    assert res_forest.hits[0].external_id == "edge_lib_0"
    assert abs(res_forest.hits[0].score - 1.0) <= 1e-12


def test_zero_intensity_library_build_and_alignment():
    """2. 全 0 强度谱建库：验证列等长校验、微块对齐与无越界。"""
    metas = [
        SpectrumMeta(
            external_id="zero_spec",
            precursor_mz=400.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("test.mgf", 0),
            raw_metadata={},
        ),
        SpectrumMeta(
            external_id="normal_spec",
            precursor_mz=401.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("test.mgf", 1),
            raw_metadata={},
        ),
    ]

    # zero_spec 有 3 个峰，但强度全为 0
    m0 = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    it0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pid0 = np.arange(3, dtype=np.int64)

    # normal_spec 正常
    m1 = np.array([150.0, 250.0], dtype=np.float64)
    it1 = np.array([10.0, 20.0], dtype=np.float64)
    pid1 = np.arange(2, dtype=np.int64)

    parsed = ParsedLibrary(
        source_path="test.mgf",
        spectra=tuple(metas),
        mass=np.concatenate([m0, m1]),
        intensity=np.concatenate([it0, it1]),
        peak_id=np.concatenate([pid0, pid1]),
        spectrum_offsets=np.array([0, 3, 5], dtype=np.int64),
    )

    lib = preprocess_library(parsed)

    # 断言 LibraryPeaks 多列等长
    assert lib.peaks.mass.shape == lib.peaks.intensity.shape == lib.peaks.energy.shape == lib.peaks.peak_id.shape
    assert lib.peaks.n_peaks == 5

    # 验证切片对齐
    sp0 = lib.peaks.spectrum_at(0)
    assert sp0.mass.shape == sp0.intensity.shape == (3,)
    assert np.all(sp0.intensity == 0.0)
    assert sp0.norm == 0.0

    sp1 = lib.peaks.spectrum_at(1)
    assert sp1.mass.shape == sp1.intensity.shape == (2,)
    assert sp1.norm > 0.0

    # 索引构建测试：全 0 谱应被放入零能量侧车
    forest = build_forest_index(lib)
    assert forest.zero_energy_members.n_members == 1
    assert forest.zero_energy_members.member[0] == 0

    # 校验 LibraryPeaks 长度不一致时显式抛出 ValueError
    with pytest.raises(ValueError, match="列长度必须一致"):
        LibraryPeaks(
            mass=np.array([100.0, 200.0]),
            intensity=np.array([1.0]),  # 长度不同
            energy=np.array([1.0, 1.0]),
            peak_id=np.array([0, 1]),
            spectrum_offsets=np.array([0, 2]),
            norm=np.array([1.0]),
        )


def test_threshold_critical_boundary_consistency():
    """3. 阈值临界分一致性：验证得分落在 [threshold - 1e-12, threshold - 1e-13] 时森林与穷举完全一致。"""
    # 构造精确打分为 0.5000000000000000 的谱对
    # query: 峰 100.0 (强度 sqrt(0.5)), 峰 200.0 (强度 sqrt(0.5)) -> L2 norm = 1.0
    # target_lib: 峰 100.0 (强度 sqrt(0.5)), 峰 300.0 (强度 sqrt(0.5)) -> L2 norm = 1.0
    # 匹配峰为 100.0，cosine score = sqrt(0.5) * sqrt(0.5) = 0.5
    u = np.sqrt(0.5)
    metas = [
        SpectrumMeta(
            external_id="target_half",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("test.mgf", 0),
            raw_metadata={},
        ),
        SpectrumMeta(
            external_id="unrelated",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("test.mgf", 1),
            raw_metadata={},
        ),
    ]
    m0 = np.array([100.0, 300.0], dtype=np.float64)
    it0 = np.array([u, u], dtype=np.float64)
    pid0 = np.arange(2, dtype=np.int64)

    m1 = np.array([400.0, 500.0], dtype=np.float64)
    it1 = np.array([u, u], dtype=np.float64)
    pid1 = np.arange(2, dtype=np.int64)

    parsed = ParsedLibrary(
        source_path="test.mgf",
        spectra=tuple(metas),
        mass=np.concatenate([m0, m1]),
        intensity=np.concatenate([it0, it1]),
        peak_id=np.concatenate([pid0, pid1]),
        spectrum_offsets=np.array([0, 2, 4], dtype=np.int64),
    )
    lib = preprocess_library(parsed)
    forest = build_forest_index(lib)

    q_peaks = SpectrumPeaks._create_unchecked(
        mass=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([u, u], dtype=np.float64),
        energy=np.array([0.5, 0.5], dtype=np.float64),
        peak_id=np.arange(2, dtype=np.int64),
        norm=1.0,
    )

    # 实际打分
    exact_res = score_greedy_cosine(q_peaks, lib.peaks.spectrum_at(0))
    score = exact_res.score

    # 设置门槛比实际分数略高 5e-13 (处于 [score, score + 1e-12] 区间)
    threshold = score + 5e-13
    cfg = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=threshold,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )

    # ResultSet.theta() 应返回 threshold - 1e-12 < score，从而允许上界放行
    res_set = ResultSet(cfg)
    assert res_set.theta() == threshold - 1e-12
    assert score >= res_set.theta()

    # 森林检索与穷举检索应 100% 一致接收该命中
    res_forest = search_forest(q_peaks, forest, lib, cfg)
    res_exh = search_exhaustive(q_peaks, lib, cfg)

    assert len(res_forest.hits) == len(res_exh.hits) == 1
    assert res_forest.hits[0].external_id == "target_half"
    assert res_forest.hits[0].score == res_exh.hits[0].score


def test_zero_peak_query_fast_short_circuit():
    """4. 0 峰查询快速短路测试：验证空谱查询即时返回且结果结构有效。"""
    metas = [
        SpectrumMeta(
            external_id="lib_normal",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("test.mgf", 0),
            raw_metadata={},
        ),
        SpectrumMeta(
            external_id="lib_zero_energy",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("test.mgf", 1),
            raw_metadata={},
        ),
    ]

    m0 = np.array([100.0, 200.0], dtype=np.float64)
    it0 = np.array([10.0, 20.0], dtype=np.float64)
    pid0 = np.arange(2, dtype=np.int64)

    m1 = np.array([300.0], dtype=np.float64)
    it1 = np.array([0.0], dtype=np.float64)
    pid1 = np.arange(1, dtype=np.int64)

    parsed = ParsedLibrary(
        source_path="test.mgf",
        spectra=tuple(metas),
        mass=np.concatenate([m0, m1]),
        intensity=np.concatenate([it0, it1]),
        peak_id=np.concatenate([pid0, pid1]),
        spectrum_offsets=np.array([0, 2, 3], dtype=np.int64),
    )
    lib = preprocess_library(parsed)
    forest = build_forest_index(lib)

    empty_query = SpectrumPeaks(
        mass=np.empty(0, dtype=np.float64),
        intensity=np.empty(0, dtype=np.float64),
        energy=np.empty(0, dtype=np.float64),
        peak_id=np.empty(0, dtype=np.int64),
        norm=0.0,
    )

    # 4.1 min_matched_peaks >= 1 时：无任何命中，且 0 节点访问
    cfg_normal = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        min_matched_peaks=1,
        ion_mode=IonMode.POSITIVE,
    )
    res = search_forest(empty_query, forest, lib, cfg_normal)
    assert res.complete is True
    assert len(res.hits) == 0
    assert res.stats.nodes_visited == 0
    assert res.stats.n_scored == 0
    assert res.stats.bound_eval_time_ms == 0.0

    # 4.2 min_matched_peaks == 0 且接受 0 分时：零能量侧车补足正常生效
    cfg_zero = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        min_matched_peaks=0,
        ion_mode=IonMode.POSITIVE,
    )
    res_zero = search_forest(empty_query, forest, lib, cfg_zero)
    assert res_zero.complete is True
    assert res_zero.stats.nodes_visited == 0
    assert len(res_zero.hits) == 1
    assert res_zero.hits[0].external_id == "lib_zero_energy"
    assert res_zero.hits[0].score == 0.0
