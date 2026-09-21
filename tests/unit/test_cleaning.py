"""Unit tests for matchms spectrum cleaning and preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from jetf.builder import build_forest_index
from jetf.cleaning import (
    DEFAULT_CLEAN_CONFIG,
    MatchmsCleanConfig,
    clean_parsed_library,
    clean_spectrum_with_matchms,
)
from jetf.mgf import ParsedLibrary
from jetf.preprocessing import preprocess_library
from jetf.query import QueryConfig, SearchMode
from jetf.search import search_forest
from jetf.types import (
    INTENSITY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    PEAK_ID_DTYPE,
    IonMode,
    SourceRef,
    SpectrumMeta,
)


def _make_dummy_parsed_library() -> ParsedLibrary:
    """构造包含微弱噪声峰与长尾峰的测试库。"""
    source = SourceRef("dummy.mgf", 0)

    # 谱 0: 正常谱 (含 1 个强峰，1 个中峰，多个低于 0.001 的微弱噪声峰)
    m0 = np.array([100.0, 150.0, 200.0, 250.0, 300.0], dtype=MASS_DTYPE)
    it0 = np.array([1000.0, 200.0, 0.05, 0.01, 150.0], dtype=INTENSITY_DTYPE)
    pid0 = np.arange(len(m0), dtype=PEAK_ID_DTYPE)
    meta0 = SpectrumMeta(
        external_id="SPEC_0",
        ion_mode=IonMode.POSITIVE,
        precursor_mz=500.25,
        charge=1,
        source=source,
        raw_metadata={},
    )

    # 谱 1: 长尾大峰数谱 (10 个峰，用于测试 max_peaks 截断)
    m1 = np.linspace(100.0, 600.0, 10, dtype=MASS_DTYPE)
    it1 = np.arange(1.0, 11.0, dtype=INTENSITY_DTYPE) * 100.0  # 100 到 1000
    pid1 = np.arange(len(m1), dtype=PEAK_ID_DTYPE)
    meta1 = SpectrumMeta(
        external_id="SPEC_1",
        ion_mode=IonMode.POSITIVE,
        precursor_mz=650.50,
        charge=1,
        source=source,
        raw_metadata={},
    )

    # 谱 2: 极少峰谱 (只有 1 个峰，会被 min_peaks=3 过滤)
    m2 = np.array([120.0], dtype=MASS_DTYPE)
    it2 = np.array([500.0], dtype=INTENSITY_DTYPE)
    pid2 = np.arange(len(m2), dtype=PEAK_ID_DTYPE)
    meta2 = SpectrumMeta(
        external_id="SPEC_2",
        ion_mode=IonMode.POSITIVE,
        precursor_mz=300.0,
        charge=1,
        source=source,
        raw_metadata={},
    )

    mass = np.concatenate([m0, m1, m2])
    intensity = np.concatenate([it0, it1, it2])
    peak_id = np.concatenate([pid0, pid1, pid2])
    offsets = np.array([0, len(m0), len(m0) + len(m1), len(mass)], dtype=INTERNAL_ID_DTYPE)

    return ParsedLibrary(
        source_path="dummy.mgf",
        spectra=(meta0, meta1, meta2),
        mass=mass,
        intensity=intensity,
        peak_id=peak_id,
        spectrum_offsets=offsets,
    )


def test_clean_config_validation():
    """测试配置类的边界值合法性检查。"""
    cfg = MatchmsCleanConfig(max_peaks=100, min_relative_intensity=0.01)
    assert cfg.max_peaks == 100
    assert cfg.min_relative_intensity == 0.01

    with pytest.raises(ValueError, match="mz_min 必须非负"):
        MatchmsCleanConfig(mz_min=-1.0)

    with pytest.raises(ValueError, match="mz_max.*必须大于"):
        MatchmsCleanConfig(mz_min=500.0, mz_max=400.0)

    with pytest.raises(ValueError, match="max_peaks 必须为正整数"):
        MatchmsCleanConfig(max_peaks=0)


def test_clean_parsed_library_filtering():
    """测试 clean_parsed_library 对微弱峰截断、Top-N 峰截断与极少峰剔除。"""
    raw_lib = _make_dummy_parsed_library()
    assert raw_lib.n_spectra == 3
    assert raw_lib.n_peaks == 5 + 10 + 1

    cfg = MatchmsCleanConfig(
        max_peaks=5,                  # 谱 1 应该被截断为 5 个峰
        min_relative_intensity=0.001, # 谱 0 的 0.05 和 0.01 (相对于 1000.0 分别为 0.00005 和 0.00001) 会被剔除
        min_peaks=3,                  # 谱 2 只有 1 个峰，会被直接剔除
        metadata_cleaning=True,
    )

    cleaned_lib = clean_parsed_library(raw_lib, cfg)

    # 验证谱总数由 3 变为 2 (谱 2 被剔除)
    assert cleaned_lib.n_spectra == 2
    assert len(cleaned_lib.rejected) == 1
    assert "峰数不足" in cleaned_lib.rejected[0].detail

    # 验证谱 0: 原 5 个峰，其中两个微弱噪声被剔除，剩 3 个峰
    p0_count = cleaned_lib.spectrum_offsets[1] - cleaned_lib.spectrum_offsets[0]
    assert p0_count == 3
    p0_masses = cleaned_lib.mass[:p0_count]
    assert np.allclose(p0_masses, [100.0, 150.0, 300.0])

    # 验证谱 1: 原 10 个峰，被 max_peaks=5 截断为 5 个最高峰
    p1_count = cleaned_lib.spectrum_offsets[2] - cleaned_lib.spectrum_offsets[1]
    assert p1_count == 5

    # 验证整体连续拓扑与排序
    assert cleaned_lib.mass.shape[0] == 3 + 5
    assert cleaned_lib.spectrum_offsets[-1] == cleaned_lib.mass.shape[0]
    # 验证每条谱的 mass 严格升序
    assert np.all(cleaned_lib.mass[:3][:-1] <= cleaned_lib.mass[:3][1:])
    assert np.all(cleaned_lib.mass[3:][:-1] <= cleaned_lib.mass[3:][1:])


def test_cleaned_library_indexing_and_search():
    """验证清洗后的 ParsedLibrary 可以顺利预处理、建树并完成开放检索。"""
    raw_lib = _make_dummy_parsed_library()
    cleaned_lib = clean_parsed_library(raw_lib, MatchmsCleanConfig(max_peaks=5, min_peaks=2))

    prep_lib = preprocess_library(cleaned_lib)
    forest = build_forest_index(prep_lib)
    assert forest.n_spectra == cleaned_lib.n_spectra

    query = prep_lib.peaks.spectrum_at(0)
    cfg = QueryConfig(mode=SearchMode.TOP_K, k=5, ion_mode=prep_lib.spectra[0].ion_mode)
    outcome = search_forest(query, forest, prep_lib, cfg)

    # 自比得分应该为 1.0 且排在第一位
    assert len(outcome.hits) > 0
    assert outcome.hits[0].spectrum_index == 0
    assert pytest.approx(outcome.hits[0].score, abs=1e-6) == 1.0


def test_parse_mgf_streaming_cleaning(tmp_path):
    """验证 parse_mgf 流式清洗与离线 clean_parsed_library 结果完全一致。"""
    from jetf.mgf import parse_mgf

    mgf_file = tmp_path / "test_stream.mgf"
    mgf_content = """BEGIN IONS
TITLE=SPEC_GOOD
PEPMASS=400.0
CHARGE=1+
IONMODE=Positive
100.0 1000.0
150.0 500.0
200.0 0.05
250.0 300.0
END IONS
BEGIN IONS
TITLE=SPEC_BAD_FEW_PEAKS
PEPMASS=500.0
CHARGE=1+
IONMODE=Positive
100.0 1000.0
END IONS
"""
    mgf_file.write_text(mgf_content, encoding="utf-8")

    cfg = MatchmsCleanConfig(min_peaks=2, min_relative_intensity=0.001)

    # 1. 传统方式：解析后清洗
    parsed_raw = parse_mgf(mgf_file)
    cleaned_batch = clean_parsed_library(parsed_raw, config=cfg)

    # 2. 流式方式：解析中直接清洗
    cleaned_stream = parse_mgf(mgf_file, clean_config=cfg)

    assert cleaned_stream.n_spectra == cleaned_batch.n_spectra == 1
    assert cleaned_stream.spectra[0].external_id == "SPEC_GOOD"
    assert np.allclose(cleaned_stream.mass, cleaned_batch.mass)
    assert np.allclose(cleaned_stream.intensity, cleaned_batch.intensity)
    assert np.array_equal(cleaned_stream.peak_id, cleaned_batch.peak_id)
    assert np.array_equal(cleaned_stream.spectrum_offsets, cleaned_batch.spectrum_offsets)
    assert len(cleaned_stream.rejected) == len(cleaned_batch.rejected) == 1

