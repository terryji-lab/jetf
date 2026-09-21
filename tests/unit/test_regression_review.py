"""Regression tests verifying fixes for all issues identified in code-review-forest.md."""

from __future__ import annotations

import numpy as np
import pytest

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FOREST_SPEC,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    IonMode,
    ParsedLibrary,
    PreprocessSpec,
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
    single_spectrum_bound,
)
from jetf.mgf import _parse_ion_mode
from jetf.structure import ForestSpec


def _make_synth_library(n_spectra: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    metas, mass, inten, pid, offs = [], [], [], [], [0]
    for i in range(n_spectra):
        peaks = [
            (100.0, rng.uniform(500, 1000)),
            (200.0, rng.uniform(1, 20)),
            (300.0 + i * 0.5, 30.0),
        ]
        metas.append(
            SpectrumMeta(
                external_id=f"L{i}",
                precursor_mz=500.0,
                charge=1,
                ion_mode=IonMode.POSITIVE,
                source=SourceRef("synth.mgf", i),
                raw_metadata={},
            )
        )
        m = np.array([p[0] for p in peaks], dtype=np.float64)
        it = np.array([p[1] for p in peaks], dtype=np.float64)
        mass.append(m)
        inten.append(it)
        pid.append(np.arange(len(peaks), dtype=np.int64))
        offs.append(offs[-1] + len(peaks))

    parsed = ParsedLibrary(
        source_path="synth.mgf",
        spectra=tuple(metas),
        mass=np.concatenate(mass),
        intensity=np.concatenate(inten),
        peak_id=np.concatenate(pid),
        spectrum_offsets=np.array(offs, dtype=np.int64),
    )
    lib = preprocess_library(parsed)
    forest = build_forest_index(lib)
    return lib, forest


def test_p0_negative_intensity_clamped_and_consistent():
    """P0: 负强度查询经 preprocess_query 截断清洗后，森林与穷举检索结果完全一致（零漏检）。"""
    library, forest = _make_synth_library(50, seed=5)

    # 原始查询包含负峰（基线噪声）
    raw = np.array([-50.0, 40.0, 10.0])
    raw_peaks = SpectrumPeaks._create_unchecked(
        mass=np.array([100.0, 200.0, 300.0]),
        intensity=raw,
        energy=raw * raw,
        peak_id=np.arange(3, dtype=np.int64),
        norm=float(np.sqrt(np.sum(raw * raw))),
    )

    # 经过预处理守护：负峰截断为 0，归一化有效峰
    q = preprocess_query(raw_peaks)
    assert q.intensity[0] == 0.0
    assert q.intensity[1] > 0.0
    assert pytest.approx(float(np.sum(q.intensity**2)), abs=1e-6) == 1.0

    cfg = QueryConfig(mode=SearchMode.TOP_K, k=5, ion_mode=IonMode.POSITIVE)
    rf = search_forest(q, forest, library, cfg)
    re_ = search_exhaustive(q, library, cfg)

    assert rf.complete is True
    assert len(rf.hits) == len(re_.hits)
    for hf, he in zip(rf.hits, re_.hits):
        assert hf.external_id == he.external_id
        assert hf.score == he.score


def test_p0_negative_intensity_direct_construction_rejected():
    """P0: 直接构造含负强度的 SpectrumPeaks 必须抛出 ValueError。"""
    with pytest.raises(ValueError, match="包含负数"):
        SpectrumPeaks(
            mass=np.array([100.0, 200.0]),
            intensity=np.array([-1.0, 1.0]),
            energy=np.array([1.0, 1.0]),
            peak_id=np.array([0, 1], dtype=np.int64),
        )


def test_p1_1_unsorted_mass_rejected():
    """P1-1: 库谱 mass 乱序时，直接调用 score_greedy_cosine 或构造 SpectrumPeaks 必须报错。"""
    with pytest.raises(ValueError, match="按升序排列"):
        SpectrumPeaks(
            mass=np.array([300.0, 100.0, 200.0]),
            intensity=np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
            energy=np.array([1.0, 1.0, 1.0]) / 3,
            peak_id=np.arange(3, dtype=np.int64),
        )

    # 构造合规查询与非法无序库谱
    q = SpectrumPeaks(
        mass=np.array([100.0, 200.0]),
        intensity=np.array([1.0, 1.0]) / np.sqrt(2),
        energy=np.array([1.0, 1.0]) / 2,
        peak_id=np.arange(2, dtype=np.int64),
    )
    unsorted_lib = SpectrumPeaks._create_unchecked(
        mass=np.array([200.0, 100.0]),
        intensity=np.array([1.0, 1.0]) / np.sqrt(2),
        energy=np.array([1.0, 1.0]) / 2,
        peak_id=np.arange(2, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="按升序排列"):
        score_greedy_cosine(q, unsorted_lib)

    with pytest.raises(ValueError, match="按升序排列"):
        single_spectrum_bound(q, unsorted_lib)


def test_p1_2_library_mismatch_rejected():
    """P1-2: 森林索引与库不匹配时，search_forest 必须立即拦截。"""
    lib1, forest1 = _make_synth_library(30, seed=1)
    lib2, forest2 = _make_synth_library(40, seed=2)

    q = lib1.peaks.spectrum_at(0)
    cfg = QueryConfig(mode=SearchMode.TOP_K, k=5)

    with pytest.raises(ValueError, match="总谱数不一致|指纹.*不匹配"):
        search_forest(q, forest1, lib2, cfg)


def test_p1_3_unnormalized_query_rejected():
    """P1-3: 未归一化查询传入 search_forest 必须被明确拦截。"""
    library, forest = _make_synth_library(20, seed=3)

    raw_unnorm = SpectrumPeaks(
        mass=np.array([100.0, 200.0]),
        intensity=np.array([100.0, 200.0]),
        energy=np.array([10000.0, 40000.0]),
        peak_id=np.arange(2, dtype=np.int64),
        norm=1.0,
    )
    cfg = QueryConfig(mode=SearchMode.TOP_K, k=5)

    with pytest.raises(ValueError, match="必须经 L2 归一化"):
        search_forest(raw_unnorm, forest, library, cfg)


def test_p3_5_grid_mismatch_rejected():
    """P3-5: 预处理网格与森林索引网格不一致时构建森林必须报错。"""
    library, _ = _make_synth_library(20, seed=4)
    mismatched_spec = ForestSpec(summary_grid_da=0.05)  # 库是 0.02 Da

    with pytest.raises(ValueError, match="网格配置不一致"):
        build_forest_index(library, mismatched_spec)


def test_p3_6_ion_mode_parsing_unified():
    """P3-6: 离子模式解析词表统一验证。"""
    assert _parse_ion_mode("positive") == IonMode.POSITIVE
    assert _parse_ion_mode("Positive") == IonMode.POSITIVE
    assert _parse_ion_mode("pos") == IonMode.POSITIVE
    assert _parse_ion_mode("+") == IonMode.POSITIVE
    assert _parse_ion_mode("1") == IonMode.POSITIVE

    assert _parse_ion_mode("negative") == IonMode.NEGATIVE
    assert _parse_ion_mode("neg") == IonMode.NEGATIVE
    assert _parse_ion_mode("-") == IonMode.NEGATIVE
    assert _parse_ion_mode("-1") == IonMode.NEGATIVE

    assert _parse_ion_mode("unknown") == IonMode.UNKNOWN
    assert _parse_ion_mode(None) == IonMode.UNKNOWN


def test_fingerprint_content_collision_prevented():
    """问题 1: 同路径同谱数同峰数但内容不同时，指纹必须不同且阻止检索。"""
    metas_a = [
        SpectrumMeta(
            external_id=f"A{i}",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("same_path.mgf", i),
            raw_metadata={},
        )
        for i in range(6)
    ]
    mass_a = np.array([100.0, 200.0, 300.0] * 6, dtype=np.float64)
    inten_a = np.array([10.0, 20.0, 30.0] * 6, dtype=np.float64)
    pid_a = np.arange(18, dtype=np.int64)
    offs_a = np.array([0, 3, 6, 9, 12, 15, 18], dtype=np.int64)
    parsed_a = ParsedLibrary("same_path.mgf", tuple(metas_a), mass_a, inten_a, pid_a, offs_a)
    lib_a = preprocess_library(parsed_a)
    forest_a = build_forest_index(lib_a)

    metas_b = [
        SpectrumMeta(
            external_id=f"B{i}",
            precursor_mz=500.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("same_path.mgf", i),
            raw_metadata={},
        )
        for i in range(6)
    ]
    mass_b = np.array([150.0, 250.0, 350.0] * 6, dtype=np.float64)
    inten_b = np.array([10.0, 20.0, 30.0] * 6, dtype=np.float64)
    pid_b = np.arange(18, dtype=np.int64)
    offs_b = np.array([0, 3, 6, 9, 12, 15, 18], dtype=np.int64)
    parsed_b = ParsedLibrary("same_path.mgf", tuple(metas_b), mass_b, inten_b, pid_b, offs_b)
    lib_b = preprocess_library(parsed_b)

    assert lib_a.fingerprint != lib_b.fingerprint
    q = SpectrumPeaks(
        mass=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        energy=np.array([1.0, 1.0, 1.0]) / 3,
        peak_id=np.arange(3, dtype=np.int64),
        norm=1.0,
    )
    cfg = QueryConfig(mode=SearchMode.TOP_K, k=3, ion_mode=IonMode.POSITIVE)
    with pytest.raises(ValueError, match="指纹.*不匹配"):
        search_forest(q, forest_a, lib_b, cfg)


def test_zero_intensity_query_handling():
    """问题 2: 零强度非空查询经预处理后安全早退，直接传入未归一全零查询给出明确语义错误。"""
    from jetf.types import validate_query

    # 1. 经 preprocess_query 处理后安全折叠为空谱并可检索
    raw_zero = SpectrumPeaks._create_unchecked(
        mass=np.array([100.0, 200.0]),
        intensity=np.array([0.0, 0.0]),
        energy=np.array([0.0, 0.0]),
        peak_id=np.arange(2, dtype=np.int64),
        norm=0.0,
    )
    q_prep = preprocess_query(raw_zero)
    assert q_prep.mass.size == 0
    assert q_prep.norm == 0.0

    lib, forest = _make_synth_library(10, seed=12)
    cfg = QueryConfig(mode=SearchMode.TOP_K, k=3)
    res_forest = search_forest(q_prep, forest, lib, cfg)
    res_exh = search_exhaustive(q_prep, lib, cfg)
    assert res_forest.complete is True
    assert len(res_forest.hits) == len(res_exh.hits)

    # 2. 未预处理直接构造非空全零谱传入 validate_query 抛出明确语义报错
    with pytest.raises(ValueError, match="所有峰强度均为 0，无可匹配特征"):
        validate_query(raw_zero)


def test_snapshot_grid_mismatch_assertion(tmp_path):
    """问题 3: 快照跨字段网格不一致时 load_forest_snapshot 必须报错。"""
    from jetf.serialization import load_forest_snapshot, save_forest_snapshot

    lib, forest = _make_synth_library(10, seed=13)
    snap_path = tmp_path / "test_forest.npz"
    save_forest_snapshot(forest, snap_path)

    # 读取并篡改 env_grid_da
    with np.load(snap_path) as loaded:
        data = dict(loaded)
    data["env_grid_da"] = np.float64(0.05)  # spec 是 0.02
    corrupt_path = tmp_path / "corrupt_forest.npz"
    np.savez(corrupt_path, **data)

    with pytest.raises(ValueError, match="快照网格配置不一致"):
        load_forest_snapshot(corrupt_path)


def test_sah_node_cost_unification():
    """问题 4: sah_node_cost 与 sah_cost 逻辑一致性验证。"""
    from jetf.bvh_sah import sah_cost, sah_node_cost

    cells = {10, 20, 30}
    amps = {10: 1.0, 20: 2.0, 30: 3.0}
    assert sah_node_cost(cells, amps) == 18.0
    assert sah_cost([cells], [amps]) == 18.0


def test_mgf_precursor_mz_and_spectrum_id_dialects(tmp_path):
    """验证 MGF 多方言字段: PRECURSOR_MZ, PARENT_MASS 及 SPECTRUM_ID 正常解析。"""
    from jetf.mgf import parse_mgf

    mgf_content = """BEGIN IONS
SPECTRUM_ID=CCMSLIB00000001547
CHARGE=1+
IONMODE=positive
PRECURSOR_MZ=981.54
100.0 50.0
200.0 100.0
END IONS

BEGIN IONS
TITLE=SPEC_PARENT_MASS
CHARGE=2+
IONMODE=negative
PARENT_MASS=450.25
150.0 80.0
250.0 120.0
END IONS

BEGIN IONS
TITLE=SPEC_STANDARD_PEPMASS
CHARGE=1+
IONMODE=positive
PEPMASS=300.15 1000
110.0 90.0
210.0 110.0
END IONS
"""
    test_file = tmp_path / "dialects.mgf"
    test_file.write_text(mgf_content, encoding="utf-8")

    parsed = parse_mgf(test_file)
    assert parsed.n_spectra == 3
    assert len(parsed.rejected) == 0

    # 谱 0: PRECURSOR_MZ 与 SPECTRUM_ID
    s0 = parsed.spectra[0]
    assert s0.external_id == "CCMSLIB00000001547"
    assert s0.precursor_mz == pytest.approx(981.54)
    assert s0.charge == 1
    assert s0.ion_mode == IonMode.POSITIVE

    # 谱 1: PARENT_MASS
    s1 = parsed.spectra[1]
    assert s1.external_id == "SPEC_PARENT_MASS"
    assert s1.precursor_mz == pytest.approx(450.25)
    assert s1.charge == 2
    assert s1.ion_mode == IonMode.NEGATIVE

    # 谱 2: PEPMASS 带强度
    s2 = parsed.spectra[2]
    assert s2.external_id == "SPEC_STANDARD_PEPMASS"
    assert s2.precursor_mz == pytest.approx(300.15)


