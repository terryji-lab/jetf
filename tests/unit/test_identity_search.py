"""Unit tests for targeted identity search with precursor window binary search."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _subset import stratified_subset_indices, subset_parsed_library

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FOREST_SPEC,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestSpec,
    IonMode,
    ParsedLibrary,
    PrecursorWindow,
    QueryConfig,
    SCORER_VERSIONED_ID,
    SearchMode,
    SourceRef,
    SpectrumMeta,
    build_forest_index,
    is_eligible,
    parse_mgf,
    preprocess_library,
    preprocess_query,
    search_exhaustive,
    search_forest,
)
from jetf.types import SpectrumPeaks

LIBRARY_PATH = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"


# =========================================================================
# 1. PrecursorWindow Dataclass 属性与校验测试
# =========================================================================

def test_precursor_window_basics():
    w = PrecursorWindow(mz=300.5, tolerance_da=0.5)
    assert w.mz == 300.5
    assert w.tolerance_da == 0.5
    assert w.min_mz == 300.0
    assert w.max_mz == 310.0 or pytest.approx(w.max_mz) == 311.0 or pytest.approx(w.max_mz) == 301.0
    assert pytest.approx(w.max_mz) == 301.0

    # contains
    assert w.contains(300.5) is True
    assert w.contains(300.0) is True
    assert w.contains(301.0) is True
    assert w.contains(299.99) is False
    assert w.contains(301.01) is False
    assert w.contains(None) is False
    assert w.contains(float("nan")) is False


def test_precursor_window_validation():
    with pytest.raises(ValueError, match="前体 mz 必须为正有限数"):
        PrecursorWindow(mz=0.0, tolerance_da=0.5)
    with pytest.raises(ValueError, match="前体 mz 必须为正有限数"):
        PrecursorWindow(mz=-10.0, tolerance_da=0.5)
    with pytest.raises(ValueError, match="前体 mz 必须为正有限数"):
        PrecursorWindow(mz=float("nan"), tolerance_da=0.5)
    with pytest.raises(ValueError, match="前体容差 tolerance_da 必须为正有限数"):
        PrecursorWindow(mz=100.0, tolerance_da=0.0)
    with pytest.raises(ValueError, match="前体容差 tolerance_da 必须为正有限数"):
        PrecursorWindow(mz=100.0, tolerance_da=-0.1)


# =========================================================================
# 2. QueryConfig 与 is_eligible 资格判定测试
# =========================================================================

def test_query_config_and_is_eligible():
    w = PrecursorWindow(mz=200.0, tolerance_da=1.0)
    cfg = QueryConfig(mode=SearchMode.TOP_K, k=5, ion_mode=IonMode.POSITIVE, precursor_window=w)
    assert cfg.precursor_window == w

    with pytest.raises(TypeError, match="precursor_window 必须为 PrecursorWindow 实例"):
        QueryConfig(mode=SearchMode.TOP_K, k=5, precursor_window="invalid")  # type: ignore

    meta_match = SpectrumMeta(
        external_id="S1", precursor_mz=200.5, charge=1, ion_mode=IonMode.POSITIVE, source=SourceRef("x", 0)
    )
    meta_mismatch = SpectrumMeta(
        external_id="S2", precursor_mz=205.0, charge=1, ion_mode=IonMode.POSITIVE, source=SourceRef("x", 1)
    )
    meta_nan = SpectrumMeta(
        external_id="S3", precursor_mz=None, charge=1, ion_mode=IonMode.POSITIVE, source=SourceRef("x", 2)
    )

    assert is_eligible(cfg, meta_match) is True
    assert is_eligible(cfg, meta_mismatch) is False
    assert is_eligible(cfg, meta_nan) is False

    # 若 precursor_window 为 None，则不限制前体
    cfg_open = QueryConfig(mode=SearchMode.TOP_K, k=5, ion_mode=IonMode.POSITIVE)
    assert is_eligible(cfg_open, meta_match) is True
    assert is_eligible(cfg_open, meta_mismatch) is True
    assert is_eligible(cfg_open, meta_nan) is True


# =========================================================================
# 3. 构造受控多树子库（包含 NaN 前体、零能量谱、多区间树）进行 Bit-exact 验证
# =========================================================================

def _create_controlled_library() -> ParsedLibrary:
    """构造包含 40 条谱的合成库，前体分布在不同区间，并包含 NaN 前体与零能量谱。"""
    rng = np.random.default_rng(20260921)
    spectra: list[SpectrumMeta] = []
    masses: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    pids: list[np.ndarray] = []
    offsets = [0]

    # 区间设定：
    # 0..9: 前体 ~100.0 Da (Positive)
    # 10..19: 前体 ~200.0 Da (Positive)
    # 20..29: 前体 ~300.0 Da (Positive)
    # 30..34: 前体 ~400.0 Da (Positive)
    # 35..37: 前体 None (NaN) (Positive)
    # 38: 零能量谱 (0 峰)，前体 200.0 Da
    # 39: 零能量谱 (0 峰)，前体 None (NaN)
    precursor_targets = (
        [100.0 + rng.uniform(-0.1, 0.1) for _ in range(10)]
        + [200.0 + rng.uniform(-0.1, 0.1) for _ in range(10)]
        + [300.0 + rng.uniform(-0.1, 0.1) for _ in range(10)]
        + [400.0 + rng.uniform(-0.1, 0.1) for _ in range(5)]
        + [None, None, None]
        + [200.0, None]
    )

    for i, prec in enumerate(precursor_targets):
        ext_id = f"SYNTH_{i:04d}"
        if i in (38, 39):
            # 零能量谱
            n_p = 0
            m = np.empty(0, dtype=np.float64)
            it = np.empty(0, dtype=np.float64)
            pid = np.empty(0, dtype=np.int64)
        else:
            n_p = 15
            m = np.sort(rng.uniform(50.0, 500.0, size=n_p))
            it = rng.uniform(10.0, 100.0, size=n_p)
            pid = np.arange(n_p, dtype=np.int64)

        masses.append(m)
        intensities.append(it)
        pids.append(pid)
        offsets.append(offsets[-1] + n_p)

        meta = SpectrumMeta(
            external_id=ext_id,
            precursor_mz=prec,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("synth.mgf", i),
            raw_metadata={"TITLE": ext_id},
        )
        spectra.append(meta)

    return ParsedLibrary(
        source_path="synth.mgf",
        spectra=tuple(spectra),
        mass=np.concatenate(masses),
        intensity=np.concatenate(intensities),
        peak_id=np.concatenate(pids),
        spectrum_offsets=np.array(offsets, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def synth_library_and_forest():
    parsed = _create_controlled_library()
    library = preprocess_library(parsed, CORRECTNESS_V1)
    # 使用较小 tree_capacity=5，生成多棵小树以充分测试树剪枝
    spec = ForestSpec(tree_capacity=5, leaf_capacity=2)
    forest = build_forest_index(library, spec)
    return library, forest


def test_controlled_precursor_window_tree_pruning(synth_library_and_forest):
    library, forest = synth_library_and_forest
    # 查询第 2 条谱（前体 ~100.0 Da）
    q_peaks = library.peaks.spectrum_at(2)
    prec = library.spectra[2].precursor_mz

    window = PrecursorWindow(mz=prec, tolerance_da=0.5)
    config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=10,
        ion_mode=IonMode.POSITIVE,
        precursor_window=window,
    )

    res_forest = search_forest(q_peaks, forest, library, config)
    res_ex = search_exhaustive(q_peaks, library, config)

    # 1. 验证 100% Bit-exact 一致性
    assert len(res_forest.hits) == len(res_ex.hits)
    assert len(res_forest.hits) > 0
    for h_f, h_e in zip(res_forest.hits, res_ex.hits):
        assert h_f.external_id == h_e.external_id
        assert h_f.spectrum_index == h_e.spectrum_index
        assert pytest.approx(h_f.score, abs=1e-12) == h_e.score
        assert h_f.n_matched == h_e.n_matched

    # 2. 验证所有命中记录的前体都在窗口内
    for h in res_forest.hits:
        h_prec = library.spectra[h.spectrum_index].precursor_mz
        assert window.contains(h_prec)

    # 3. 验证树剪枝：比较开放式检索与靶向检索的访问节点数与候选树
    config_open = QueryConfig(
        mode=SearchMode.TOP_K,
        k=10,
        ion_mode=IonMode.POSITIVE,
    )
    res_open = search_forest(q_peaks, forest, library, config_open)
    assert res_forest.stats.nodes_visited < res_open.stats.nodes_visited
    assert res_forest.stats.nodes_visited < forest.nodes.n_nodes


def test_controlled_threshold_search_with_precursor_window(synth_library_and_forest):
    library, forest = synth_library_and_forest
    # 查询第 12 条谱（前体 ~200.0 Da）
    q_peaks = library.peaks.spectrum_at(12)
    prec = library.spectra[12].precursor_mz

    window = PrecursorWindow(mz=prec, tolerance_da=0.3)
    config = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.01,
        ion_mode=IonMode.POSITIVE,
        precursor_window=window,
    )

    res_forest = search_forest(q_peaks, forest, library, config)
    res_ex = search_exhaustive(q_peaks, library, config)

    assert len(res_forest.hits) == len(res_ex.hits)
    for h_f, h_e in zip(res_forest.hits, res_ex.hits):
        assert h_f.external_id == h_e.external_id
        assert h_f.spectrum_index == h_e.spectrum_index
        assert pytest.approx(h_f.score, abs=1e-12) == h_e.score
        assert h_f.n_matched == h_e.n_matched


def test_zero_energy_supplement_with_precursor_window(synth_library_and_forest):
    """测试 min_matched_peaks=0 时，零分补足受前体窗口限制，不引入窗口外记录。"""
    library, forest = synth_library_and_forest
    q_peaks = library.peaks.spectrum_at(0)

    # 窗口设置为 200.0 Da，合成库中谱 38 是前体为 200.0 的零能量谱，谱 39 是前体为 None 的零能量谱
    window = PrecursorWindow(mz=200.0, tolerance_da=0.5)
    config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=20,
        min_matched_peaks=0,
        threshold=None,
        ion_mode=IonMode.POSITIVE,
        precursor_window=window,
    )

    res_forest = search_forest(q_peaks, forest, library, config)
    res_ex = search_exhaustive(q_peaks, library, config)

    assert len(res_forest.hits) == len(res_ex.hits)
    for h_f, h_e in zip(res_forest.hits, res_ex.hits):
        assert h_f.external_id == h_e.external_id
        assert h_f.spectrum_index == h_e.spectrum_index
        assert pytest.approx(h_f.score, abs=1e-12) == h_e.score

    # 验证谱 38（前体 200.0）可补入，谱 39（前体 None）不可补入
    hit_indices = {h.spectrum_index for h in res_forest.hits}
    assert 39 not in hit_indices
    # 所有命中谱前体必须在窗口内
    for h in res_forest.hits:
        h_prec = library.spectra[h.spectrum_index].precursor_mz
        assert window.contains(h_prec)


def test_precursor_window_disjoint_returns_empty(synth_library_and_forest):
    """测试窗口无任何库谱落在其中时，检索正常返回 0 命中，且根节点完全不展开。"""
    library, forest = synth_library_and_forest
    q_peaks = library.peaks.spectrum_at(0)

    window = PrecursorWindow(mz=888.8, tolerance_da=0.5)
    config = QueryConfig(
        mode=SearchMode.TOP_K,
        k=10,
        ion_mode=IonMode.POSITIVE,
        precursor_window=window,
    )

    res_forest = search_forest(q_peaks, forest, library, config)
    res_ex = search_exhaustive(q_peaks, library, config)

    assert len(res_forest.hits) == 0
    assert len(res_ex.hits) == 0
    assert res_forest.stats.nodes_visited == 0


# =========================================================================
# 4. 真实 GNPS 子集上的靶向检索 Bit-exact 等价性与加速比验证
# =========================================================================

@pytest.fixture(scope="module")
def gnps_data():
    if not LIBRARY_PATH.is_file():
        pytest.skip(f"缺少参考质谱库文件: {LIBRARY_PATH}")
    parsed = parse_mgf(LIBRARY_PATH)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    library = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(library, DEFAULT_FOREST_SPEC)
    return library, forest


def test_gnps_identity_search_bit_exact(gnps_data):
    library, forest = gnps_data

    # 选取 5 条不同质量前体的真实谱
    test_rows = [10, 50, 100, 200, 300]
    for row in test_rows:
        meta = library.spectra[row]
        if meta.precursor_mz is None or not math.isfinite(meta.precursor_mz):
            continue

        q_peaks = library.peaks.spectrum_at(row)

        # 严格窄窗口 (0.02 Da)
        window = PrecursorWindow(mz=meta.precursor_mz, tolerance_da=0.02)
        config = QueryConfig(
            mode=SearchMode.TOP_K,
            k=5,
            ion_mode=meta.ion_mode,
            precursor_window=window,
        )

        res_forest = search_forest(q_peaks, forest, library, config)
        res_ex = search_exhaustive(q_peaks, library, config)

        assert res_forest.complete is True
        assert len(res_forest.hits) == len(res_ex.hits)
        for h_f, h_e in zip(res_forest.hits, res_ex.hits):
            assert h_f.external_id == h_e.external_id
            assert h_f.spectrum_index == h_e.spectrum_index
            assert pytest.approx(h_f.score, abs=1e-12) == h_e.score
            assert h_f.n_matched == h_e.n_matched

        # 验证根节点剪枝：因为有前体二分，访问的根节点远小于分区全量树
        # 对应分区的总树数
        partition = next(p for p in forest.partitions if p.ion_mode == meta.ion_mode)
        assert res_forest.stats.nodes_visited <= partition.n_trees
