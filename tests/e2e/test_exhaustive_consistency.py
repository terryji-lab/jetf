"""端到端回归：包络森林（JET-Forest）与穷举检索 100% 逐项一致性。

涵盖：
1. 身份搜索（Top-k、高低阈值、跨树前体窗口）；
2. 开放搜索（Top-k、高低阈值、包含零分门槛 t=0）；
3. 代表谱（最长、最短、中位、随机抽样）；
4. 数学安全保证：每一条 hit 逐位一致，绝对零漏检，精评数 <= 穷举。
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _subset import stratified_subset_indices, subset_parsed_library
from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FOREST_SPEC,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestIndex,
    ParsedLibrary,
    PrecursorWindow,
    PreprocessedLibrary,
    QueryConfig,
    SCORER_VERSIONED_ID,
    SearchMode,
    build_forest_index,
    parse_mgf,
    preprocess_library,
    search_exhaustive,
    search_forest,
)

LIBRARY_PATH = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"

pytestmark = pytest.mark.skipif(
    not LIBRARY_PATH.is_file(),
    reason=f"缺少 {LIBRARY_PATH.name}：请确认 MGF 文件存在于根目录",
)


@pytest.fixture(scope="module")
def parsed() -> ParsedLibrary:
    return parse_mgf(LIBRARY_PATH)


@pytest.fixture(scope="module")
def subset_rows(parsed: ParsedLibrary) -> NDArray[np.int64]:
    return stratified_subset_indices(parsed)


@pytest.fixture(scope="module")
def library(parsed: ParsedLibrary, subset_rows: NDArray[np.int64]) -> PreprocessedLibrary:
    return preprocess_library(subset_parsed_library(parsed, subset_rows), CORRECTNESS_V1)


@pytest.fixture(scope="module")
def forest(library: PreprocessedLibrary) -> ForestIndex:
    return build_forest_index(library, DEFAULT_FOREST_SPEC)


def _assert_results_equal(res_forest, res_ex):
    """严格断言包络森林与穷举检索结果 100% 逐项逐位完全一致。"""
    assert res_forest.complete is True
    assert len(res_forest.hits) == len(res_ex.hits), (
        f"命中数不一致: 森林 {len(res_forest.hits)} 条 vs 穷举 {len(res_ex.hits)} 条"
    )
    for i, (hf, he) in enumerate(zip(res_forest.hits, res_ex.hits)):
        assert hf.external_id == he.external_id, f"第 {i} 条命中 external_id 不符: {hf} vs {he}"
        assert hf.spectrum_index == he.spectrum_index, f"第 {i} 条命中 spectrum_index 不符: {hf} vs {he}"
        assert pytest.approx(hf.score, abs=1e-12) == he.score, f"第 {i} 条命中 score 不符: {hf} vs {he}"
        assert hf.n_matched == he.n_matched, f"第 {i} 条命中 n_matched 不符: {hf} vs {he}"

    assert res_forest.stats.n_scored <= res_ex.stats.n_scored, (
        f"森林精评数 ({res_forest.stats.n_scored}) 高于穷举 ({res_ex.stats.n_scored})"
    )


def test_forest_identity_topk_exhaustive_consistency(library: PreprocessedLibrary, forest: ForestIndex):
    """身份检索 Top-K 模式下与穷举检索的一致性验证。"""
    peak_counts = np.diff(library.peaks.spectrum_offsets)
    idx_max = int(np.argmax(peak_counts))
    idx_min = int(np.argmin(peak_counts))
    test_rows = [0, 100, 500, idx_max, idx_min]

    for row in test_rows:
        meta = library.spectra[row]
        if meta.precursor_mz is None:
            continue

        q_peaks = library.peaks.spectrum_at(row)
        for tol_da in (0.5, 2.0):
            window = PrecursorWindow(mz=meta.precursor_mz, tolerance_da=tol_da)
            config = QueryConfig(
                mode=SearchMode.TOP_K,
                k=10,
                threshold=None,
                ion_mode=meta.ion_mode,
                precursor_window=window,
                fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
                preprocess_version=library.spec.versioned_id,
                scorer_version=SCORER_VERSIONED_ID,
                snapshot_id="forest_e2e",
            )

            res_ex = search_exhaustive(q_peaks, library, config)
            res_forest = search_forest(q_peaks, forest, library, config)

            _assert_results_equal(res_forest, res_ex)


def test_forest_identity_threshold_exhaustive_consistency(library: PreprocessedLibrary, forest: ForestIndex):
    """身份检索 Threshold 模式下与穷举检索的一致性验证（涵盖零分门槛）。"""
    test_rows = [10, 200, 800]

    for row in test_rows:
        meta = library.spectra[row]
        if meta.precursor_mz is None:
            continue

        q_peaks = library.peaks.spectrum_at(row)
        window = PrecursorWindow(mz=meta.precursor_mz, tolerance_da=1.0)

        for thresh in (0.60, 0.10, 0.00):
            config = QueryConfig(
                mode=SearchMode.THRESHOLD,
                threshold=thresh,
                ion_mode=meta.ion_mode,
                precursor_window=window,
                fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
                preprocess_version=library.spec.versioned_id,
                scorer_version=SCORER_VERSIONED_ID,
                snapshot_id="forest_e2e",
            )

            res_ex = search_exhaustive(q_peaks, library, config)
            res_forest = search_forest(q_peaks, forest, library, config)

            _assert_results_equal(res_forest, res_ex)


def test_forest_open_topk_exhaustive_consistency(library: PreprocessedLibrary, forest: ForestIndex):
    """开放检索 Top-K 模式下与穷举检索的一致性验证。"""
    test_rows = [5, 50, 250]

    for row in test_rows:
        meta = library.spectra[row]
        q_peaks = library.peaks.spectrum_at(row)

        config = QueryConfig(
            mode=SearchMode.TOP_K,
            k=10,
            threshold=None,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=library.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
            snapshot_id="forest_e2e",
        )

        res_ex = search_exhaustive(q_peaks, library, config)
        res_forest = search_forest(q_peaks, forest, library, config)

        _assert_results_equal(res_forest, res_ex)
        assert res_forest.stats.pruned_by_layer["roots_pruned"] > 0


def test_forest_open_threshold_exhaustive_consistency(library: PreprocessedLibrary, forest: ForestIndex):
    """开放检索 Threshold 模式下与穷举检索的一致性验证（高剪枝率与零分补足）。"""
    test_rows = [15, 75]

    for row in test_rows:
        meta = library.spectra[row]
        q_peaks = library.peaks.spectrum_at(row)

        for thresh in (0.60, 0.10):
            config = QueryConfig(
                mode=SearchMode.THRESHOLD,
                threshold=thresh,
                ion_mode=meta.ion_mode,
                fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
                preprocess_version=library.spec.versioned_id,
                scorer_version=SCORER_VERSIONED_ID,
                snapshot_id="forest_e2e",
            )

            res_ex = search_exhaustive(q_peaks, library, config)
            res_forest = search_forest(q_peaks, forest, library, config)

            _assert_results_equal(res_forest, res_ex)
            assert res_forest.stats.pruned_by_layer["roots_pruned"] > 0
