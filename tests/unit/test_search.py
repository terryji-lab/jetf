"""Unit tests for search_forest execution engine."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _subset import stratified_subset_indices, subset_parsed_library
from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FOREST_SPEC,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
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


@pytest.fixture(scope="module")
def subset_data():
    parsed = parse_mgf(LIBRARY_PATH)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    library = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(library, DEFAULT_FOREST_SPEC)
    return library, forest


def test_search_forest_open_topk_matches_exhaustive(subset_data):
    library, forest = subset_data

    # 挑选 3 条带不同前体的真实谱作为全库开放 Top-K 测试查询
    test_rows = [0, 50, 150]
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
            snapshot_id="forest_test",
        )

        res_ex = search_exhaustive(q_peaks, library, config)
        res_forest = search_forest(q_peaks, forest, library, config)

        assert res_forest.complete is True
        assert len(res_forest.hits) == len(res_ex.hits)
        for h_f, h_e in zip(res_forest.hits, res_ex.hits):
            assert h_f.external_id == h_e.external_id
            assert h_f.spectrum_index == h_e.spectrum_index
            assert pytest.approx(h_f.score, abs=1e-12) == h_e.score
            assert h_f.n_matched == h_e.n_matched

        assert res_forest.stats.nodes_visited > 0


def test_search_forest_open_matches_exhaustive(subset_data):
    library, forest = subset_data

    # 开放检索模式 (无前体限制, threshold=0.10)
    test_rows = [0, 20]
    for row in test_rows:
        meta = library.spectra[row]
        q_peaks = library.peaks.spectrum_at(row)

        config = QueryConfig(
            mode=SearchMode.THRESHOLD,
            threshold=0.10,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            preprocess_version=library.spec.versioned_id,
            scorer_version=SCORER_VERSIONED_ID,
            snapshot_id="forest_test",
        )

        res_ex = search_exhaustive(q_peaks, library, config)
        res_forest = search_forest(q_peaks, forest, library, config)

        assert res_forest.complete is True
        assert len(res_forest.hits) == len(res_ex.hits)
        for h_f, h_e in zip(res_forest.hits, res_ex.hits):
            assert h_f.external_id == h_e.external_id
            assert h_f.spectrum_index == h_e.spectrum_index
            assert pytest.approx(h_f.score, abs=1e-12) == h_e.score
            assert h_f.n_matched == h_e.n_matched

        assert res_forest.stats.pruned_by_layer["roots_pruned"] > 0
