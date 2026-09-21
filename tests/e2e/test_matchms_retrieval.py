"""端到端集成测试：验证 JET-Forest 检索与 matchms 全库打分基准的 100% 召回与零漏检。"""

from __future__ import annotations

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
    ForestIndex,
    ParsedLibrary,
    PreprocessedLibrary,
    QueryConfig,
    SearchMode,
    build_forest_index,
    parse_mgf,
    preprocess_library,
    search_forest,
)
from jetf.benchmarks.adapter import jetf_peaks_to_matchms

try:
    import matchms
    from matchms.similarity import CosineGreedy

    HAS_MATCHMS = True
except ImportError:
    HAS_MATCHMS = False

LIBRARY_PATH = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"

pytestmark = [
    pytest.mark.skipif(not HAS_MATCHMS, reason="缺少 matchms 依赖，跳过检索对比测试"),
    pytest.mark.skipif(
        not LIBRARY_PATH.is_file(),
        reason=f"缺少 {LIBRARY_PATH.name}：请确认 MGF 文件存在于根目录",
    ),
]


@pytest.fixture(scope="module")
def parsed() -> ParsedLibrary:
    return parse_mgf(LIBRARY_PATH)


@pytest.fixture(scope="module")
def subset_rows(parsed: ParsedLibrary) -> np.ndarray:
    # 抽 400 条谱作为轻量端到端测试库，保证测试高效运行
    counts = parsed.peak_counts()
    rng = np.random.default_rng(42)
    valid_rows = np.flatnonzero(counts >= 5)
    return np.sort(rng.choice(valid_rows, size=min(400, len(valid_rows)), replace=False))


@pytest.fixture(scope="module")
def library(parsed: ParsedLibrary, subset_rows: np.ndarray) -> PreprocessedLibrary:
    return preprocess_library(subset_parsed_library(parsed, subset_rows), CORRECTNESS_V1)


@pytest.fixture(scope="module")
def forest(library: PreprocessedLibrary) -> ForestIndex:
    return build_forest_index(library, DEFAULT_FOREST_SPEC)


def test_retrieval_open_topk_matchms_zero_false_dismissals(
    library: PreprocessedLibrary, forest: ForestIndex
):
    """验证全库开放检索 Top-K 下 JET-Forest 检出结果覆盖 matchms 真实 Top-K (Zero False Dismissals)。"""
    scorer = CosineGreedy(
        tolerance=DEFAULT_FRAGMENT_TOLERANCE_DA, mz_power=0.0, intensity_power=1.0
    )
    matchms_lib = [
        jetf_peaks_to_matchms(library.peaks.spectrum_at(r), meta=library.spectra[r])
        for r in range(library.n_spectra)
    ]

    test_rows = [0, 10, 50, 100]
    for row in test_rows:
        meta = library.spectra[row]
        q_peaks = library.peaks.spectrum_at(row)
        q_mms = jetf_peaks_to_matchms(q_peaks)

        # 1. JET-Forest 检索 Top-10
        config = QueryConfig(
            mode=SearchMode.TOP_K,
            k=10,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        )
        outcome = search_forest(q_peaks, forest, library, config)
        jetf_hit_ids = {h.external_id for h in outcome.hits}

        # 2. matchms 遍历打分真值
        gt: list[tuple[float, str]] = []
        for r, m in enumerate(library.spectra):
            if m.ion_mode != meta.ion_mode:
                continue
            res = scorer.pair(q_mms, matchms_lib[r])
            s_val = float(res["score"])
            n_m = int(res["matches"])
            if n_m >= config.min_matched_peaks:
                gt.append((s_val, m.external_id))

        gt.sort(key=lambda x: -x[0])
        top_gt = gt[:10]

        # 校验：matchms 真实 Top-10 候选在 JETF 中零漏检
        min_jetf_score = outcome.hits[-1].score if outcome.hits else 0.0
        for s_val, ext_id in top_gt:
            if ext_id not in jetf_hit_ids:
                # 必须不能高于截止分数
                assert s_val <= min_jetf_score + 1e-7, (
                    f"发现漏检！matchms 得分 {s_val} > JETF 截止分 {min_jetf_score}，候选 {ext_id}"
                )


def test_retrieval_open_top5_matchms_zero_false_dismissals(
    library: PreprocessedLibrary, forest: ForestIndex
):
    """验证全库开放检索 Top-5 下 JET-Forest 检出结果与 matchms 的一致性。"""
    scorer = CosineGreedy(
        tolerance=DEFAULT_FRAGMENT_TOLERANCE_DA, mz_power=0.0, intensity_power=1.0
    )
    matchms_lib = [
        jetf_peaks_to_matchms(library.peaks.spectrum_at(r), meta=library.spectra[r])
        for r in range(library.n_spectra)
    ]

    test_rows = [5, 25]
    for row in test_rows:
        meta = library.spectra[row]
        q_peaks = library.peaks.spectrum_at(row)
        q_mms = jetf_peaks_to_matchms(q_peaks)

        config = QueryConfig(
            mode=SearchMode.TOP_K,
            k=5,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        )
        outcome = search_forest(q_peaks, forest, library, config)
        jetf_hit_ids = {h.external_id for h in outcome.hits}

        gt: list[tuple[float, str]] = []
        for r, m in enumerate(library.spectra):
            if m.ion_mode != meta.ion_mode:
                continue
            res = scorer.pair(q_mms, matchms_lib[r])
            s_val = float(res["score"])
            n_m = int(res["matches"])
            if n_m >= config.min_matched_peaks:
                gt.append((s_val, m.external_id))

        gt.sort(key=lambda x: -x[0])
        top_gt = gt[:5]

        min_jetf_score = outcome.hits[-1].score if outcome.hits else 0.0
        for s_val, ext_id in top_gt:
            if ext_id not in jetf_hit_ids:
                assert s_val <= min_jetf_score + 1e-7, (
                    f"开放检索发现漏检！matchms 得分 {s_val} > JETF 截止分 {min_jetf_score}，候选 {ext_id}"
                )


def test_retrieval_open_threshold_matchms_zero_false_dismissals(
    library: PreprocessedLibrary, forest: ForestIndex
):
    """验证全库开放检索 Threshold 下 JET-Forest 检出结果与 matchms 的一致性。"""
    scorer = CosineGreedy(
        tolerance=DEFAULT_FRAGMENT_TOLERANCE_DA, mz_power=0.0, intensity_power=1.0
    )
    matchms_lib = [
        jetf_peaks_to_matchms(library.peaks.spectrum_at(r), meta=library.spectra[r])
        for r in range(library.n_spectra)
    ]

    test_rows = [5, 25]
    for row in test_rows:
        meta = library.spectra[row]
        q_peaks = library.peaks.spectrum_at(row)
        q_mms = jetf_peaks_to_matchms(q_peaks)

        config = QueryConfig(
            mode=SearchMode.THRESHOLD,
            threshold=0.20,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        )
        outcome = search_forest(q_peaks, forest, library, config)
        jetf_hit_ids = {h.external_id for h in outcome.hits}

        gt_ids = set()
        for r, m in enumerate(library.spectra):
            if m.ion_mode != meta.ion_mode:
                continue
            res = scorer.pair(q_mms, matchms_lib[r])
            s_val = float(res["score"])
            n_m = int(res["matches"])
            if s_val >= 0.20 and n_m >= config.min_matched_peaks:
                gt_ids.add(m.external_id)

        assert gt_ids == jetf_hit_ids
