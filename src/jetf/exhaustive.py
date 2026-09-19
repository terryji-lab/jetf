"""确定性穷举检索器：作为数学真值基准验证检索算法的 100% 正确性。"""

from __future__ import annotations

import time
import numpy as np

from jetf.preprocessing import PreprocessedLibrary
from jetf.query import QueryConfig, SearchMode, is_eligible
from jetf.results import (
    EvalTimers,
    ResultSet,
    SearchHit,
    SearchOutcome,
    SearchStats,
    ZeroScoreCandidates,
    elapsed_ms,
    hit_ranking_key,
    needs_zero_supplement,
    search_versions,
    supplement_zero_score,
)
from jetf.scoring import score_greedy_cosine
from jetf.types import SpectrumPeaks


def search_exhaustive(
    query: SpectrumPeaks,
    library: PreprocessedLibrary,
    config: QueryConfig,
) -> SearchOutcome:
    """全库逐条精评穷举检索（不设任何上界与剪枝）。"""
    timers = EvalTimers()
    results = ResultSet(config)
    scored_mask = np.zeros(library.n_spectra, dtype=np.bool_)
    n_scored = 0

    frag_tau = config.fragment_tolerance_da
    min_matched = config.min_matched_peaks

    # 1. 遍历全部库谱，按元数据过滤资格
    for row, meta in enumerate(library.spectra):
        if not is_eligible(config, meta):
            continue

        member_peaks = library.peaks.spectrum_at(row)

        t_start = time.perf_counter()
        score_res = score_greedy_cosine(query, member_peaks, frag_tau)
        timers.exact_eval_ms += elapsed_ms(t_start)

        n_scored += 1
        scored_mask[row] = True

        if score_res.n_matched >= min_matched:
            results.update(
                SearchHit(
                    score=score_res.score,
                    external_id=meta.external_id,
                    spectrum_index=row,
                    n_matched=score_res.n_matched,
                )
            )

    # 2. 零分补足
    if needs_zero_supplement(results, config):
        candidates = ZeroScoreCandidates(
            member=np.arange(library.n_spectra, dtype=np.int64),
            ion_mode=np.array(
                [
                    0 if meta.ion_mode.value == "positive" else 1 if meta.ion_mode.value == "negative" else 2
                    for meta in library.spectra
                ],
                dtype=np.int8,
            ),
            precursor_mz=np.array(
                [
                    meta.precursor_mz if meta.precursor_mz is not None else np.nan
                    for meta in library.spectra
                ],
                dtype=np.float64,
            ),
        )
        supplement_zero_score(results, library, config, candidates, scored_mask)

    hits = results.finish()

    stats = SearchStats(
        nodes_visited=0,
        pruned_by_layer={},
        n_scored=n_scored,
        bound_eval_time_ms=0.0,
        exact_eval_time_ms=timers.exact_eval_ms,
        seed_count=0,
        skipped_block_intervals=(),
    )

    versions = search_versions(library, config)

    return SearchOutcome(
        mode=config.mode,
        hits=hits,
        complete=True,
        stats=stats,
        versions=versions,
    )
