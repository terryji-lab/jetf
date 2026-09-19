"""确定性 Greedy Cosine 评分器与单谱 Uind 精确上界。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jetf.types import (
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ENERGY_DTYPE,
    INTERNAL_ID_DTYPE,
    SpectrumPeaks,
    check_column,
)

SCORER_VERSIONED_ID = "greedy_cosine/1"
ENVELOPE_MARGIN_FACTOR = 1.0 + 1e-12


def inflate_upper_bound(val: float) -> float:
    """上界数值余量处理 (乘以 1 + 10^-12)。"""
    return float(val * ENVELOPE_MARGIN_FACTOR)


def inflate_upper_bounds(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """一维数组批量上偏。"""
    return arr * ENVELOPE_MARGIN_FACTOR


def sum_float64(arr: NDArray[np.float64]) -> float:
    """浮点求和工具。"""
    return float(np.sum(arr, dtype=np.float64))


@dataclass(frozen=True, eq=False)
class GreedyCosineResult:
    """Greedy Cosine 评分结果。"""

    score: float
    query_index: NDArray[np.int64]
    library_index: NDArray[np.int64]
    contribution: NDArray[np.float64]

    def __post_init__(self) -> None:
        check_column("query_index", self.query_index, INTERNAL_ID_DTYPE)
        check_column("library_index", self.library_index, INTERNAL_ID_DTYPE)
        check_column("contribution", self.contribution, ENERGY_DTYPE)

    @property
    def n_matched(self) -> int:
        return int(self.query_index.shape[0])


def _empty_result() -> GreedyCosineResult:
    return GreedyCosineResult(
        score=0.0,
        query_index=np.empty(0, dtype=INTERNAL_ID_DTYPE),
        library_index=np.empty(0, dtype=INTERNAL_ID_DTYPE),
        contribution=np.empty(0, dtype=ENERGY_DTYPE),
    )


def _legal_edge_indices(
    query: SpectrumPeaks, library: SpectrumPeaks, tolerance_da: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """双指针二分枚举全部合法匹配边 (|m_i - n_j| <= tolerance_da)。"""
    n_query = query.mass.shape[0]
    if n_query == 0 or library.mass.shape[0] == 0:
        return (
            np.empty(0, dtype=INTERNAL_ID_DTYPE),
            np.empty(0, dtype=INTERNAL_ID_DTYPE),
        )

    lower = np.searchsorted(library.mass, query.mass - tolerance_da, side="left")
    upper = np.searchsorted(library.mass, query.mass + tolerance_da, side="right")
    counts = upper - lower
    total = int(counts.sum())
    if total == 0:
        return (
            np.empty(0, dtype=INTERNAL_ID_DTYPE),
            np.empty(0, dtype=INTERNAL_ID_DTYPE),
        )

    starts = np.cumsum(counts, dtype=INTERNAL_ID_DTYPE) - counts
    query_index = np.repeat(np.arange(n_query, dtype=INTERNAL_ID_DTYPE), counts)
    library_index = np.repeat(lower, counts) + (
        np.arange(total, dtype=INTERNAL_ID_DTYPE) - np.repeat(starts, counts)
    )
    return query_index, library_index


def _greedy_accept(
    n_query: int,
    n_library: int,
    query_index: list[int],
    library_index: list[int],
) -> list[int]:
    """顺序贪心扫描候选边，确保一对一匹配。"""
    used_query = bytearray(n_query)
    used_library = bytearray(n_library)
    accepted: list[int] = []
    for position, (i, j) in enumerate(zip(query_index, library_index)):
        if used_query[i] or used_library[j]:
            continue
        used_query[i] = 1
        used_library[j] = 1
        accepted.append(position)
    return accepted


def score_greedy_cosine(
    query: SpectrumPeaks,
    library: SpectrumPeaks,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
) -> GreedyCosineResult:
    """确定性 Greedy Cosine 质谱相似度评分算法（唯一真值基准）。"""
    if tolerance_da <= 0.0 or not math.isfinite(tolerance_da):
        raise ValueError(f"片断容差必须为正有限数，得到 {tolerance_da}")

    query_index, library_index = _legal_edge_indices(query, library, tolerance_da)
    if query_index.shape[0] == 0:
        return _empty_result()

    weights = query.intensity[query_index] * library.intensity[library_index]
    positive = weights > 0.0
    query_index = query_index[positive]
    library_index = library_index[positive]
    weights = weights[positive]
    if query_index.shape[0] == 0:
        return _empty_result()

    # 排序规则：权重降序；并列按 query peak_id 升序，再按 library peak_id 升序
    order = np.lexsort((library.peak_id[library_index], query.peak_id[query_index], -weights))
    query_index = query_index[order]
    library_index = library_index[order]
    weights = weights[order]

    accepted = _greedy_accept(
        query.mass.shape[0], library.mass.shape[0], query_index.tolist(), library_index.tolist()
    )
    if not accepted:
        return _empty_result()

    picked = np.array(accepted, dtype=INTERNAL_ID_DTYPE)
    return GreedyCosineResult(
        score=sum_float64(weights[picked]),
        query_index=query_index[picked],
        library_index=library_index[picked],
        contribution=weights[picked],
    )


def single_spectrum_bound(
    query: SpectrumPeaks,
    library: SpectrumPeaks,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
) -> float:
    """Uind = sum_i u_i · max_(legal j) v_j：单谱放松精确上界。"""
    n_query = query.mass.shape[0]
    query_index, library_index = _legal_edge_indices(query, library, tolerance_da)
    if query_index.shape[0] == 0:
        return 0.0

    counts = np.bincount(query_index, minlength=n_query)
    starts = np.cumsum(counts, dtype=INTERNAL_ID_DTYPE) - counts
    support_peaks = np.flatnonzero(counts)
    per_peak = np.zeros(n_query, dtype=ENERGY_DTYPE)
    per_peak[support_peaks] = np.maximum.reduceat(
        library.intensity[library_index], starts[support_peaks]
    )
    return inflate_upper_bound(sum_float64(query.intensity * per_peak))
