"""确定性 Greedy Cosine 评分器与单谱 Uind 精确上界。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import numba

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

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
    """双指针二分枚举全部合法匹配边 (|m_i - n_j| <= tolerance_da)。

    前置条件:
    - library.mass 必须按升序排列（用于 searchsorted 二分定位与连续索引切片）；
    - query 与 library 峰强度必须非负 (>= 0)。
    """
    n_query = query.mass.shape[0]
    n_lib = library.mass.shape[0]
    if n_query == 0 or n_lib == 0:
        return (
            np.empty(0, dtype=INTERNAL_ID_DTYPE),
            np.empty(0, dtype=INTERNAL_ID_DTYPE),
        )

    if n_lib > 1 and not np.all(library.mass[:-1] <= library.mass[1:]):
        raise ValueError("库谱峰质量 (mass) 必须按升序排列")

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
    score = sum_float64(weights[picked])
    # 浮点舍入容差钳制：CPU 使用 FP64，理论值 1.0 时累加可能因浮点舍入产生微小扰动 (例如 1.0000000000000002 或 0.9999999999999999)
    # 将 [1.0 - 1e-12, +inf) 规范化钳制为 1.0 (与 GPU 端的 FP32 1e-6 截断带逻辑统一对应)
    if abs(score - 1.0) <= 1e-12 or score > 1.0:
        score = 1.0
    return GreedyCosineResult(
        score=score,
        query_index=query_index[picked],
        library_index=library_index[picked],
        contribution=weights[picked],
    )


if _HAVE_NUMBA:
    @numba.njit(fastmath=False, nogil=True)
    def _single_spectrum_bound_numba(
        q_mass: NDArray[np.float64],
        q_intensity: NDArray[np.float64],
        lib_mass: NDArray[np.float64],
        lib_intensity: NDArray[np.float64],
        tolerance_da: float,
    ) -> float:
        """Numba JIT 内核：双二分定位与标量展开，零数组切片分配。"""
        n_query = q_mass.shape[0]
        n_lib = lib_mass.shape[0]
        if n_query == 0 or n_lib == 0:
            return 0.0

        if lib_mass[n_lib - 1] < q_mass[0] - tolerance_da or lib_mass[0] > q_mass[n_query - 1] + tolerance_da:
            return 0.0

        total = 0.0
        k_left = 0
        for p in range(n_query):
            c_low = q_mass[p] - tolerance_da
            c_high = q_mass[p] + tolerance_da

            # 二分查找：定位 lib_mass 中 >= c_low 的最左侧位置
            low = k_left
            high = n_lib
            while low < high:
                mid = (low + high) >> 1
                if lib_mass[mid] < c_low:
                    low = mid + 1
                else:
                    high = mid
            k_left = low
            if k_left >= n_lib:
                break

            # 二分查找：定位 lib_mass 中 > c_high 的最左侧位置
            low = k_left
            high = n_lib
            while low < high:
                mid = (low + high) >> 1
                if lib_mass[mid] <= c_high:
                    low = mid + 1
                else:
                    high = mid
            k_right = low

            if k_right > k_left:
                m_val = lib_intensity[k_left]
                for k in range(k_left + 1, k_right):
                    val = lib_intensity[k]
                    if val > m_val:
                        m_val = val
                total += q_intensity[p] * m_val

        if total > 0.0:
            return total * (1.0 + 1e-12)
        return 0.0


def _single_spectrum_bound_numpy(
    query: SpectrumPeaks,
    library: SpectrumPeaks,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
) -> float:
    """NumPy 降级实现。"""
    starts = np.searchsorted(library.mass, query.mass - tolerance_da, side="left")
    stops = np.searchsorted(library.mass, query.mass + tolerance_da, side="right")

    diff = stops - starts
    matched_idx = np.flatnonzero(diff > 0)
    if matched_idx.size == 0:
        return 0.0

    lib_inten = library.intensity
    q_inten = query.intensity

    total = 0.0
    for i in matched_idx:
        st = int(starts[i])
        sp = int(stops[i])
        if sp == st + 1:
            total += q_inten[i] * lib_inten[st]
        else:
            total += q_inten[i] * float(np.max(lib_inten[st:sp]))

    return inflate_upper_bound(total)


def single_spectrum_bound(
    query: SpectrumPeaks,
    library: SpectrumPeaks,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
) -> float:
    """Uind = sum_i u_i · max_(legal j) v_j：单谱放松精确上界。"""
    n_query = query.mass.shape[0]
    n_lib = library.mass.shape[0]
    if n_query == 0 or n_lib == 0:
        return 0.0

    if n_lib > 1 and not np.all(library.mass[:-1] <= library.mass[1:]):
        raise ValueError("库谱峰质量 (mass) 必须按升序排列")

    if _HAVE_NUMBA:
        return _single_spectrum_bound_numba(
            query.mass, query.intensity, library.mass, library.intensity, tolerance_da
        )
    return _single_spectrum_bound_numpy(query, library, tolerance_da)

