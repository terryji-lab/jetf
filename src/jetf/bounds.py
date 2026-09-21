"""查询包络上下文与节点峰上界 (Peak Bound) 计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import numba

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

if TYPE_CHECKING:
    from jetf.structure import ForestIndex

from jetf.scoring import (
    inflate_upper_bound,
    sum_float64,
)
from jetf.types import (
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ENERGY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    SpectrumPeaks,
    check_column,
)


@dataclass(frozen=True, eq=False)
class NodeEnvelope:
    """单个节点的包络摘要切片。"""

    grid_da: float
    cell_index: NDArray[np.int64]
    max_peak_amplitude: NDArray[np.float64]

    def __post_init__(self) -> None:
        check_column("cell_index", self.cell_index, INTERNAL_ID_DTYPE)
        check_column("max_peak_amplitude", self.max_peak_amplitude, ENERGY_DTYPE)
        if not (self.cell_index.shape == self.max_peak_amplitude.shape):
            raise ValueError("NodeEnvelope 的 cell_index, max_peak_amplitude 必须等长")

    @classmethod
    def _create_unchecked(
        cls,
        grid_da: float,
        cell_index: NDArray[np.int64],
        max_peak_amplitude: NDArray[np.float64],
    ) -> NodeEnvelope:
        obj = object.__new__(cls)
        object.__setattr__(obj, "grid_da", grid_da)
        object.__setattr__(obj, "cell_index", cell_index)
        object.__setattr__(obj, "max_peak_amplitude", max_peak_amplitude)
        return obj

    @property
    def n_cells(self) -> int:
        return int(self.cell_index.shape[0])


def window_cells(
    mass: NDArray[np.float64], tolerance_da: float, grid_da: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """逐查询峰枚举兼容 cell 的保守超集 [cell_lower, cell_upper]。"""
    lower = np.nextafter(mass - tolerance_da + 1e-12, -np.inf) / grid_da
    upper = np.nextafter(mass + tolerance_da + 1e-12, np.inf) / grid_da
    return (
        np.floor(lower).astype(INTERNAL_ID_DTYPE),
        np.floor(upper).astype(INTERNAL_ID_DTYPE),
    )


def _support_spans(
    support: NDArray[np.int64],
    cell_lower: NDArray[np.int64],
    cell_upper: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """每个查询峰在节点支持集上的命中数与起点。"""
    starts = np.searchsorted(support, cell_lower, side="left").astype(INTERNAL_ID_DTYPE)
    stops = np.searchsorted(support, cell_upper, side="right").astype(INTERNAL_ID_DTYPE)
    return stops - starts, starts


def _expand_spans(starts: NDArray[np.int64], counts: NDArray[np.int64]) -> NDArray[np.int64]:
    """把逐峰的命中位置区间展开平铺。"""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=INTERNAL_ID_DTYPE)
    heads = np.cumsum(counts, dtype=INTERNAL_ID_DTYPE) - counts
    return np.repeat(starts, counts) + (
        np.arange(total, dtype=INTERNAL_ID_DTYPE) - np.repeat(heads, counts)
    )


@dataclass(frozen=True, eq=False)
class QueryContext:
    """一个节点面对单条查询谱时的上下文结构。"""

    grid_da: float
    tolerance_da: float
    mass: NDArray[np.float64]
    intensity: NDArray[np.float64]
    cell_lower: NDArray[np.int64]
    cell_upper: NDArray[np.int64]
    peak_offsets: NDArray[np.int64]
    compatible_positions: NDArray[np.int64]
    n_support_cells: int

    @property
    def n_peaks(self) -> int:
        return int(self.mass.shape[0])


def build_query_context(
    query: SpectrumPeaks,
    envelope: NodeEnvelope,
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
    cell_lower: NDArray[np.int64] | None = None,
    cell_upper: NDArray[np.int64] | None = None,
) -> QueryContext:
    """构建查询谱针对该节点包络的上下文。"""
    support = envelope.cell_index
    if cell_lower is None or cell_upper is None:
        cell_lower, cell_upper = window_cells(query.mass, tolerance_da, envelope.grid_da)
    counts, starts = _support_spans(support, cell_lower, cell_upper)

    peak_offsets = np.zeros(int(query.mass.shape[0]) + 1, dtype=INTERNAL_ID_DTYPE)
    np.cumsum(counts, out=peak_offsets[1:])

    compat_positions = _expand_spans(starts, counts)

    return QueryContext(
        grid_da=envelope.grid_da,
        tolerance_da=tolerance_da,
        mass=query.mass,
        intensity=query.intensity,
        cell_lower=cell_lower,
        cell_upper=cell_upper,
        peak_offsets=peak_offsets,
        compatible_positions=compat_positions,
        n_support_cells=int(support.shape[0]),
    )


def peak_bound(context: QueryContext, envelope: NodeEnvelope) -> float:
    """U_peak = sum_i u_i * max_(h in E_B(i)) m_h(B)：节点峰上界。"""
    if context.n_support_cells != envelope.n_cells:
        raise ValueError(
            f"上下文与包络节点不匹配: 上下文声明 {context.n_support_cells} cells, 包络实际 {envelope.n_cells} cells"
        )

    support_peaks = np.flatnonzero(np.diff(context.peak_offsets))
    if support_peaks.shape[0] == 0:
        return 0.0

    amplitudes = envelope.max_peak_amplitude[context.compatible_positions]
    per_peak = np.zeros(context.n_peaks, dtype=ENERGY_DTYPE)
    per_peak[support_peaks] = np.maximum.reduceat(
        amplitudes, context.peak_offsets[support_peaks]
    )
    return inflate_upper_bound(sum_float64(context.intensity * per_peak))


if _HAVE_NUMBA:
    @numba.njit(parallel=True, fastmath=False, nogil=True)
    def _batch_root_bounds_numba(
        tree_ids: NDArray[np.int64],
        root_node_ids: NDArray[np.int64],
        env_offsets: NDArray[np.int64],
        cell_index: NDArray[np.int64],
        max_peak_amplitude: NDArray[np.float64],
        q_intensity: NDArray[np.float64],
        cell_lower: NDArray[np.int64],
        cell_upper: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        n_trees = tree_ids.shape[0]
        n_peaks = q_intensity.shape[0]
        out_bounds = np.zeros(n_trees, dtype=np.float64)

        for i in numba.prange(n_trees):
            t_id = tree_ids[i]
            root_id = root_node_ids[t_id]
            start = env_offsets[root_id]
            end = env_offsets[root_id + 1]
            if start >= end:
                continue

            # 1. 快速包络盒相交过滤：若树最大 cell < 查询最小 cell 或树最小 cell > 查询最大 cell，直接短路
            if cell_index[end - 1] < cell_lower[0] or cell_index[start] > cell_upper[n_peaks - 1]:
                continue

            s = 0.0
            k_left = start
            for p in range(n_peaks):
                c_low = cell_lower[p]
                c_high = cell_upper[p]

                # 单调二分：query 峰质量升序，故 >= c_low 的起点单调不减
                low = k_left
                high = end
                while low < high:
                    mid = (low + high) >> 1
                    if cell_index[mid] < c_low:
                        low = mid + 1
                    else:
                        high = mid
                k_left = low
                if k_left >= end:
                    break

                # 二分查找：定位 cell_index[k_left:end] 中 > c_high 的最左侧位置
                low = k_left
                high = end
                while low < high:
                    mid = (low + high) >> 1
                    if cell_index[mid] <= c_high:
                        low = mid + 1
                    else:
                        high = mid
                k_right = low

                if k_right > k_left:
                    m_val = max_peak_amplitude[k_left]
                    for k in range(k_left + 1, k_right):
                        val = max_peak_amplitude[k]
                        if val > m_val:
                            m_val = val
                    s += q_intensity[p] * m_val

            if s > 0.0:
                out_bounds[i] = s * (1.0 + 1e-12)
            else:
                out_bounds[i] = 0.0

        return out_bounds

    @numba.njit(fastmath=False, nogil=True)
    def _batch_node_bounds_numba(
        node_ids: NDArray[np.int64],
        env_offsets: NDArray[np.int64],
        cell_index: NDArray[np.int64],
        max_peak_amplitude: NDArray[np.float64],
        q_intensity: NDArray[np.float64],
        cell_lower: NDArray[np.int64],
        cell_upper: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """批量计算一组节点（如叶节点）的上界，单线程纯寄存器累加无内存动态分配。"""
        n_nodes = node_ids.shape[0]
        n_peaks = q_intensity.shape[0]
        out_bounds = np.zeros(n_nodes, dtype=np.float64)

        for i in range(n_nodes):
            nid = node_ids[i]
            start = env_offsets[nid]
            end = env_offsets[nid + 1]
            if start >= end:
                continue

            if cell_index[end - 1] < cell_lower[0] or cell_index[start] > cell_upper[n_peaks - 1]:
                continue

            s = 0.0
            k_left = start
            for p in range(n_peaks):
                c_low = cell_lower[p]
                c_high = cell_upper[p]

                low = k_left
                high = end
                while low < high:
                    mid = (low + high) >> 1
                    if cell_index[mid] < c_low:
                        low = mid + 1
                    else:
                        high = mid
                k_left = low
                if k_left >= end:
                    break

                low = k_left
                high = end
                while low < high:
                    mid = (low + high) >> 1
                    if cell_index[mid] <= c_high:
                        low = mid + 1
                    else:
                        high = mid
                k_right = low

                if k_right > k_left:
                    m_val = max_peak_amplitude[k_left]
                    for k in range(k_left + 1, k_right):
                        val = max_peak_amplitude[k]
                        if val > m_val:
                            m_val = val
                    s += q_intensity[p] * m_val

            if s > 0.0:
                out_bounds[i] = s * (1.0 + 1e-12)
            else:
                out_bounds[i] = 0.0

        return out_bounds

    @numba.njit(fastmath=False, nogil=True)
    def _single_node_bound_numba(
        nid: int,
        env_offsets: NDArray[np.int64],
        cell_index: NDArray[np.int64],
        max_peak_amplitude: NDArray[np.float64],
        q_intensity: NDArray[np.float64],
        cell_lower: NDArray[np.int64],
        cell_upper: NDArray[np.int64],
    ) -> float:
        """单节点/叶节点上界标量 JIT 求值内核。"""
        start = env_offsets[nid]
        end = env_offsets[nid + 1]
        if start >= end:
            return 0.0

        n_peaks = q_intensity.shape[0]
        if cell_index[end - 1] < cell_lower[0] or cell_index[start] > cell_upper[n_peaks - 1]:
            return 0.0

        s = 0.0
        k_left = start
        for p in range(n_peaks):
            c_low = cell_lower[p]
            c_high = cell_upper[p]

            low = k_left
            high = end
            while low < high:
                mid = (low + high) >> 1
                if cell_index[mid] < c_low:
                    low = mid + 1
                else:
                    high = mid
            k_left = low
            if k_left >= end:
                break

            low = k_left
            high = end
            while low < high:
                mid = (low + high) >> 1
                if cell_index[mid] <= c_high:
                    low = mid + 1
                else:
                    high = mid
            k_right = low

            if k_right > k_left:
                m_val = max_peak_amplitude[k_left]
                for k in range(k_left + 1, k_right):
                    val = max_peak_amplitude[k]
                    if val > m_val:
                        m_val = val
                s += q_intensity[p] * m_val

        if s > 0.0:
            return s * (1.0 + 1e-12)
        return 0.0

    _leaf_bound_numba = _single_node_bound_numba


def _batch_root_bounds_numpy(
    tree_ids: NDArray[np.int64],
    root_node_ids: NDArray[np.int64],
    env_offsets: NDArray[np.int64],
    cell_index: NDArray[np.int64],
    max_peak_amplitude: NDArray[np.float64],
    q_intensity: NDArray[np.float64],
    cell_lower: NDArray[np.int64],
    cell_upper: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_peaks = int(q_intensity.shape[0])
    n_trees = int(tree_ids.shape[0])
    out_bounds = np.zeros(n_trees, dtype=ENERGY_DTYPE)

    per_peak = np.zeros(n_peaks, dtype=ENERGY_DTYPE)
    peak_offsets = np.zeros(n_peaks + 1, dtype=INTERNAL_ID_DTYPE)

    for i in range(n_trees):
        t_id = int(tree_ids[i])
        root_id = int(root_node_ids[t_id])
        start = int(env_offsets[root_id])
        end = int(env_offsets[root_id + 1])
        if start >= end:
            continue

        support = cell_index[start:end]
        amps = max_peak_amplitude[start:end]

        starts = np.searchsorted(support, cell_lower, side="left").astype(INTERNAL_ID_DTYPE)
        stops = np.searchsorted(support, cell_upper, side="right").astype(INTERNAL_ID_DTYPE)
        counts = stops - starts

        total = int(counts.sum())
        if total == 0:
            continue

        np.cumsum(counts, out=peak_offsets[1:])
        support_peaks = np.flatnonzero(counts)

        heads = peak_offsets[:-1]
        compat_positions = np.repeat(starts, counts) + (
            np.arange(total, dtype=INTERNAL_ID_DTYPE) - np.repeat(heads, counts)
        )

        per_peak.fill(0.0)
        per_peak[support_peaks] = np.maximum.reduceat(
            amps[compat_positions], peak_offsets[support_peaks]
        )
        s = sum_float64(q_intensity * per_peak)
        if s > 0.0:
            out_bounds[i] = inflate_upper_bound(s)
        else:
            out_bounds[i] = 0.0

    return out_bounds


def _batch_node_bounds_numpy(
    node_ids: NDArray[np.int64],
    env_offsets: NDArray[np.int64],
    cell_index: NDArray[np.int64],
    max_peak_amplitude: NDArray[np.float64],
    q_intensity: NDArray[np.float64],
    cell_lower: NDArray[np.int64],
    cell_upper: NDArray[np.int64],
) -> NDArray[np.float64]:
    n_peaks = int(q_intensity.shape[0])
    n_nodes = int(node_ids.shape[0])
    out_bounds = np.zeros(n_nodes, dtype=ENERGY_DTYPE)

    per_peak = np.zeros(n_peaks, dtype=ENERGY_DTYPE)
    peak_offsets = np.zeros(n_peaks + 1, dtype=INTERNAL_ID_DTYPE)

    for i in range(n_nodes):
        nid = int(node_ids[i])
        start = int(env_offsets[nid])
        end = int(env_offsets[nid + 1])
        if start >= end:
            continue

        support = cell_index[start:end]
        amps = max_peak_amplitude[start:end]

        starts = np.searchsorted(support, cell_lower, side="left").astype(INTERNAL_ID_DTYPE)
        stops = np.searchsorted(support, cell_upper, side="right").astype(INTERNAL_ID_DTYPE)
        counts = stops - starts

        total = int(counts.sum())
        if total == 0:
            continue

        np.cumsum(counts, out=peak_offsets[1:])
        support_peaks = np.flatnonzero(counts)

        heads = peak_offsets[:-1]
        compat_positions = np.repeat(starts, counts) + (
            np.arange(total, dtype=INTERNAL_ID_DTYPE) - np.repeat(heads, counts)
        )

        per_peak.fill(0.0)
        per_peak[support_peaks] = np.maximum.reduceat(
            amps[compat_positions], peak_offsets[support_peaks]
        )
        s = sum_float64(q_intensity * per_peak)
        if s > 0.0:
            out_bounds[i] = inflate_upper_bound(s)
        else:
            out_bounds[i] = 0.0

    return out_bounds


def batch_root_bounds(
    query: SpectrumPeaks,
    forest: ForestIndex,
    tree_ids: Sequence[int] | NDArray[np.int64],
    cell_lower: NDArray[np.int64],
    cell_upper: NDArray[np.int64],
) -> NDArray[np.float64]:
    """批量计算一组候选树根节点的峰上界 U_peak(Root)。

    若环境存在 Numba，采用多线程 JIT 并行二分内核进行树级别并行求值；
    无 Numba 环境时自动降级至 NumPy 向量化复用内核。
    保证与单个 peak_bound(build_query_context(...)) 计算结果数学完全等价，
    并在浮点误差范围内严格遵守零漏检（Zero False Dismissals）性质。
    """
    n_peaks = int(query.mass.shape[0])
    n_trees = len(tree_ids)
    if n_peaks == 0 or n_trees == 0:
        return np.zeros(n_trees, dtype=ENERGY_DTYPE)

    # 校验输入向量维度一致性
    if not (cell_lower.shape[0] == cell_upper.shape[0] == query.intensity.shape[0] == n_peaks):
        raise ValueError(
            f"查询向量长度不匹配: mass={n_peaks}, intensity={query.intensity.shape[0]}, "
            f"cell_lower={cell_lower.shape[0]}, cell_upper={cell_upper.shape[0]}"
        )

    tree_ids_arr = np.ascontiguousarray(tree_ids, dtype=INTERNAL_ID_DTYPE)
    # 防御非法树索引（避免 Numba 访问越界引发未定义行为）
    if tree_ids_arr.size > 0:
        if bool(np.any(tree_ids_arr < 0) or np.any(tree_ids_arr >= forest.n_trees)):
            raise IndexError(
                f"tree_ids 存在超出合法范围 [0, {forest.n_trees}) 的非法索引"
            )

    cell_lower_arr = np.ascontiguousarray(cell_lower, dtype=INTERNAL_ID_DTYPE)
    cell_upper_arr = np.ascontiguousarray(cell_upper, dtype=INTERNAL_ID_DTYPE)
    q_intensity = np.ascontiguousarray(query.intensity, dtype=ENERGY_DTYPE)

    root_node_ids = forest.trees.root_node_id
    env_offsets = forest.envelopes.node_envelope_offsets
    cell_index = forest.envelopes.cell_index
    max_peak_amplitude = forest.envelopes.max_peak_amplitude

    if _HAVE_NUMBA:
        return _batch_root_bounds_numba(
            tree_ids_arr,
            root_node_ids,
            env_offsets,
            cell_index,
            max_peak_amplitude,
            q_intensity,
            cell_lower_arr,
            cell_upper_arr,
        )
    return _batch_root_bounds_numpy(
        tree_ids_arr,
        root_node_ids,
        env_offsets,
        cell_index,
        max_peak_amplitude,
        q_intensity,
        cell_lower_arr,
        cell_upper_arr,
    )


def batch_node_bounds(
    query: SpectrumPeaks,
    forest: ForestIndex,
    node_ids: Sequence[int] | NDArray[np.int64],
    cell_lower: NDArray[np.int64],
    cell_upper: NDArray[np.int64],
) -> NDArray[np.float64]:
    """批量计算一组节点（根节点或叶节点）的包络峰上界 U_peak(Node)。

    若环境存在 Numba，采用零内存分配 JIT 双二分内核进行寄存器级求值；
    无 Numba 环境时自动降级至 NumPy 向量化复用内核。
    保证与单个 peak_bound(build_query_context(...)) 计算结果数学完全等价，
    并在浮点误差范围内严格遵守零漏检（Zero False Dismissals）性质。
    """
    n_peaks = int(query.mass.shape[0])
    n_nodes = len(node_ids)
    if n_peaks == 0 or n_nodes == 0:
        return np.zeros(n_nodes, dtype=ENERGY_DTYPE)

    if not (cell_lower.shape[0] == cell_upper.shape[0] == query.intensity.shape[0] == n_peaks):
        raise ValueError(
            f"查询向量长度不匹配: mass={n_peaks}, intensity={query.intensity.shape[0]}, "
            f"cell_lower={cell_lower.shape[0]}, cell_upper={cell_upper.shape[0]}"
        )

    node_ids_arr = np.ascontiguousarray(node_ids, dtype=INTERNAL_ID_DTYPE)
    if node_ids_arr.size > 0:
        if bool(np.any(node_ids_arr < 0) or np.any(node_ids_arr >= forest.envelopes.n_nodes)):
            raise IndexError(
                f"node_ids 存在超出合法范围 [0, {forest.envelopes.n_nodes}) 的非法索引"
            )

    cell_lower_arr = np.ascontiguousarray(cell_lower, dtype=INTERNAL_ID_DTYPE)
    cell_upper_arr = np.ascontiguousarray(cell_upper, dtype=INTERNAL_ID_DTYPE)
    q_intensity = np.ascontiguousarray(query.intensity, dtype=ENERGY_DTYPE)

    env_offsets = forest.envelopes.node_envelope_offsets
    cell_index = forest.envelopes.cell_index
    max_peak_amplitude = forest.envelopes.max_peak_amplitude

    if _HAVE_NUMBA:
        return _batch_node_bounds_numba(
            node_ids_arr,
            env_offsets,
            cell_index,
            max_peak_amplitude,
            q_intensity,
            cell_lower_arr,
            cell_upper_arr,
        )
    return _batch_node_bounds_numpy(
        node_ids_arr,
        env_offsets,
        cell_index,
        max_peak_amplitude,
        q_intensity,
        cell_lower_arr,
        cell_upper_arr,
    )

