"""查询包络上下文与节点峰上界 (Peak Bound) 计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jetf.scoring import sum_float64
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
    max_cell_energy: NDArray[np.float64]

    def __post_init__(self) -> None:
        check_column("cell_index", self.cell_index, INTERNAL_ID_DTYPE)
        check_column("max_peak_amplitude", self.max_peak_amplitude, ENERGY_DTYPE)
        check_column("max_cell_energy", self.max_cell_energy, ENERGY_DTYPE)
        if not (self.cell_index.shape == self.max_peak_amplitude.shape == self.max_cell_energy.shape):
            raise ValueError("NodeEnvelope 的 cell_index, max_peak_amplitude, max_cell_energy 必须等长")

    @property
    def n_cells(self) -> int:
        return int(self.cell_index.shape[0])


def window_cells(
    mass: NDArray[np.float64], tolerance_da: float, grid_da: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """逐查询峰枚举兼容 cell 的保守超集 [cell_lower, cell_upper]。"""
    lower = np.nextafter(mass - tolerance_da, -np.inf) / grid_da
    upper = np.nextafter(mass + tolerance_da, np.inf) / grid_da
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
    energy: NDArray[np.float64]
    cell_lower: NDArray[np.int64]
    cell_upper: NDArray[np.int64]
    peak_offsets: NDArray[np.int64]
    compatible_positions: NDArray[np.int64]
    n_support_cells: int
    support_energy: float

    @property
    def n_peaks(self) -> int:
        return int(self.mass.shape[0])


def build_query_context(
    query: SpectrumPeaks, envelope: NodeEnvelope, tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA
) -> QueryContext:
    """构建查询谱针对该节点包络的上下文。"""
    support = envelope.cell_index
    cell_lower, cell_upper = window_cells(query.mass, tolerance_da, envelope.grid_da)
    counts, starts = _support_spans(support, cell_lower, cell_upper)

    peak_offsets = np.zeros(int(query.mass.shape[0]) + 1, dtype=INTERNAL_ID_DTYPE)
    np.cumsum(counts, out=peak_offsets[1:])

    compat_positions = _expand_spans(starts, counts)
    support_e = sum_float64(query.energy[counts > 0]) if counts.size > 0 else 0.0

    return QueryContext(
        grid_da=envelope.grid_da,
        tolerance_da=tolerance_da,
        mass=query.mass,
        intensity=query.intensity,
        energy=query.energy,
        cell_lower=cell_lower,
        cell_upper=cell_upper,
        peak_offsets=peak_offsets,
        compatible_positions=compat_positions,
        n_support_cells=int(support.shape[0]),
        support_energy=support_e,
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
    return sum_float64(context.intensity * per_peak)
