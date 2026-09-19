"""JET-Forest 核心数据类型与列式数组验证工具。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

# 基础数值类型标准定义
MASS_DTYPE = np.dtype(np.float64)
INTENSITY_DTYPE = np.dtype(np.float64)
ENERGY_DTYPE = np.dtype(np.float64)
INTERNAL_ID_DTYPE = np.dtype(np.int64)
PEAK_ID_DTYPE = np.dtype(np.int64)

DEFAULT_FRAGMENT_TOLERANCE_DA = 0.02
"""质谱片断匹配默认容差 (0.02 Da)。"""


class IonMode(str, Enum):
    """离子模式枚举。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, val: str | None) -> IonMode:
        if not val:
            return cls.UNKNOWN
        v = val.strip().lower()
        if "pos" in v or v in ("+", "1"):
            return cls.POSITIVE
        if "neg" in v or v in ("-", "-1"):
            return cls.NEGATIVE
        return cls.UNKNOWN


@dataclass(frozen=True, slots=True)
class PrecursorWindow:
    """前体质量检索窗口 [mz - tolerance_da, mz + tolerance_da]。"""

    mz: float
    tolerance_da: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mz) or self.mz <= 0.0:
            raise ValueError(f"前体 m/z 必须为正有限数，得到 {self.mz!r}")
        if not math.isfinite(self.tolerance_da) or self.tolerance_da < 0.0:
            raise ValueError(f"前体容差必须为非负有限数，得到 {self.tolerance_da!r}")

    def contains(self, mz: float | None) -> bool:
        """判断给定 m/z 是否落在前体窗口内（闭区间）。"""
        if mz is None or not math.isfinite(mz):
            return False
        return bool(self.mz - self.tolerance_da <= mz <= self.mz + self.tolerance_da)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """谱记录源文件引用。"""

    path: str
    record_index: int


@dataclass(frozen=True)
class SpectrumMeta:
    """单条谱的元数据记录。"""

    external_id: str
    precursor_mz: float | None
    charge: int | None
    ion_mode: IonMode
    source: SourceRef
    raw_metadata: dict[str, str]


@dataclass(frozen=True, eq=False)
class SpectrumPeaks:
    """单条谱的峰数据表示（连续切片）。"""

    mass: NDArray[np.float64]
    intensity: NDArray[np.float64]
    energy: NDArray[np.float64]
    peak_id: NDArray[np.int64]
    norm: float = 1.0

    def __post_init__(self) -> None:
        check_column("mass", self.mass, MASS_DTYPE)
        check_column("intensity", self.intensity, INTENSITY_DTYPE)
        check_column("energy", self.energy, ENERGY_DTYPE)
        check_column("peak_id", self.peak_id, PEAK_ID_DTYPE)
        if not (self.mass.shape == self.intensity.shape == self.energy.shape == self.peak_id.shape):
            raise ValueError("SpectrumPeaks 的 mass, intensity, energy, peak_id 列长度必须一致")

    @property
    def n_peaks(self) -> int:
        return int(self.mass.shape[0])


@dataclass(frozen=True)
class Spectrum:
    """包含元数据与峰数据的完整谱对象。"""

    meta: SpectrumMeta
    peaks: SpectrumPeaks


def check_column(name: str, array: NDArray[Any], expected_dtype: np.dtype) -> None:
    """校验一维 NumPy 数组的类型与维度。"""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} 必须是 ndarray，得到 {type(array).__name__}")
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，得到 {array.ndim} 维")
    if array.dtype != expected_dtype:
        raise TypeError(f"{name} dtype 必须是 {expected_dtype}，得到 {array.dtype}")


def check_peak_columns(
    mass: NDArray[np.float64], intensity: NDArray[np.float64], peak_id: NDArray[np.int64]
) -> None:
    """校验质谱三列数据的类型与长度一致性。"""
    check_column("mass", mass, MASS_DTYPE)
    check_column("intensity", intensity, INTENSITY_DTYPE)
    check_column("peak_id", peak_id, PEAK_ID_DTYPE)
    if not (mass.shape == intensity.shape == peak_id.shape):
        raise ValueError(
            f"峰数据列长度不一致: mass={mass.shape}, intensity={intensity.shape}, peak_id={peak_id.shape}"
        )


def check_spectrum_offsets(
    name: str,
    offsets: NDArray[np.int64],
    n_spectra: int,
    n_items: int,
    item_label: str = "元素总数",
) -> None:
    """校验 spectrum_offsets 的单调性与边界不变量。"""
    check_column(name, offsets, INTERNAL_ID_DTYPE)
    if offsets.shape[0] != n_spectra + 1:
        raise ValueError(f"{name} 长度必须为 n_spectra + 1 = {n_spectra + 1}，得到 {offsets.shape[0]}")
    if offsets.shape[0] > 0 and offsets[0] != 0:
        raise ValueError(f"{name} 首项必须为 0，得到 {offsets[0]}")
    if offsets.shape[0] > 0 and offsets[-1] != n_items:
        raise ValueError(f"{name} 末项必须等于 {item_label} {n_items}，得到 {offsets[-1]}")
    if offsets.shape[0] > 1 and bool((np.diff(offsets) < 0).any()):
        raise ValueError(f"{name} 必须单调不减")
