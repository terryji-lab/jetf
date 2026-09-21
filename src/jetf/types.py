"""JET-Forest 核心数据类型与列式数组验证工具。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
    """前体离子荷质比检索窗口 [mz - tolerance_da, mz + tolerance_da]。"""

    mz: float
    tolerance_da: float

    def __post_init__(self) -> None:
        if self.mz <= 0.0 or not math.isfinite(self.mz):
            raise ValueError(f"前体 mz 必须为正有限数，得到 {self.mz}")
        if self.tolerance_da <= 0.0 or not math.isfinite(self.tolerance_da):
            raise ValueError(f"前体容差 tolerance_da 必须为正有限数，得到 {self.tolerance_da}")

    @property
    def min_mz(self) -> float:
        return self.mz - self.tolerance_da

    @property
    def max_mz(self) -> float:
        return self.mz + self.tolerance_da

    def contains(self, mz: float | None) -> bool:
        """判定目标前体 mz 是否落在窗口内。"""
        if mz is None or math.isnan(mz):
            return False
        return abs(mz - self.mz) <= self.tolerance_da


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
    raw_metadata: dict[str, str] = field(default_factory=dict)


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
        if np.any(self.intensity < 0.0):
            raise ValueError("SpectrumPeaks 峰强度 (intensity) 包含负数，违反非负前置条件")
        if self.mass.shape[0] > 1 and not np.all(self.mass[:-1] <= self.mass[1:]):
            raise ValueError("SpectrumPeaks 峰质量 (mass) 必须按升序排列")

    @classmethod
    def _create_unchecked(
        cls,
        mass: NDArray[np.float64],
        intensity: NDArray[np.float64],
        energy: NDArray[np.float64],
        peak_id: NDArray[np.int64],
        norm: float = 1.0,
    ) -> SpectrumPeaks:
        obj = object.__new__(cls)
        object.__setattr__(obj, "mass", mass)
        object.__setattr__(obj, "intensity", intensity)
        object.__setattr__(obj, "energy", energy)
        object.__setattr__(obj, "peak_id", peak_id)
        object.__setattr__(obj, "norm", norm)
        return obj

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


def validate_query(query: SpectrumPeaks) -> None:
    """校验查询谱的合法性与数学不变量前提。"""
    if query.mass.size == 0:
        return
    if np.any(query.intensity < 0.0):
        raise ValueError("查询谱峰强度包含负数，违反非负前置条件")
    if query.mass.shape[0] > 1 and not np.all(query.mass[:-1] <= query.mass[1:]):
        raise ValueError("查询谱峰质量 (mass) 必须按升序排列")
    l2_sum = float(np.sum(query.intensity * query.intensity, dtype=np.float64))
    l2_norm = math.sqrt(l2_sum) if l2_sum > 0.0 else 0.0
    if l2_norm == 0.0:
        raise ValueError("查询谱所有峰强度均为 0，无可匹配特征，无法参与相似度检索")
    if abs(l2_norm - 1.0) > 1e-4:
        raise ValueError(
            f"查询谱必须经 L2 归一化 (||v||=1.0)，当前范数为 {l2_norm:.6f}。"
            f"请先调用 preprocess_query(query) 进行预处理。"
        )

