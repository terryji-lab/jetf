"""质谱标准化预处理：强度变换（平方根）、L2 归一化与 0.02 Da 摘要网格能量聚合。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from jetf.mgf import ParsedLibrary
from jetf.types import (
    ENERGY_DTYPE,
    INTENSITY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    PEAK_ID_DTYPE,
    SpectrumMeta,
    SpectrumPeaks,
    check_column,
    check_spectrum_offsets,
)


@dataclass(frozen=True)
class PreprocessSpec:
    """预处理规格定义。"""

    spec_id: str = "correctness_v1"
    version: str = "1"
    alpha: float = 1.0  # 强度幂次变换指数（默认 1.0，线性强度）
    beta: float = 0.0   # 质量幂次变换指数（默认 0.0，不按质量加权）
    grid_da: float = 0.02  # 摘要网格默认 0.02 Da

    @property
    def versioned_id(self) -> str:
        return f"{self.spec_id}/{self.version}"

    def __post_init__(self) -> None:
        if not self.spec_id:
            raise ValueError("spec_id 不能为空")
        if not self.version:
            raise ValueError("version 不能为空")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError(f"alpha 必须是非负有限数，得到 {self.alpha!r}")
        if not math.isfinite(self.beta):
            raise ValueError(f"beta 必须是有限数，得到 {self.beta!r}")
        if self.grid_da <= 0.0:
            raise ValueError("grid_da 必须为正")


CORRECTNESS_V1 = PreprocessSpec()


@dataclass(frozen=True, eq=False)
class LibraryResources:
    """按谱聚合的摘要网格 cell 与能量数据（连续列式存储）。"""

    grid_da: float
    cell_index: NDArray[np.int64]
    cell_energy: NDArray[np.float64]
    spectrum_offsets: NDArray[np.int64]

    def __post_init__(self) -> None:
        check_column("cell_index", self.cell_index, INTERNAL_ID_DTYPE)
        check_column("cell_energy", self.cell_energy, ENERGY_DTYPE)
        check_column("spectrum_offsets", self.spectrum_offsets, INTERNAL_ID_DTYPE)


@dataclass(frozen=True, eq=False)
class LibraryPeaks:
    """全库归一化后的精评峰集合（连续列式切片）。"""

    mass: NDArray[np.float64]
    intensity: NDArray[np.float64]
    energy: NDArray[np.float64]
    peak_id: NDArray[np.int64]
    spectrum_offsets: NDArray[np.int64]
    norm: NDArray[np.float64]  # 每条谱的原始 L2 范数

    def __post_init__(self) -> None:
        check_column("mass", self.mass, MASS_DTYPE)
        check_column("intensity", self.intensity, INTENSITY_DTYPE)
        check_column("energy", self.energy, ENERGY_DTYPE)
        check_column("peak_id", self.peak_id, PEAK_ID_DTYPE)
        check_column("spectrum_offsets", self.spectrum_offsets, INTERNAL_ID_DTYPE)
        check_column("norm", self.norm, ENERGY_DTYPE)

    def spectrum_at(self, row: int) -> SpectrumPeaks:
        start = int(self.spectrum_offsets[row])
        end = int(self.spectrum_offsets[row + 1])
        return SpectrumPeaks(
            mass=self.mass[start:end],
            intensity=self.intensity[start:end],
            energy=self.energy[start:end],
            peak_id=self.peak_id[start:end],
            norm=float(self.norm[row]),
        )


@dataclass(frozen=True, eq=False)
class PreprocessedLibrary:
    """预处理后的全库对象。"""

    spec: PreprocessSpec
    source_path: str
    spectra: tuple[SpectrumMeta, ...]
    peaks: LibraryPeaks
    resources: LibraryResources

    @property
    def n_spectra(self) -> int:
        return len(self.spectra)


def preprocess_query(
    peaks: SpectrumPeaks,
    spec: PreprocessSpec = CORRECTNESS_V1,
) -> SpectrumPeaks:
    """对单条查询谱执行强度变换与 L2 归一化。"""
    if peaks.mass.size == 0:
        return SpectrumPeaks(
            mass=np.empty(0, dtype=MASS_DTYPE),
            intensity=np.empty(0, dtype=INTENSITY_DTYPE),
            energy=np.empty(0, dtype=ENERGY_DTYPE),
            peak_id=np.empty(0, dtype=PEAK_ID_DTYPE),
            norm=0.0,
        )

    # 1. 强度与质量幂次变换 y = intensity^alpha * mass^beta
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.power(peaks.intensity, spec.alpha) * np.power(peaks.mass, spec.beta)
    # 2. L2 范数计算
    l2_sum = float(np.sum(transformed * transformed, dtype=np.float64))
    raw_norm = math.sqrt(l2_sum) if l2_sum > 0.0 and math.isfinite(l2_sum) else 0.0
    if raw_norm > 0.0:
        norm_v = (transformed / raw_norm).astype(INTENSITY_DTYPE)
    else:
        norm_v = np.zeros_like(transformed, dtype=INTENSITY_DTYPE)
        raw_norm = 0.0

    energy = (norm_v * norm_v).astype(ENERGY_DTYPE)
    return SpectrumPeaks(
        mass=peaks.mass,
        intensity=norm_v,
        energy=energy,
        peak_id=peaks.peak_id,
        norm=raw_norm,
    )


def preprocess_library(
    parsed: ParsedLibrary,
    spec: PreprocessSpec = CORRECTNESS_V1,
) -> PreprocessedLibrary:
    """对全库执行预处理并构建 0.02 Da 摘要网格索引。"""
    n_spectra = parsed.n_spectra
    grid_da = spec.grid_da

    norm_list: list[float] = []
    norm_intensities: list[NDArray[np.float64]] = []
    norm_energies: list[NDArray[np.float64]] = []

    res_cells_list: list[NDArray[np.int64]] = []
    res_energies_list: list[NDArray[np.float64]] = []
    res_offsets: list[int] = [0]

    for row in range(n_spectra):
        start = int(parsed.spectrum_offsets[row])
        end = int(parsed.spectrum_offsets[row + 1])
        n_p = end - start

        if n_p == 0:
            norm_list.append(0.0)
            res_offsets.append(res_offsets[-1])
            continue

        raw_m = parsed.mass[start:end]
        raw_it = parsed.intensity[start:end]

        # 强度与质量变换与归一化
        with np.errstate(over="ignore", invalid="ignore"):
            t_it = np.power(raw_it, spec.alpha) * np.power(raw_m, spec.beta)
        l2_sum = float(np.sum(t_it * t_it, dtype=np.float64))
        raw_norm = math.sqrt(l2_sum) if l2_sum > 0.0 and math.isfinite(l2_sum) else 0.0

        if raw_norm > 0.0:
            v_arr = (t_it / raw_norm).astype(INTENSITY_DTYPE)
        else:
            v_arr = np.zeros_like(t_it, dtype=INTENSITY_DTYPE)
            raw_norm = 0.0

        e_arr = v_arr * v_arr
        norm_list.append(raw_norm)
        norm_intensities.append(v_arr)
        norm_energies.append(e_arr)

        # 0.02 Da 摘要网格能量聚合
        if raw_norm > 0.0:
            cells = np.floor(raw_m / grid_da).astype(INTERNAL_ID_DTYPE)
            uniq_cells, inv_idx = np.unique(cells, return_inverse=True)
            cell_e = np.bincount(inv_idx, weights=e_arr).astype(ENERGY_DTYPE)

            # 只保留正能量 cell
            pos_mask = cell_e > 0.0
            u_c = uniq_cells[pos_mask]
            u_e = cell_e[pos_mask]

            res_cells_list.append(u_c)
            res_energies_list.append(u_e)
            res_offsets.append(res_offsets[-1] + len(u_c))
        else:
            res_offsets.append(res_offsets[-1])

    flat_v = np.concatenate(norm_intensities) if norm_intensities else np.empty(0, dtype=INTENSITY_DTYPE)
    flat_e = np.concatenate(norm_energies) if norm_energies else np.empty(0, dtype=ENERGY_DTYPE)

    peaks = LibraryPeaks(
        mass=parsed.mass,
        intensity=flat_v,
        energy=flat_e,
        peak_id=parsed.peak_id,
        spectrum_offsets=parsed.spectrum_offsets,
        norm=np.array(norm_list, dtype=ENERGY_DTYPE),
    )

    resources = LibraryResources(
        grid_da=grid_da,
        cell_index=np.concatenate(res_cells_list) if res_cells_list else np.empty(0, dtype=INTERNAL_ID_DTYPE),
        cell_energy=np.concatenate(res_energies_list) if res_energies_list else np.empty(0, dtype=ENERGY_DTYPE),
        spectrum_offsets=np.array(res_offsets, dtype=INTERNAL_ID_DTYPE),
    )

    return PreprocessedLibrary(
        spec=spec,
        source_path=parsed.source_path,
        spectra=parsed.spectra,
        peaks=peaks,
        resources=resources,
    )
