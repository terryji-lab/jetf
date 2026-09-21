"""质谱标准化预处理：强度与质量幂次变换（默认线性强度）、L2 归一化与 0.02 Da 摘要网格能量聚合。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
        if not (self.mass.shape == self.intensity.shape == self.energy.shape == self.peak_id.shape):
            raise ValueError("LibraryPeaks 的 mass, intensity, energy, peak_id 列长度必须一致")

    @property
    def n_peaks(self) -> int:
        return int(self.mass.shape[0])

    def spectrum_at(self, row: int) -> SpectrumPeaks:
        start = int(self.spectrum_offsets[row])
        end = int(self.spectrum_offsets[row + 1])
        return SpectrumPeaks._create_unchecked(
            mass=self.mass[start:end],
            intensity=self.intensity[start:end],
            energy=self.energy[start:end],
            peak_id=self.peak_id[start:end],
            norm=float(self.norm[row]),
        )


def compute_library_fingerprint(
    source_path: str,
    n_spectra: int,
    n_peaks: int,
    mass: NDArray[np.float64],
    intensity: NDArray[np.float64],
    precursors: Sequence[float | None],
) -> str:
    """计算库的采样内容哈希与元数据复合指纹。

    格式: {n_spectra}:{n_peaks}:{source_path}:{content_digest[:16]}
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(n_spectra).encode("utf-8"))
    hasher.update(str(n_peaks).encode("utf-8"))
    hasher.update(str(source_path).encode("utf-8"))

    # 1. 前体 m/z 数组哈希
    precs = np.array(
        [p if (p is not None and math.isfinite(p)) else np.nan for p in precursors],
        dtype=np.float64,
    )
    hasher.update(precs.tobytes())

    # 2. 峰数据采样哈希 (前 1 MB 约 131,072 个 float64 元素)
    sample_len = min(int(mass.shape[0]), 131072)
    if sample_len > 0:
        hasher.update(mass[:sample_len].tobytes())
        hasher.update(intensity[:sample_len].tobytes())

    digest = hasher.hexdigest()
    return f"{n_spectra}:{n_peaks}:{source_path}:{digest[:16]}"


@dataclass(frozen=True, eq=False)
class PreprocessedLibrary:
    """预处理后的全库对象。"""

    spec: PreprocessSpec
    source_path: str
    spectra: tuple[SpectrumMeta, ...]
    peaks: LibraryPeaks
    resources: LibraryResources
    library_fingerprint: str = ""

    @property
    def n_spectra(self) -> int:
        return len(self.spectra)

    @property
    def fingerprint(self) -> str:
        if self.library_fingerprint:
            return self.library_fingerprint
        return compute_library_fingerprint(
            self.source_path,
            self.n_spectra,
            self.peaks.n_peaks,
            self.peaks.mass,
            self.peaks.intensity,
            [s.precursor_mz for s in self.spectra],
        )


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

    # 0. 负强度清洗截断 (消除基线负噪声) 与 mass 升序保护
    clean_intensity = np.maximum(peaks.intensity, 0.0)
    mass = peaks.mass
    peak_id = peaks.peak_id
    if mass.shape[0] > 1 and not np.all(mass[:-1] <= mass[1:]):
        order = np.argsort(mass, kind="stable")
        mass = mass[order]
        clean_intensity = clean_intensity[order]
        peak_id = peak_id[order]

    # 1. 强度与质量幂次变换 y = intensity^alpha * mass^beta
    # 防御边界：当 alpha=0 时，零强度峰保持为 0（避免 0**0=1 虚增峰）；质量 <=0 且 beta<0 时置 0
    if spec.alpha == 0.0:
        it_trans = np.where(clean_intensity > 0.0, 1.0, 0.0)
    else:
        it_trans = np.power(clean_intensity, spec.alpha)

    if spec.beta == 0.0:
        mz_trans = 1.0
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mz_trans = np.where(mass > 0.0, np.power(mass, spec.beta), 0.0)

    transformed = it_trans * mz_trans
    transformed = np.where(np.isfinite(transformed), transformed, 0.0)

    # 2. L2 范数计算
    l2_sum = float(np.sum(transformed * transformed, dtype=np.float64))
    raw_norm = math.sqrt(l2_sum) if l2_sum > 0.0 and math.isfinite(l2_sum) else 0.0
    if raw_norm == 0.0:
        return SpectrumPeaks(
            mass=np.empty(0, dtype=MASS_DTYPE),
            intensity=np.empty(0, dtype=INTENSITY_DTYPE),
            energy=np.empty(0, dtype=ENERGY_DTYPE),
            peak_id=np.empty(0, dtype=PEAK_ID_DTYPE),
            norm=0.0,
        )

    norm_v = (transformed / raw_norm).astype(INTENSITY_DTYPE)
    energy = (norm_v * norm_v).astype(ENERGY_DTYPE)
    return SpectrumPeaks(
        mass=mass,
        intensity=norm_v,
        energy=energy,
        peak_id=peak_id,
        norm=raw_norm,
    )


def preprocess_library(
    parsed: ParsedLibrary, spec: PreprocessSpec = CORRECTNESS_V1
) -> PreprocessedLibrary:
    """对已解析列式质谱库执行全库批量标准化预处理与摘要网格生成。"""
    n_spectra = parsed.n_spectra
    grid_da = spec.grid_da

    norm_list: list[float] = []
    norm_intensities: list[NDArray[np.float64]] = []
    norm_energies: list[NDArray[np.float64]] = []

    res_cells_list: list[NDArray[np.int64]] = []
    res_energies_list: list[NDArray[np.float64]] = []
    res_offsets: list[int] = [0]

    for i in range(n_spectra):
        st = int(parsed.spectrum_offsets[i])
        sp = int(parsed.spectrum_offsets[i + 1])
        raw_m = parsed.mass[st:sp]
        raw_v = np.maximum(parsed.intensity[st:sp], 0.0)

        if raw_m.shape[0] > 1 and not np.all(raw_m[:-1] <= raw_m[1:]):
            order = np.argsort(raw_m, kind="stable")
            raw_m = raw_m[order]
            raw_v = raw_v[order]

        if spec.alpha == 0.0:
            it_trans = np.where(raw_v > 0.0, 1.0, 0.0)
        else:
            it_trans = np.power(raw_v, spec.alpha)

        if spec.beta == 0.0:
            mz_trans = 1.0
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                mz_trans = np.where(raw_m > 0.0, np.power(raw_m, spec.beta), 0.0)

        t_arr = it_trans * mz_trans
        t_arr = np.where(np.isfinite(t_arr), t_arr, 0.0)

        l2_sum = float(np.sum(t_arr * t_arr, dtype=np.float64))
        if l2_sum > 0.0 and math.isfinite(l2_sum):
            raw_norm = math.sqrt(l2_sum)
            v_arr = (t_arr / raw_norm).astype(INTENSITY_DTYPE)
        else:
            raw_norm = 0.0
            v_arr = np.zeros_like(raw_m, dtype=INTENSITY_DTYPE)

        e_arr = v_arr * v_arr
        norm_list.append(raw_norm)
        norm_intensities.append(v_arr)
        norm_energies.append(e_arr)

        # 0.02 Da 摘要网格能量聚合（加 1e-12 避免浮点舍入导致的 off-by-one 偏差）
        if raw_norm > 0.0:
            cells = np.floor((raw_m + 1e-12) / grid_da).astype(INTERNAL_ID_DTYPE)
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

    fp = compute_library_fingerprint(
        parsed.source_path,
        n_spectra,
        peaks.n_peaks,
        peaks.mass,
        peaks.intensity,
        [s.precursor_mz for s in parsed.spectra],
    )

    return PreprocessedLibrary(
        spec=spec,
        source_path=parsed.source_path,
        spectra=parsed.spectra,
        peaks=peaks,
        resources=resources,
        library_fingerprint=fp,
    )

