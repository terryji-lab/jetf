"""基于 matchms 的质谱清洗与预处理流水线 (Data Cleaning Pipeline)。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from jetf.mgf import ParsedLibrary, RejectedRecord, RejectReason
from jetf.types import (
    INTENSITY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    PEAK_ID_DTYPE,
    IonMode,
    SourceRef,
    SpectrumMeta,
    SpectrumPeaks,
)


def silence_matchms_logging() -> None:
    """静音 matchms 的 WARNING 与 INFO 日志输出，避免海量谱图处理时 I/O 阻塞。"""
    matchms_logger = logging.getLogger("matchms")
    matchms_logger.setLevel(logging.ERROR)
    for handler in matchms_logger.handlers:
        handler.setLevel(logging.ERROR)


# 模块加载时执行静音
silence_matchms_logging()


@dataclass(frozen=True)
class MatchmsCleanConfig:
    """matchms 质谱清洗配置参数。"""

    metadata_cleaning: bool = True
    mz_min: float = 10.0
    mz_max: float = 2000.0
    min_relative_intensity: float = 0.001
    max_peaks: int = 300
    min_peaks: int = 3
    normalize_intensities: bool = True

    def __post_init__(self) -> None:
        if self.mz_min < 0.0:
            raise ValueError(f"mz_min 必须非负，得到 {self.mz_min}")
        if self.mz_max <= self.mz_min:
            raise ValueError(f"mz_max ({self.mz_max}) 必须大于 mz_min ({self.mz_min})")
        if self.min_relative_intensity < 0.0:
            raise ValueError(f"min_relative_intensity 必须非负，得到 {self.min_relative_intensity}")
        if self.max_peaks <= 0:
            raise ValueError(f"max_peaks 必须为正整数，得到 {self.max_peaks}")
        if self.min_peaks < 0:
            raise ValueError(f"min_peaks 必须非负，得到 {self.min_peaks}")


DEFAULT_CLEAN_CONFIG = MatchmsCleanConfig()


def clean_spectrum_with_matchms(
    spectrum: "matchms.Spectrum",
    config: MatchmsCleanConfig = DEFAULT_CLEAN_CONFIG,
) -> "matchms.Spectrum" | None:
    """使用 matchms 标准算子对单条谱图执行元数据与碎片峰清洗。"""
    import matchms.filtering as mf

    silence_matchms_logging()

    s = spectrum
    if s is None:
        return None

    # 1. 元数据清洗与推导
    if config.metadata_cleaning:
        s = mf.default_filters(s)
        if s is None:
            return None
        s = mf.add_precursor_mz(s)
        if s is None:
            return None
        s = mf.derive_ionmode(s)
        if s is None:
            return None
        s = mf.correct_charge(s)
        if s is None:
            return None

    # 2. 峰范围过滤 (m/z 边界)
    s = mf.select_by_mz(s, mz_from=config.mz_min, mz_to=config.mz_max, clone=False)
    if s is None or len(s.peaks.mz) == 0:
        return None

    # 3. 峰相对强度过滤
    if config.min_relative_intensity > 0.0:
        s = mf.select_by_relative_intensity(s, intensity_from=config.min_relative_intensity, clone=False)
        if s is None or len(s.peaks.mz) == 0:
            return None

    # 4. 保留最高强度 Top-N 峰 (截断长尾低信噪比峰)
    if config.max_peaks > 0:
        s = mf.reduce_to_number_of_peaks(s, n_max=config.max_peaks, clone=False)
        if s is None or len(s.peaks.mz) == 0:
            return None

    # 5. 最低峰数校验
    if config.min_peaks > 0:
        s = mf.require_minimum_number_of_peaks(s, n_required=config.min_peaks, clone=False)
        if s is None or len(s.peaks.mz) < config.min_peaks:
            return None

    # 6. 强度标准化
    if config.normalize_intensities:
        s = mf.normalize_intensities(s)
        if s is None or len(s.peaks.mz) == 0:
            return None

    return s


def clean_single_spectrum_record(
    meta: SpectrumMeta,
    mass: NDArray[np.float64],
    intensity: NDArray[np.float64],
    config: MatchmsCleanConfig = DEFAULT_CLEAN_CONFIG,
) -> tuple[SpectrumMeta, NDArray[np.float64], NDArray[np.float64]] | None:
    """对单条谱记录元数据与峰数组执行 matchms 清洗。

    若谱图无效或有效峰数低于 config.min_peaks 则返回 None；
    否则返回 (updated_meta, cleaned_mass, cleaned_intensity)。
    """
    from matchms import Spectrum

    metadata: dict[str, object] = {}
    if meta.raw_metadata:
        metadata.update(meta.raw_metadata)
    if meta.external_id:
        metadata["title"] = meta.external_id
        metadata["spectrumid"] = meta.external_id
    if meta.precursor_mz is not None and math.isfinite(meta.precursor_mz):
        metadata["precursor_mz"] = float(meta.precursor_mz)
        metadata["pepmass"] = (float(meta.precursor_mz),)
    if meta.ion_mode != IonMode.UNKNOWN:
        metadata["ionmode"] = meta.ion_mode.value
    if meta.charge is not None:
        metadata["charge"] = int(meta.charge)

    raw_spec = Spectrum(mz=mass, intensities=intensity, metadata=metadata)
    cleaned_spec = clean_spectrum_with_matchms(raw_spec, config=config)

    if cleaned_spec is None or len(cleaned_spec.peaks.mz) < config.min_peaks:
        return None

    c_mz = np.asarray(cleaned_spec.peaks.mz, dtype=np.float64)
    c_it = np.asarray(cleaned_spec.peaks.intensities, dtype=np.float64)

    # 确保按 mass 严格升序排序
    if not np.all(c_mz[:-1] <= c_mz[1:]):
        order = np.argsort(c_mz, kind="stable")
        c_mz = c_mz[order]
        c_it = c_it[order]

    # 提取更新后的元数据
    cleaned_meta_dict = cleaned_spec.metadata
    prec_val = cleaned_meta_dict.get("precursor_mz")
    try:
        new_prec = float(prec_val) if prec_val is not None and math.isfinite(float(prec_val)) else meta.precursor_mz
    except (ValueError, TypeError):
        new_prec = meta.precursor_mz

    ion_str = cleaned_meta_dict.get("ionmode")
    new_ion = IonMode.from_str(str(ion_str)) if ion_str else meta.ion_mode

    chg_val = cleaned_meta_dict.get("charge")
    try:
        new_charge = int(chg_val) if chg_val is not None else meta.charge
    except (ValueError, TypeError):
        new_charge = meta.charge

    updated_meta = SpectrumMeta(
        external_id=meta.external_id,
        ion_mode=new_ion,
        precursor_mz=new_prec,
        charge=new_charge,
        source=meta.source,
        raw_metadata=meta.raw_metadata,
    )
    return updated_meta, c_mz, c_it


def clean_parsed_library(
    parsed: ParsedLibrary,
    config: MatchmsCleanConfig = DEFAULT_CLEAN_CONFIG,
) -> ParsedLibrary:
    """对 ParsedLibrary 全量执行 matchms 清洗，并重新编译为紧凑列式结构。

    清洗动作包含：
    - 去除低于相对强度阈值的噪音峰
    - 限制每条谱的最大峰数 (默认 300)
    - 剔除有效峰数过少或损坏的谱图
    - 标准化前体质量、电荷与离子模式元数据
    """
    cleaned_spectra: list[SpectrumMeta] = []
    mass_chunks: list[NDArray[np.float64]] = []
    intensity_chunks: list[NDArray[np.float64]] = []
    offsets = [0]
    new_rejected = list(parsed.rejected)

    for row in range(parsed.n_spectra):
        meta = parsed.spectra[row]
        p_start = int(parsed.spectrum_offsets[row])
        p_end = int(parsed.spectrum_offsets[row + 1])

        mz = np.asarray(parsed.mass[p_start:p_end], dtype=np.float64)
        intensities = np.asarray(parsed.intensity[p_start:p_end], dtype=np.float64)

        res = clean_single_spectrum_record(meta, mz, intensities, config=config)
        if res is None:
            new_rejected.append(
                RejectedRecord(
                    source=meta.source,
                    external_id=meta.external_id,
                    reason=RejectReason.UNPARSEABLE,
                    detail=f"经 matchms 预处理后谱图无效或峰数不足 ({config.min_peaks})",
                )
            )
            continue

        updated_meta, c_mz, c_it = res
        cleaned_spectra.append(updated_meta)
        mass_chunks.append(c_mz)
        intensity_chunks.append(c_it)
        offsets.append(offsets[-1] + int(c_mz.shape[0]))

    final_mass = np.concatenate(mass_chunks) if mass_chunks else np.empty(0, dtype=MASS_DTYPE)
    final_it = np.concatenate(intensity_chunks) if intensity_chunks else np.empty(0, dtype=INTENSITY_DTYPE)
    offsets_arr = np.array(offsets, dtype=INTERNAL_ID_DTYPE)
    total_peaks = int(offsets_arr[-1]) if offsets_arr.size > 0 else 0

    # 预分配连续 peak_id，避免循环中创建数百万个微型 np.arange 数组
    final_pid = np.empty(total_peaks, dtype=PEAK_ID_DTYPE)
    for i in range(len(cleaned_spectra)):
        st, ed = int(offsets_arr[i]), int(offsets_arr[i + 1])
        final_pid[st:ed] = np.arange(ed - st, dtype=PEAK_ID_DTYPE)

    return ParsedLibrary(
        source_path=parsed.source_path,
        spectra=tuple(cleaned_spectra),
        mass=final_mass,
        intensity=final_it,
        peak_id=final_pid,
        spectrum_offsets=offsets_arr,
        rejected=tuple(new_rejected),
    )
