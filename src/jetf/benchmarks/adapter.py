"""JET-Forest 与 matchms 间的数据结构适配器。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

logging.getLogger("matchms").setLevel(logging.ERROR)

from jetf.cleaning import DEFAULT_CLEAN_CONFIG, MatchmsCleanConfig, clean_spectrum_with_matchms

if TYPE_CHECKING:
    import matchms
    from jetf.preprocessing import PreprocessedLibrary
    from jetf.types import SpectrumMeta, SpectrumPeaks


def check_matchms_available() -> None:
    """检查 matchms 是否安装，未安装时抛出友好的 ImportError。"""
    try:
        import matchms  # noqa: F401
        m_logger = logging.getLogger("matchms")
        m_logger.setLevel(logging.ERROR)
        for h in m_logger.handlers:
            h.setLevel(logging.ERROR)
    except ImportError as exc:
        raise ImportError(
            "未检测到 matchms 库。请通过 `uv pip install -e '.[benchmark]'` 或 `pip install matchms` 安装评测依赖。"
        ) from exc


def jetf_peaks_to_matchms(
    peaks: SpectrumPeaks,
    meta: SpectrumMeta | None = None,
    use_normalized_intensity: bool = True,
    clean: bool = True,
    clean_config: MatchmsCleanConfig | None = None,
) -> matchms.Spectrum:
    """将 JETF SpectrumPeaks 转换为 matchms.Spectrum 对象。

    参数:
    - peaks: JETF SpectrumPeaks 对象
    - meta: 可选的元数据 SpectrumMeta
    - use_normalized_intensity: 若为 True，使用 L2 归一化后的强度 (范数为 1)；若为 False，恢复原始强度
    - clean: 是否启用 matchms 工业级数据清洗流水线 (默认 True)
    - clean_config: 清洗参数配置 (默认 DEFAULT_CLEAN_CONFIG)
    """
    from matchms import Spectrum

    mz = np.asarray(peaks.mass, dtype=np.float64)
    if use_normalized_intensity or peaks.norm <= 0.0:
        intensities = np.asarray(peaks.intensity, dtype=np.float64)
    else:
        intensities = np.asarray(peaks.intensity * peaks.norm, dtype=np.float64)

    metadata: dict[str, object] = {}
    if meta is not None:
        if meta.precursor_mz is not None:
            metadata["precursor_mz"] = float(meta.precursor_mz)
            metadata["pepmass"] = (float(meta.precursor_mz),)
        metadata["ionmode"] = meta.ion_mode.value
        metadata["external_id"] = meta.external_id
        if meta.charge is not None:
            metadata["charge"] = int(meta.charge)
        if meta.raw_metadata:
            metadata.update(meta.raw_metadata)

    spec = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    if clean:
        cfg = clean_config if clean_config is not None else DEFAULT_CLEAN_CONFIG
        cleaned = clean_spectrum_with_matchms(spec, config=cfg)
        if cleaned is not None and len(cleaned.peaks.mz) > 0:
            return cleaned

    return spec


def jetf_library_to_matchms(
    library: PreprocessedLibrary,
    indices: Sequence[int] | None = None,
    use_normalized_intensity: bool = True,
    clean: bool = True,
    clean_config: MatchmsCleanConfig | None = None,
) -> list[matchms.Spectrum]:
    """将 JETF PreprocessedLibrary 中的谱批量转换为 matchms.Spectrum 列表。"""
    rows = range(library.n_spectra) if indices is None else indices
    spectra: list[matchms.Spectrum] = []
    for row in rows:
        peaks = library.peaks.spectrum_at(int(row))
        meta = library.spectra[int(row)]
        spectra.append(
            jetf_peaks_to_matchms(
                peaks,
                meta=meta,
                use_normalized_intensity=use_normalized_intensity,
                clean=clean,
                clean_config=clean_config,
            )
        )
    return spectra
