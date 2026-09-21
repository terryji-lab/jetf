"""高性能 MGF 质谱文件解析器：转换为内存连续列式结构 ParsedLibrary。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jetf.types import (
    INTENSITY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    PEAK_ID_DTYPE,
    IonMode,
    SourceRef,
    SpectrumMeta,
    SpectrumPeaks,
    check_peak_columns,
    check_spectrum_offsets,
)

_BEGIN_IONS = "BEGIN IONS"
_END_IONS = "END IONS"
_EXTERNAL_ID_KEYS = ("SPECTRUM_ID", "SPECTRUMID", "TITLE", "NAME")
_ION_MODE_VALUES = {"positive": IonMode.POSITIVE, "negative": IonMode.NEGATIVE}


class RejectReason(str, Enum):
    """记录被隔离的原因。"""

    NON_FINITE_MZ = "non_finite_mz"
    NON_POSITIVE_MZ = "non_positive_mz"
    NEGATIVE_INTENSITY = "negative_intensity"
    NON_FINITE_INTENSITY = "non_finite_intensity"
    UNPARSEABLE = "unparseable"
    MISSING_EXTERNAL_ID = "missing_external_id"


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """被隔离的记录。"""

    source: SourceRef
    external_id: str | None
    reason: RejectReason
    detail: str


@dataclass(frozen=True, slots=True)
class _Record:
    index: int
    header: tuple[str, ...]
    peaks: tuple[str, ...]
    terminated: bool


class _RecordRejected(Exception):
    def __init__(self, reason: RejectReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, eq=False)
class ParsedLibrary:
    """全库谱数据：元数据列表 + 连续列式峰数据。"""

    source_path: str
    spectra: tuple[SpectrumMeta, ...]
    mass: NDArray[np.float64]
    intensity: NDArray[np.float64]
    peak_id: NDArray[np.int64]
    spectrum_offsets: NDArray[np.int64]
    rejected: tuple[RejectedRecord, ...] = ()
    n_rejected: int | None = None

    def __post_init__(self) -> None:
        check_peak_columns(self.mass, self.intensity, self.peak_id)
        check_spectrum_offsets(
            "spectrum_offsets", self.spectrum_offsets, len(self.spectra), self.n_peaks, "峰总数"
        )

    @property
    def n_spectra(self) -> int:
        return len(self.spectra)

    @property
    def n_peaks(self) -> int:
        return int(self.mass.shape[0])

    @property
    def rejected_count(self) -> int:
        if self.n_rejected is not None:
            return self.n_rejected
        return len(self.rejected)

    def peak_counts(self) -> NDArray[np.int64]:
        return np.diff(self.spectrum_offsets)

    def spectrum_at(self, row: int) -> SpectrumPeaks:
        start = int(self.spectrum_offsets[row])
        end = int(self.spectrum_offsets[row + 1])
        m = self.mass[start:end]
        it = self.intensity[start:end]
        pid = self.peak_id[start:end]
        return SpectrumPeaks(
            mass=m,
            intensity=it,
            energy=it * it,
            peak_id=pid,
            norm=float(np.sqrt(np.sum(it * it))) if it.size > 0 else 0.0,
        )


def _concat_columns(chunks: list[NDArray[np.number]], dtype: np.dtype) -> NDArray[np.number]:
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)


def _print_mgf_progress(total_records: int, valid_count: int, rejected_count: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    rate = total_records / elapsed if elapsed > 0 else 0.0
    print(
        f"    [解析进度] 已处理 {total_records:,} 条谱 | 有效: {valid_count:,} | "
        f"隔离: {rejected_count:,} | 速率: {rate:.1f} spec/s | 已用时: {elapsed:.1f}s",
        flush=True,
    )


def parse_mgf(
    path: str | Path,
    max_records: int | None = None,
    clean_config: Any | None = None,
    keep_rejected: bool = True,
    progress_interval: int | None = 20000,
) -> ParsedLibrary:
    """解析 MGF 文件：合法记录进谱列表，非法记录隔离在 rejected。

    若指定 clean_config，则在流式解析时直接执行 matchms 工业级清洗，
    避免在内存中积聚全量未清洗峰数组，大幅降低百万级谱库的物理内存峰值。
    若 keep_rejected 为 False，则仅统计隔离记录计数，不再保留详细对象以节省百万级谱库内存。
    """
    source_path = str(path)
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"质谱文件不存在: {file_path}")

    spectra: list[SpectrumMeta] = []
    rejected: list[RejectedRecord] = []
    rejected_count = 0
    mass_chunks: list[NDArray[np.float64]] = []
    intensity_chunks: list[NDArray[np.float64]] = []
    offsets = [0]
    total_records = 0
    t0 = time.perf_counter()

    with open(source_path, "r", encoding="utf-8-sig", errors="replace") as handle:
        for record in _iter_records(handle):
            total_records += 1
            source = SourceRef(path=source_path, record_index=record.index)
            try:
                meta, mass, intensity, _ = _parse_record(record, source)
            except _RecordRejected as exc:
                rejected_count += 1
                if keep_rejected:
                    rejected.append(
                        RejectedRecord(
                            source=source,
                            external_id=_best_effort_external_id(record),
                            reason=exc.reason,
                            detail=exc.detail,
                        )
                    )
                if progress_interval and total_records % progress_interval == 0:
                    _print_mgf_progress(total_records, len(spectra), rejected_count, t0)
                continue

            if clean_config is not None:
                from jetf.cleaning import clean_single_spectrum_record

                res = clean_single_spectrum_record(meta, mass, intensity, config=clean_config)
                if res is None:
                    rejected_count += 1
                    if keep_rejected:
                        rejected.append(
                            RejectedRecord(
                                source=source,
                                external_id=meta.external_id,
                                reason=RejectReason.UNPARSEABLE,
                                detail=f"经 matchms 预处理后谱图无效或峰数不足 ({clean_config.min_peaks})",
                            )
                        )
                    if progress_interval and total_records % progress_interval == 0:
                        _print_mgf_progress(total_records, len(spectra), rejected_count, t0)
                    continue
                meta, mass, intensity = res

            spectra.append(meta)
            mass_chunks.append(mass)
            intensity_chunks.append(intensity)
            offsets.append(offsets[-1] + int(mass.shape[0]))

            if progress_interval and total_records % progress_interval == 0:
                _print_mgf_progress(total_records, len(spectra), rejected_count, t0)

            if max_records is not None and len(spectra) >= max_records:
                break

    offsets_arr = np.array(offsets, dtype=INTERNAL_ID_DTYPE)
    total_peaks = int(offsets_arr[-1]) if offsets_arr.size > 0 else 0
    final_pid = np.empty(total_peaks, dtype=PEAK_ID_DTYPE)
    for i in range(len(spectra)):
        st, ed = int(offsets_arr[i]), int(offsets_arr[i + 1])
        final_pid[st:ed] = np.arange(ed - st, dtype=PEAK_ID_DTYPE)

    return ParsedLibrary(
        source_path=source_path,
        spectra=tuple(spectra),
        mass=_concat_columns(mass_chunks, MASS_DTYPE),
        intensity=_concat_columns(intensity_chunks, INTENSITY_DTYPE),
        peak_id=final_pid,
        spectrum_offsets=offsets_arr,
        rejected=tuple(rejected),
        n_rejected=rejected_count,
    )


def _iter_records(lines: Iterable[str]) -> Iterator[_Record]:
    header: list[str] = []
    peaks: list[str] = []
    open_index: int | None = None
    counter = 0

    for raw_line in lines:
        text = raw_line.rstrip("\r\n")
        stripped = text.strip()
        if stripped == _BEGIN_IONS:
            if open_index is not None:
                yield _Record(open_index, tuple(header), tuple(peaks), False)
            open_index = counter
            counter += 1
            header, peaks = [], []
            continue
        if stripped == _END_IONS:
            if open_index is None:
                continue
            yield _Record(open_index, tuple(header), tuple(peaks), True)
            open_index = None
            continue
        if open_index is None or not stripped:
            continue
        (header if _is_header_line(text) else peaks).append(text)

    if open_index is not None:
        yield _Record(open_index, tuple(header), tuple(peaks), False)


def _is_header_line(text: str) -> bool:
    if "=" not in text:
        return False
    try:
        float(text.partition("=")[0].strip())
    except ValueError:
        return True
    return False


def _parse_record(
    record: _Record, source: SourceRef
) -> tuple[SpectrumMeta, NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    raw_metadata, fields = _header_fields(record)
    external_id = _external_id(fields)
    raw_prec = (
        fields.get("PEPMASS")
        or fields.get("PRECURSOR_MZ")
        or fields.get("PRECURSORMZ")
        or fields.get("PARENT_MASS")
    )
    precursor_mz = _parse_precursor_mz(raw_prec)
    precursor_charge = _parse_charge(fields.get("CHARGE"))
    if not record.terminated:
        raise _RecordRejected(RejectReason.UNPARSEABLE, "记录缺少 END IONS")
    mass, intensity, peak_id = _parse_peaks(record)
    meta = SpectrumMeta(
        external_id=external_id,
        ion_mode=_parse_ion_mode(fields.get("IONMODE")),
        precursor_mz=precursor_mz,
        charge=precursor_charge,
        source=source,
        raw_metadata=raw_metadata,
    )
    return meta, mass, intensity, peak_id


def _header_fields(record: _Record) -> tuple[dict[str, str], dict[str, str]]:
    raw_metadata: dict[str, str] = {}
    fields: dict[str, str] = {}
    for text in record.header:
        key, _, value = text.partition("=")
        raw_key = key.strip()
        raw_val = value.strip()
        raw_metadata[raw_key] = raw_val
        fields[raw_key.upper()] = raw_val
    return raw_metadata, fields


def _best_effort_external_id(record: _Record) -> str | None:
    _, fields = _header_fields(record)
    for key in _EXTERNAL_ID_KEYS:
        value = fields.get(key)
        if value:
            return value
    return None


def _external_id(fields: dict[str, str]) -> str:
    for key in _EXTERNAL_ID_KEYS:
        value = fields.get(key)
        if value:
            return value
    raise _RecordRejected(
        RejectReason.MISSING_EXTERNAL_ID,
        f"记录缺少外部 ID ({'/'.join(_EXTERNAL_ID_KEYS)})",
    )


def _parse_ion_mode(raw: str | None) -> IonMode:
    return IonMode.from_str(raw)


def _parse_precursor_mz(raw: str | None) -> float | None:
    if raw is None or not raw.strip():
        return None
    token = raw.split()[0]
    try:
        value = float(token)
    except ValueError as exc:
        raise _RecordRejected(RejectReason.UNPARSEABLE, f"前体质量无法解析: {raw!r}") from exc
    if not np.isfinite(value):
        raise _RecordRejected(RejectReason.UNPARSEABLE, f"前体质量非有限: {raw!r}")
    return value


def _parse_charge(raw: str | None) -> int | None:
    if raw is None or not raw.strip():
        return None
    token = raw.strip()
    # 支持多电荷表达（如 "2+ and 3+", "2+,3+", "2+ / 3+"）：提取首个子串
    for sep in (" and ", ",", "/", ";"):
        if sep in token:
            token = token.split(sep, 1)[0].strip()
            break
    parts = token.split()
    if parts:
        token = parts[0]

    sign = 1
    if token and token[-1] in "+-":
        sign = -1 if token[-1] == "-" else 1
        token = token[:-1]
    elif token and token[0] in "+-":
        sign = -1 if token[0] == "-" else 1
        token = token[1:]
    try:
        charge = int(token) * sign
    except ValueError as exc:
        raise _RecordRejected(RejectReason.UNPARSEABLE, f"CHARGE 无法解析: {raw!r}") from exc
    return charge


def _parse_peaks(record: _Record) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    lines = record.peaks
    if not lines:
        return (
            np.empty(0, dtype=MASS_DTYPE),
            np.empty(0, dtype=INTENSITY_DTYPE),
            np.empty(0, dtype=PEAK_ID_DTYPE),
        )

    tokens = " ".join(lines).split()
    try:
        values = np.array(tokens, dtype=np.float64)
    except ValueError:
        raise _RecordRejected(RejectReason.UNPARSEABLE, _locate_bad_peak_line(lines)) from None
    if values.shape[0] != 2 * len(lines):
        raise _RecordRejected(RejectReason.UNPARSEABLE, _locate_bad_peak_line(lines))

    pairs = values.reshape(-1, 2)
    mass: NDArray[np.float64] = pairs[:, 0]
    intensity: NDArray[np.float64] = pairs[:, 1]
    _reject_illegal_peaks(mass, intensity)

    peak_id: NDArray[np.int64] = np.arange(len(lines), dtype=PEAK_ID_DTYPE)
    keep = intensity > 0.0
    mass, intensity, peak_id = mass[keep], intensity[keep], peak_id[keep]

    order = np.lexsort((peak_id, mass))
    return mass[order], intensity[order], peak_id[order]


def _locate_bad_peak_line(lines: tuple[str, ...]) -> str:
    for offset, text in enumerate(lines):
        parts = text.split()
        if len(parts) != 2:
            return f"第 {offset} 个峰行不是两列: {text.strip()!r}"
        for token in parts:
            try:
                float(token)
            except ValueError:
                return f"第 {offset} 个峰行不是数值: {text.strip()!r}"
    return f"峰行数值个数与行数不一致 ({len(lines)} 行)"


_PEAK_FAULT_REASONS = (
    RejectReason.NEGATIVE_INTENSITY,
    RejectReason.NON_FINITE_INTENSITY,
    RejectReason.NON_POSITIVE_MZ,
    RejectReason.NON_FINITE_MZ,
)


def _reject_illegal_peaks(mass: NDArray[np.float64], intensity: NDArray[np.float64]) -> None:
    codes = np.zeros(mass.shape[0], dtype=np.int8)
    labels: list[str] = []
    for code, (mask, label) in enumerate(
        (
            (intensity < 0.0, "强度为负"),
            (~np.isfinite(intensity), "强度非有限"),
            (mass <= 0.0, "m/z 非正"),
            (~np.isfinite(mass), "m/z 非有限"),
        ),
        start=1,
    ):
        codes[mask] = code
        labels.append(label)

    if not codes.any():
        return
    offset = int(np.argmax(codes > 0))
    raise _RecordRejected(
        _PEAK_FAULT_REASONS[codes[offset] - 1],
        f"峰 {offset} {labels[codes[offset] - 1]}: m/z={mass[offset]!r}, 强度={intensity[offset]!r}",
    )
