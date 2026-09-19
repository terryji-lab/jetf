"""测试基准：分层 2000 谱子集的确定性抽样与物化。"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from jetf import INTERNAL_ID_DTYPE, ParsedLibrary

STRATA: tuple[tuple[int, int | None], ...] = ((0, 100), (100, 500), (500, 1000), (1000, None))
N_PER_STRATUM = 500
SUBSET_SIZE = N_PER_STRATUM * len(STRATA)
SUBSET_SALT = "jet-ms/ms1-subset"


def stratum_index(n_peaks: int) -> int:
    if n_peaks < 0:
        raise ValueError(f"峰数不能为负，得到 {n_peaks!r}")
    for index, (lower, upper) in enumerate(STRATA):
        if n_peaks >= lower and (upper is None or n_peaks < upper):
            return index
    raise ValueError(f"峰数 {n_peaks!r} 没有归属的层")


def subset_key(salt: str, external_id: str) -> bytes:
    return hashlib.sha256(f"{salt}:{external_id}".encode("utf-8")).digest()


def stratified_subset_indices(
    library: ParsedLibrary,
    *,
    n_per_stratum: int = N_PER_STRATUM,
    salt: str = SUBSET_SALT,
) -> NDArray[np.int64]:
    """从 library 抽分层子集的库内序号，返回升序的 int64 数组。"""
    counts = library.peak_counts()
    selected: list[int] = []
    for index, (lower, upper) in enumerate(STRATA):
        in_stratum = counts >= lower
        if upper is not None:
            in_stratum = in_stratum & (counts < upper)
        members = np.flatnonzero(in_stratum)
        if members.size < n_per_stratum:
            raise ValueError(
                f"第 {index} 层 [{lower}, {upper}) 只有 {members.size} 条谱，不足 {n_per_stratum} 条"
            )
        ranked = sorted((subset_key(salt, library.spectra[int(i)].external_id), int(i)) for i in members)
        selected.extend(index for _, index in ranked[:n_per_stratum])
    return np.sort(np.array(selected, dtype=INTERNAL_ID_DTYPE))


def subset_parsed_library(library: ParsedLibrary, indices: NDArray[np.int64]) -> ParsedLibrary:
    """按 indices 从 library 切出子库：元数据取子序列，峰缓冲按谱拼接。"""
    counts = library.peak_counts()
    selected_counts = counts[indices]
    keep_spectra = np.zeros(library.n_spectra, dtype=np.bool_)
    keep_spectra[indices] = True
    keep_peaks = np.repeat(keep_spectra, counts)
    offsets = np.concatenate(
        (np.zeros(1, dtype=INTERNAL_ID_DTYPE), np.cumsum(selected_counts, dtype=INTERNAL_ID_DTYPE))
    )
    return ParsedLibrary(
        source_path=library.source_path,
        spectra=tuple(library.spectra[int(i)] for i in indices),
        mass=library.mass[keep_peaks],
        intensity=library.intensity[keep_peaks],
        peak_id=library.peak_id[keep_peaks],
        spectrum_offsets=offsets,
    )
