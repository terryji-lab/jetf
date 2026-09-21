"""基准评测数据集与代表性查询采样工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from jetf.builder import build_forest_index
from jetf.cleaning import DEFAULT_CLEAN_CONFIG, MatchmsCleanConfig, clean_parsed_library
from jetf.mgf import ParsedLibrary, parse_mgf
from jetf.preprocessing import CORRECTNESS_V1, PreprocessSpec, PreprocessedLibrary, preprocess_library, preprocess_query
from jetf.structure import DEFAULT_FOREST_SPEC, ForestIndex, ForestSpec
from jetf.types import SpectrumPeaks


@dataclass(frozen=True)
class BenchmarkDataset:
    """已加载并构建好森林索引的基准数据集。"""

    parsed: ParsedLibrary
    library: PreprocessedLibrary
    forest: ForestIndex
    subset_indices: NDArray[np.int64] | None = None

    @property
    def n_spectra(self) -> int:
        return self.library.n_spectra


def resolve_default_mgf_path(explicit_path: str | Path | None = None) -> Path:
    """自动解析 MGF 库文件路径，优先查找指定路径，次之在仓库根目录查找。"""
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.is_file():
            raise FileNotFoundError(f"指定的 MGF 文件不存在: {p}")
        return p

    repo_root = Path(__file__).resolve().parents[3]
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file() or (p / "GNPS-LIBRARY.mgf").is_file():
            repo_root = p
            break

    candidates = [
        repo_root / "GNPS-LIBRARY.mgf",
        repo_root / "cleaned_spectra.mgf",
        repo_root / "ALL_GNPS.mgf",
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError("未在仓库根目录下找到默认 MGF 文件 (GNPS-LIBRARY.mgf)")


def slice_parsed_library(parsed: ParsedLibrary, indices: NDArray[np.int64]) -> ParsedLibrary:
    """按索引列表切分子库。"""
    # 保持唯一索引顺序，防止重复索引导致 offsets 与 peaks 长度不匹配
    idx_list = list(dict.fromkeys(int(i) for i in indices))

    mass_chunks = []
    intensity_chunks = []
    peak_id_chunks = []
    offsets = [0]
    spectra = []

    for idx in idx_list:
        spectra.append(parsed.spectra[idx])
        st = int(parsed.spectrum_offsets[idx])
        sp = int(parsed.spectrum_offsets[idx + 1])
        m = parsed.mass[st:sp]
        it = parsed.intensity[st:sp]
        pid = parsed.peak_id[st:sp]
        mass_chunks.append(m)
        intensity_chunks.append(it)
        peak_id_chunks.append(pid)
        offsets.append(offsets[-1] + (sp - st))

    cat_mass = np.concatenate(mass_chunks) if mass_chunks else np.empty(0, dtype=parsed.mass.dtype)
    cat_it = np.concatenate(intensity_chunks) if intensity_chunks else np.empty(0, dtype=parsed.intensity.dtype)
    cat_pid = np.concatenate(peak_id_chunks) if peak_id_chunks else np.empty(0, dtype=parsed.peak_id.dtype)

    return ParsedLibrary(
        source_path=parsed.source_path,
        spectra=tuple(spectra),
        mass=cat_mass,
        intensity=cat_it,
        peak_id=cat_pid,
        spectrum_offsets=np.array(offsets, dtype=np.int64),
        rejected=parsed.rejected,
    )


def sample_stratified_indices(
    parsed: ParsedLibrary,
    target_size: int = 2000,
    seed: int = 42,
) -> NDArray[np.int64]:
    """按谱峰数分层确定性抽样指定容量的谱序号。"""
    if target_size >= parsed.n_spectra:
        return np.arange(parsed.n_spectra, dtype=np.int64)

    rng = np.random.default_rng(seed)
    counts = parsed.peak_counts()

    # 四层分档：少峰 (0~50), 中少 (50~150), 中多 (150~500), 多峰 (500+)
    strata = [(0, 50), (50, 150), (150, 500), (500, None)]
    stratum_masks = []
    for lower, upper in strata:
        mask = counts >= lower
        if upper is not None:
            mask = mask & (counts < upper)
        stratum_masks.append(np.flatnonzero(mask))

    # 计算各层配额
    non_empty = [m for m in stratum_masks if len(m) > 0]
    base_quota = target_size // len(non_empty)
    selected: list[int] = []

    for members in stratum_masks:
        if len(members) == 0:
            continue
        take = min(len(members), base_quota)
        chosen = rng.choice(members, size=take, replace=False)
        selected.extend(chosen.tolist())

    # 差额补齐
    if len(selected) < target_size:
        rem_needed = target_size - len(selected)
        chosen_set = set(selected)
        all_remaining = [i for i in range(parsed.n_spectra) if i not in chosen_set]
        if all_remaining:
            extra = rng.choice(all_remaining, size=min(rem_needed, len(all_remaining)), replace=False)
            selected.extend(extra.tolist())

    return np.sort(np.array(selected[:target_size], dtype=np.int64))


def load_benchmark_dataset(
    mgf_path: str | Path | None = None,
    library_size: int | None = 2000,
    preprocess_spec: PreprocessSpec = CORRECTNESS_V1,
    forest_spec: ForestSpec = DEFAULT_FOREST_SPEC,
    clean_config: MatchmsCleanConfig | None = DEFAULT_CLEAN_CONFIG,
    seed: int = 42,
    max_records: int | None = None,
) -> BenchmarkDataset:
    """加载并预处理基准数据集，构建森林索引。默认执行 matchms 工业级清洗。"""
    resolved_path = resolve_default_mgf_path(mgf_path)
    file_size_mb = resolved_path.stat().st_size / (1024 * 1024)
    effective_max = max_records
    if effective_max is None and library_size is not None and file_size_mb > 100:
        effective_max = max(library_size * 5, 10000)

    parsed = parse_mgf(resolved_path, max_records=effective_max, clean_config=clean_config)

    if library_size is not None and library_size < parsed.n_spectra:
        indices = sample_stratified_indices(parsed, target_size=library_size, seed=seed)
        sub_parsed = slice_parsed_library(parsed, indices)
    else:
        indices = None
        sub_parsed = parsed

    library = preprocess_library(sub_parsed, preprocess_spec)
    forest = build_forest_index(library, forest_spec)
    return BenchmarkDataset(
        parsed=sub_parsed,
        library=library,
        forest=forest,
        subset_indices=indices,
    )


def sample_query_spectra(
    library: PreprocessedLibrary,
    n_queries: int = 50,
    seed: int = 2026,
) -> list[tuple[int, SpectrumPeaks]]:
    """从库中抽样代表性查询谱（包含各种峰数规模及正负离子模式）。

    返回: (原始行号, 查询峰 SpectrumPeaks) 列表。
    """
    rng = np.random.default_rng(seed)
    counts = np.diff(library.peaks.spectrum_offsets)

    # 优先抽取有有效前体质量的谱
    valid_rows = [
        r
        for r in range(library.n_spectra)
        if library.spectra[r].precursor_mz is not None and counts[r] >= 5
    ]
    if len(valid_rows) < n_queries:
        valid_rows = list(range(library.n_spectra))

    chosen_rows = rng.choice(valid_rows, size=min(n_queries, len(valid_rows)), replace=False)
    queries: list[tuple[int, SpectrumPeaks]] = []
    for r in chosen_rows:
        row = int(r)
        q = library.peaks.spectrum_at(row)
        queries.append((row, q))
    return queries


def sample_query_spectra_from_forest(
    forest: ForestIndex,
    n_queries: int = 50,
    seed: int = 2026,
) -> list[tuple[int, SpectrumPeaks]]:
    """从已构建的 ForestIndex 快照中直接抽样代表性查询谱（脱机工作，无需原始 MGF）。

    返回: (原始行号, 查询峰 SpectrumPeaks) 列表。
    """
    if forest.spectra is None:
        raise ValueError("传入的 ForestIndex 快照不含 spectra 元数据，无法执行查询谱抽样")

    rng = np.random.default_rng(seed)
    n_total = forest.n_spectra

    # 优先抽取已进入索引树 (internal_id >= 0) 且有有效前体质量的谱
    valid_rows = []
    for r in range(n_total):
        iid = int(forest.row_to_internal[r])
        if iid >= 0:
            meta = forest.spectra[r]
            if meta.precursor_mz is not None and np.isfinite(meta.precursor_mz):
                valid_rows.append(r)

    if len(valid_rows) < n_queries:
        valid_rows = [r for r in range(n_total) if int(forest.row_to_internal[r]) >= 0]

    chosen = rng.choice(valid_rows, size=min(n_queries, len(valid_rows)), replace=False)
    queries: list[tuple[int, SpectrumPeaks]] = []
    for r in chosen:
        row = int(r)
        iid = int(forest.row_to_internal[row])
        q = forest.postings.spectrum_at(iid)
        queries.append((row, q))
    return queries
