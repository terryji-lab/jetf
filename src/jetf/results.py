"""检索结果模型、统计规范、有界堆与零分补足。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import heapq
import time
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from jetf.preprocessing import PreprocessedLibrary
from jetf.query import QueryConfig, SearchMode, ion_mode_passes
from jetf.scoring import SCORER_VERSIONED_ID
from jetf.structure import ION_MODES_BY_CODE
from jetf.types import INTERNAL_ID_DTYPE, MASS_DTYPE, check_column


@dataclass(frozen=True)
class SearchHit:
    """单条质谱命中记录。"""

    external_id: str
    spectrum_index: int
    score: float
    n_matched: int


def hit_ranking_key(hit: SearchHit) -> tuple[float, str, int]:
    """命中排序键：分数降序、外部 ID 升序、库内序号升序。"""
    return (-hit.score, hit.external_id, hit.spectrum_index)


@dataclass(frozen=True)
class _HeapItem:
    key: tuple[float, str, int]
    hit: SearchHit

    def __lt__(self, other: _HeapItem) -> bool:
        return self.key > other.key


class ResultSet:
    """检索结果收集容器。"""

    def __init__(self, config: QueryConfig) -> None:
        self._mode = config.mode
        self._k = config.k if config.k is not None else 0
        self._threshold = config.threshold if config.threshold is not None else 0.0
        self._heap: list[_HeapItem] = []
        self._hits: list[SearchHit] = []

    def theta(self) -> float:
        """动态门槛 theta：用于安全剪枝。"""
        if self._mode is not SearchMode.TOP_K:
            return self._threshold
        if self._k == 0 or len(self._heap) < self._k:
            return -np.inf
        return self._heap[0].hit.score

    def update(self, hit: SearchHit) -> None:
        """更新结果集。"""
        if self._mode is not SearchMode.TOP_K:
            if hit.score >= self._threshold:
                self._hits.append(hit)
            return
        item = _HeapItem(key=hit_ranking_key(hit), hit=hit)
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, item)
        elif self._heap and item.key < self._heap[0].key:
            heapq.heapreplace(self._heap, item)

    def accepts_zero_scores(self) -> bool:
        """判断是否可能接收 0 分记录。"""
        if self._mode is not SearchMode.TOP_K:
            return self._threshold == 0.0
        if self._k == 0:
            return False
        if len(self._heap) < self._k:
            return True
        return self._heap[0].hit.score == 0.0

    def finish(self) -> tuple[SearchHit, ...]:
        """按确定性排序规则输出命中列表。"""
        hits = (
            [item.hit for item in self._heap]
            if self._mode is SearchMode.TOP_K
            else list(self._hits)
        )
        hits.sort(key=hit_ranking_key)
        return tuple(hits)


@dataclass(frozen=True)
class SearchStats:
    """检索过程统计指标。"""

    nodes_visited: int
    n_scored: int
    pruned_by_layer: Mapping[str, int]
    bound_eval_time_ms: float = 0.0
    exact_eval_time_ms: float = 0.0
    seed_count: int = 0
    skipped_block_intervals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.pruned_by_layer, dict):
            object.__setattr__(
                self, "pruned_by_layer", MappingProxyType(dict(self.pruned_by_layer))
            )


@dataclass(frozen=True)
class SearchOutcome:
    """检索产出的最终对象。"""

    mode: SearchMode
    hits: tuple[SearchHit, ...]
    complete: bool
    stats: SearchStats
    versions: dict[str, str]


@dataclass
class EvalTimers:
    """计时累加器（毫秒）。"""

    bound_eval_ms: float = 0.0
    exact_eval_ms: float = 0.0


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def search_versions(library: PreprocessedLibrary, config: QueryConfig) -> dict[str, str]:
    return {
        "preprocess_version": library.spec.versioned_id,
        "scorer_version": SCORER_VERSIONED_ID,
        "snapshot_id": config.snapshot_id,
    }


@dataclass(frozen=True, eq=False)
class ZeroScoreCandidates:
    """零分候选谱元数据。"""

    member: NDArray[np.int64]
    ion_mode: NDArray[np.int8]
    precursor_mz: NDArray[np.float64]

    def __post_init__(self) -> None:
        check_column("member", self.member, INTERNAL_ID_DTYPE)
        check_column("ion_mode", self.ion_mode, np.dtype(np.int8))
        check_column("precursor_mz", self.precursor_mz, MASS_DTYPE)


def needs_zero_supplement(results: ResultSet, config: QueryConfig) -> bool:
    """判断当前结果集是否需要补足零分记录。"""
    if config.min_matched_peaks > 0:
        return False
    return results.accepts_zero_scores()


def eligible_mask(
    ion_mode: NDArray[np.int8],
    precursor_mz: NDArray[np.float64],
    config: QueryConfig,
) -> NDArray[np.bool_]:
    """批量元数据过滤掩码。"""
    accepted = [
        code
        for code, ion_mode_value in enumerate(ION_MODES_BY_CODE)
        if ion_mode_passes(config.ion_mode, config.ion_mode_policy, ion_mode_value)
    ]
    mask = np.isin(ion_mode, accepted)
    window = config.precursor_window
    if window is None:
        return mask
    lower = window.mz - window.tolerance_da
    upper = window.mz + window.tolerance_da
    return mask & (precursor_mz >= lower) & (precursor_mz <= upper)


def supplement_zero_score(
    results: ResultSet,
    library: PreprocessedLibrary,
    config: QueryConfig,
    candidates: ZeroScoreCandidates,
    scored: NDArray[np.bool_],
) -> int:
    """为满足条件的零能量/零匹配谱补入 0 分命中。"""
    mask = eligible_mask(candidates.ion_mode, candidates.precursor_mz, config)
    excluded = config.exclude_spectrum_id
    added = 0
    for raw in candidates.member[mask].tolist():
        row = int(raw)
        if scored[row]:
            continue
        ext_id = library.spectra[row].external_id
        if excluded is not None and ext_id == excluded:
            continue
        results.update(
            SearchHit(
                external_id=ext_id,
                spectrum_index=row,
                score=0.0,
                n_matched=0,
            )
        )
        added += 1
    return added
