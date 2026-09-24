"""检索结果模型、统计规范、有界堆与零分补足。"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
import heapq
import time
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from jetf.preprocessing import PreprocessedLibrary
from jetf.query import QueryConfig, SearchMode, ion_mode_passes, is_eligible
from jetf.scoring import SCORER_VERSIONED_ID
from jetf.structure import ION_MODES_BY_CODE
from jetf.types import INTERNAL_ID_DTYPE, MASS_DTYPE, SpectrumMeta, check_column


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
    """检索结果收集容器。

    管理 Top-K 优先队列（有界小根堆）与 Threshold 模式命中收集列表，支持安全剪枝阈值计算与浮点容差管理。

    容差设计原则 (CPU vs GPU):
        - CPU 端：使用双精度 (FP64)，浮点精度高，`score_margin` 默认设为 1e-12（机器精度级别安全裕量）。
        - GPU 端：CUDA 核函数使用单精度 (FP32) 进行向量化贪婪余弦打分与树上界剪枝，浮点截断与累加舍入
          误差相对较大（典型误差范围 1e-6 ~ 2e-5）。GPU 端 `score_margin` 设为 5e-5（`DEFAULT_GPU_SCORE_MARGIN`）。

    数学保真与零漏检设计 (Zero False Dismissals):
        - `theta()` 返回 `threshold - score_margin` 作为 K1/K2/K3a 树上界安全剪枝的动态下界。
        - `update()` 在非 Top-K 模式下接纳 `score >= threshold - score_margin` 的命中记录。
          该机制保证了当理论真实分刚好达到或超过用户阈值（例如理论及格分 0.50），而 GPU FP32 计算
          由于下舍入得到 0.49997 时，候选谱绝不会被误丢弃，严格保证数学保真与零漏检。
        - 零分补足解耦原则：`accepts_zero_scores()` 仅由物理阈值 `threshold <= 0.0` 决定，绝不受
          `score_margin` 浮点容差扩大的影响，彻底避免了微小正阈值在 GPU 端误触发全库零分补足的缺陷。
    """

    def __init__(self, config: QueryConfig, score_margin: float = 1e-12) -> None:
        """初始化检索结果集。

        Args:
            config: 检索配置，指定模式 (TOP_K / THRESHOLD)、k 与 threshold 等。
            score_margin: 分数容差边距。CPU 端默认 1e-12 (FP64 机器精度)；
                GPU 端默认 5e-5 (DEFAULT_GPU_SCORE_MARGIN，用于覆盖 FP32 浮点累加容差)。
        """
        self._mode = config.mode
        self._k = config.k if config.k is not None else 0
        self._threshold = config.threshold if config.threshold is not None else 0.0
        self._score_margin = float(score_margin)
        self._heap: list[_HeapItem] = []
        self._hits: list[SearchHit] = []

    @property
    def score_margin(self) -> float:
        """分数容差边距（CPU 端 1e-12，GPU 端 5e-5）。"""
        return self._score_margin

    def theta(self) -> float:
        """动态门槛 theta：用于 K1/K2/K3a 安全剪枝。

        在非 TOP_K（如 THRESHOLD）模式下，返回 `threshold - score_margin` 作为安全剪枝下界；
        在 TOP_K 模式下，堆未满时返回 `-inf`，堆满时返回堆顶（第 k 大分数）。
        """
        if self._mode is not SearchMode.TOP_K:
            return self._threshold - self._score_margin
        if self._k == 0 or len(self._heap) < self._k:
            return -np.inf
        return self._heap[0].hit.score

    def update(self, hit: SearchHit) -> None:
        """更新结果集，接纳满足门槛条件的命中记录。

        在非 TOP_K 模式下，接收 `score >= threshold - score_margin` 的命中。
        这是严格保证数学保真与零漏检 (Zero False Dismissals) 的核心设计——防止 GPU FP32
        下舍入（如理论刚好及格的 0.50 因单精度累加被算为 0.49997）导致命中被误丢弃。
        在 TOP_K 模式下，按 `hit_ranking_key` 维持大小为 k 的有界堆。
        """
        if self._mode is not SearchMode.TOP_K:
            if hit.score >= self._threshold - self._score_margin:
                self._hits.append(hit)
            return
        item = _HeapItem(key=hit_ranking_key(hit), hit=hit)
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, item)
        elif self._heap and item.key < self._heap[0].key:
            heapq.heapreplace(self._heap, item)

    def accepts_zero_scores(self) -> bool:
        """判断是否可能接收 0 分记录，以决定是否需要补足零分记录。

        在非 TOP_K 模式下，严格仅当 `self._threshold <= 0.0` 时返回 True，与 `self._score_margin` 彻底解耦。

        【设计说明与防错注解】：
        此处绝不能使用 `self._threshold <= self._score_margin`！
        `score_margin` 是为了弥补 GPU FP32 计算累加与舍入误差（GPU 端设为 5e-5）而引入的打分/剪枝容差。
        若此处采用 `self._threshold <= self._score_margin`，当用户指定微小正阈值（如 `threshold = 1e-5`）时，
        在 GPU 端由于 `1e-5 <= 5e-5` 为 True，将错误判定为接收零分，导致 `needs_zero_supplement()` 返回 True，
        进而在 `supplement_zero_score` 中将全库所有零能量或零匹配谱全量补入结果集，违背了正阈值检索语义，
        造成严重的数据污染与性能灾难。
        因此，零分接纳逻辑必须严格基于物理阈值判定（`threshold <= 0.0`），不受浮点评分容差扩大影响。
        """
        if self._mode is not SearchMode.TOP_K:
            return self._threshold <= 0.0
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
    """检索过程统计指标。

    Attributes:
        nodes_visited: 访问并计算包络上界的树节点总数。
        n_scored: 实际执行细粒度打分（如 Greedy Cosine 精评）的谱总数。
        pruned_by_layer: 按分层阶段记录的剪枝计数映射。
        bound_eval_time_ms: 上界计算耗时（毫秒）。
        exact_eval_time_ms: 精评计算耗时（毫秒）。
        probe_scored: 记录在 Top-K 贪心探测 (probe) 阶段实际精评的谱数（该计数值已同时包含在 n_scored 中）。
    """

    nodes_visited: int
    n_scored: int
    pruned_by_layer: Mapping[str, int]
    bound_eval_time_ms: float = 0.0
    exact_eval_time_ms: float = 0.0
    probe_scored: int = 0

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


def search_versions(
    library: PreprocessedLibrary | None,
    config: QueryConfig,
    default_version: str = "correctness_v1/1",
) -> dict[str, str]:
    prep_ver = (
        library.spec.versioned_id
        if library is not None and hasattr(library, "spec")
        else default_version
    )
    return {
        "preprocess_version": prep_ver,
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
    config: QueryConfig,
) -> NDArray[np.bool_]:
    """批量元数据过滤掩码（离子模式）。"""
    accepted = [
        code
        for code, ion_mode_value in enumerate(ION_MODES_BY_CODE)
        if ion_mode_passes(config.ion_mode, config.ion_mode_policy, ion_mode_value)
    ]
    return np.isin(ion_mode, accepted)


def supplement_zero_score(
    results: ResultSet,
    library_or_spectra: PreprocessedLibrary | Sequence[SpectrumMeta],
    config: QueryConfig,
    candidates: ZeroScoreCandidates,
    scored: set[int] | NDArray[np.bool_] | Container[int],
) -> int:
    """为满足条件的零能量/零匹配谱补入 0 分命中。"""
    spectra = (
        library_or_spectra.spectra
        if hasattr(library_or_spectra, "spectra")
        else library_or_spectra
    )
    mask = eligible_mask(candidates.ion_mode, config)
    if config.precursor_window is not None:
        w_min = config.precursor_window.min_mz
        w_max = config.precursor_window.max_mz
        prec_mask = (candidates.precursor_mz >= w_min) & (candidates.precursor_mz <= w_max)
        mask = mask & prec_mask
    added = 0
    is_mask_arr = isinstance(scored, np.ndarray)
    for raw in candidates.member[mask].tolist():
        row = int(raw)
        if scored[row] if is_mask_arr else (row in scored):
            continue
        meta = spectra[row]
        if not is_eligible(config, meta):
            continue
        results.update(
            SearchHit(
                external_id=meta.external_id,
                spectrum_index=row,
                score=0.0,
                n_matched=0,
            )
        )
        added += 1
    return added
