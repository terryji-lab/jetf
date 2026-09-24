"""JET-Forest 与 matchms 结果一致性评测模块。

涵盖：
1. 逐对谱打分数学等价性 (Pairwise Scoring Equivalence)
2. 全库检索排序与召回一致性 (Retrieval Ranking & Recall Equivalence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from jetf.benchmarks.adapter import check_matchms_available, jetf_peaks_to_matchms
from jetf.benchmarks.dataset import BenchmarkDataset
from jetf.query import IonModePolicy, QueryConfig, SearchMode, is_eligible
from jetf.scoring import score_greedy_cosine
from jetf.search import search_forest
from jetf.types import DEFAULT_FRAGMENT_TOLERANCE_DA, IonMode, SpectrumPeaks


@dataclass(frozen=True)
class PairwiseConsistencyResult:
    """逐对谱打分一致性评测结果。"""

    n_pairs: int
    tolerance_da: float
    max_absolute_error: float
    mean_absolute_error: float
    rmse: float
    matched_peak_match_rate: float
    p50_error: float
    p95_error: float
    p99_error: float
    discrepant_pairs_count: int  # 绝对误差超过 1e-6 的异常数


@dataclass(frozen=True)
class QueryRetrievalConsistency:
    """单条查询在特定检索模式下的检索一致性。"""

    query_index: int
    mode_name: str
    jetf_hit_count: int
    matchms_hit_count: int
    recall_at_k: float
    zero_false_dismissals: bool
    max_score_diff: float
    rank_consistent: bool


@dataclass(frozen=True)
class RetrievalConsistencySummary:
    """全量查询检索一致性汇总指标。"""

    n_queries: int
    mode_name: str
    mean_recall_at_k: float
    all_zero_false_dismissals: bool
    total_false_dismissals: int
    max_score_discrepancy: float
    details: tuple[QueryRetrievalConsistency, ...] = field(default_factory=tuple)


def evaluate_pairwise_consistency(
    pairs: Sequence[tuple[SpectrumPeaks, SpectrumPeaks]],
    tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA,
) -> PairwiseConsistencyResult:
    """评测单对谱打分在 JETF 与 matchms 间的一致性。"""
    check_matchms_available()
    from matchms.similarity import CosineGreedy

    scorer = CosineGreedy(tolerance=tolerance_da, mz_power=0.0, intensity_power=1.0)

    score_diffs: list[float] = []
    matched_equal_count = 0
    discrepant_count = 0

    for p1, p2 in pairs:
        # JETF 打分
        jetf_res = score_greedy_cosine(p1, p2, tolerance_da=tolerance_da)

        # matchms 打分
        s1 = jetf_peaks_to_matchms(p1)
        s2 = jetf_peaks_to_matchms(p2)
        matchms_res = scorer.pair(s1, s2)
        matchms_score = float(matchms_res["score"])
        matchms_matches = int(matchms_res["matches"])

        diff = abs(jetf_res.score - matchms_score)
        score_diffs.append(diff)
        if diff > 1e-6:
            discrepant_count += 1

        if jetf_res.n_matched == matchms_matches:
            matched_equal_count += 1

    diffs_arr = np.array(score_diffs, dtype=np.float64)
    n_pairs = len(pairs)

    return PairwiseConsistencyResult(
        n_pairs=n_pairs,
        tolerance_da=tolerance_da,
        max_absolute_error=float(np.max(diffs_arr)) if n_pairs > 0 else 0.0,
        mean_absolute_error=float(np.mean(diffs_arr)) if n_pairs > 0 else 0.0,
        rmse=float(math.sqrt(float(np.mean(diffs_arr**2)))) if n_pairs > 0 else 0.0,
        matched_peak_match_rate=(matched_equal_count / n_pairs) if n_pairs > 0 else 1.0,
        p50_error=float(np.percentile(diffs_arr, 50)) if n_pairs > 0 else 0.0,
        p95_error=float(np.percentile(diffs_arr, 95)) if n_pairs > 0 else 0.0,
        p99_error=float(np.percentile(diffs_arr, 99)) if n_pairs > 0 else 0.0,
        discrepant_pairs_count=discrepant_count,
    )


def evaluate_retrieval_consistency(
    dataset: BenchmarkDataset,
    queries: Sequence[tuple[int, SpectrumPeaks]] | Sequence[tuple[int, SpectrumPeaks, QueryConfig]],
    config: QueryConfig | Sequence[QueryConfig] | None = None,
    mode_name: str = "custom",
    use_gpu: bool = False,
    gpu_forest: Any | None = None,
) -> RetrievalConsistencySummary:
    """评测在指定检索配置下，JET-Forest (CPU 或 GPU) 索引检出结果与 matchms 全量穷举打分的一致性与零漏检。"""
    check_matchms_available()
    import warnings
    from matchms.similarity import CosineGreedy

    scorers: dict[float, CosineGreedy] = {}

    def _get_scorer(tau: float) -> CosineGreedy:
        if tau not in scorers:
            scorers[tau] = CosineGreedy(tolerance=tau, mz_power=0.0, intensity_power=1.0)
        return scorers[tau]

    # 预先将库谱转换为 matchms 谱列表以加速基准测试
    library = dataset.library
    matchms_lib = [
        jetf_peaks_to_matchms(library.peaks.spectrum_at(r), meta=library.spectra[r])
        for r in range(library.n_spectra)
    ]

    details: list[QueryRetrievalConsistency] = []
    total_false_dismissals = 0
    max_score_diff = 0.0

    has_unknown_partition = any(p.ion_mode == IonMode.UNKNOWN for p in dataset.forest.partitions)

    # 预先解析所有查询对象与配置
    parsed_items: list[tuple[int, SpectrumPeaks, QueryConfig]] = []
    for idx, item in enumerate(queries):
        if len(item) == 3:
            query_row, q_peaks, q_cfg = item  # type: ignore[misc]
        elif isinstance(config, Sequence):
            query_row, q_peaks = item[:2]
            q_cfg = config[idx]
        elif config is not None:
            query_row, q_peaks = item[:2]
            q_cfg = config
        else:
            raise ValueError("未为查询提供有效的 QueryConfig")
        parsed_items.append((query_row, q_peaks, q_cfg))

    # 若启用 GPU 模式，在批处理流水线中一次性检索所有查询谱
    gpu_outcomes = None
    if use_gpu:
        from jetf.gpu import GpuForestIndex, require_cuda, search_forest_batch_gpu
        require_cuda()
        if gpu_forest is None:
            gpu_forest = GpuForestIndex.from_forest(dataset.forest)
        q_peaks_all = [p[1] for p in parsed_items]
        q_cfgs_all = [p[2] for p in parsed_items]
        gpu_outcomes = search_forest_batch_gpu(
            queries=q_peaks_all,
            gpu_forest=gpu_forest,
            library=library,
            config=q_cfgs_all,
            batch_size=min(128, len(q_peaks_all)),
            uind=True,
        )

    for idx, (query_row, q_peaks, q_cfg) in enumerate(parsed_items):
        if (
            q_cfg.ion_mode == IonMode.UNKNOWN
            and q_cfg.ion_mode_policy == IonModePolicy.INCLUDE_UNKNOWN
            and not has_unknown_partition
        ):
            warnings.warn(
                f"查询 {query_row} 使用了默认 UNKNOWN 离子模式，而库中无 UNKNOWN 分区，将检索不到任何结果。",
                UserWarning,
                stacklevel=2,
            )

        scorer = _get_scorer(q_cfg.fragment_tolerance_da)

        # 1. JET-Forest 检索执行 (GPU 或 CPU)
        if use_gpu and gpu_outcomes is not None:
            jetf_outcome = gpu_outcomes[idx]
        else:
            jetf_outcome = search_forest(q_peaks, dataset.forest, library, q_cfg)
        jetf_hits = jetf_outcome.hits

        # 2. matchms 遍历打分基准 (Ground Truth)
        q_matchms = jetf_peaks_to_matchms(q_peaks)
        gt_candidates: list[tuple[float, int, str, int]] = []  # (score, matches, external_id, row)

        for row, meta in enumerate(library.spectra):
            if not is_eligible(q_cfg, meta):
                continue

            lib_s = matchms_lib[row]
            score_item = scorer.pair(q_matchms, lib_s)
            score_val = float(score_item["score"])
            n_m = int(score_item["matches"])

            if n_m >= q_cfg.min_matched_peaks:
                gt_candidates.append((score_val, n_m, meta.external_id, row))

        # 按 matchms 得分降序排序，平局按 external_id 升序、最后按 row 升序 (与 JETF hit_ranking_key 一致)
        gt_candidates.sort(key=lambda x: (-x[0], x[2], x[3]))

        # 根据模式切分期望命中的集合
        if q_cfg.mode == SearchMode.TOP_K:
            k = q_cfg.k or 10
            expected_hits = gt_candidates[:k]
        else:
            thresh = q_cfg.threshold or 0.0
            expected_hits = [c for c in gt_candidates if c[0] >= thresh]

        # 3. 对比召回与一致性
        jetf_hit_ids = [h.external_id for h in jetf_hits]
        expected_hit_ids = [c[2] for c in expected_hits]

        # 召回率计算 (修复 E2: 期望为空但有检出时召回应为 0.0)
        if not expected_hit_ids:
            recall = 1.0 if not jetf_hit_ids else 0.0
        else:
            intersection = set(jetf_hit_ids).intersection(set(expected_hit_ids))
            recall = len(intersection) / len(expected_hit_ids)

        # 零漏检判定 (Zero False Dismissals):
        # 如果任何一个 matchms 真实候选，其分数显著高于已返回的 JETF 截止分数或检索阈值，却未被检出，则记为漏检
        min_jetf_score = jetf_hits[-1].score if jetf_hits else 0.0
        false_dismissal = False
        for score_val, n_m, ext_id, row in expected_hits:
            if ext_id not in jetf_hit_ids and score_val > min_jetf_score + 1e-7:
                false_dismissal = True
                total_false_dismissals += 1
                break

        # 分数逐项对比 (修复 E3: 按 external_id 对齐，避免错位虚假误差)
        query_max_diff = 0.0
        jetf_score_map = {h.external_id: h.score for h in jetf_hits}
        for score_val, _, ext_id, _ in expected_hits:
            if ext_id in jetf_score_map:
                d = abs(jetf_score_map[ext_id] - score_val)
                if d > query_max_diff:
                    query_max_diff = d
        if query_max_diff > max_score_diff:
            max_score_diff = query_max_diff

        # 排序完全一致判定 (修复 E4: 比较完整列表，不截短)
        rank_match = (jetf_hit_ids == expected_hit_ids)

        details.append(
            QueryRetrievalConsistency(
                query_index=query_row,
                mode_name=mode_name,
                jetf_hit_count=len(jetf_hits),
                matchms_hit_count=len(expected_hits),
                recall_at_k=recall,
                zero_false_dismissals=not false_dismissal,
                max_score_diff=query_max_diff,
                rank_consistent=rank_match,
            )
        )

    mean_recall = float(np.mean([d.recall_at_k for d in details])) if details else 1.0

    return RetrievalConsistencySummary(
        n_queries=len(queries),
        mode_name=mode_name,
        mean_recall_at_k=mean_recall,
        all_zero_false_dismissals=(total_false_dismissals == 0),
        total_false_dismissals=total_false_dismissals,
        max_score_discrepancy=max_score_diff,
        details=tuple(details),
    )
