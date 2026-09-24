"""JET-Forest 与 BLINK 精度对比评测模块。

涵盖：
1. 逐对打分数学精度评测 (Pairwise Accuracy Evaluation vs matchms Ground Truth)
2. 全库检索 Top-K 一致性与召回率评测 (Retrieval Jaccard & Recall@K)
3. 2x2 混淆矩阵与分类评测 (Confusion Matrix, Precision, Recall, F1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import scipy.stats

from jetf.benchmarks.adapter import (
    jetf_peaks_to_matchms,
)
from jetf.benchmarks.blink_adapter import (
    BlinkBenchmarkEngine,
    check_blink_available,
    score_blink_pair,
)
from jetf.benchmarks.dataset import BenchmarkDataset, sample_query_spectra
from jetf.query import IonModePolicy, QueryConfig, SearchMode
from jetf.scoring import score_greedy_cosine
from jetf.search import search_forest
from jetf.types import SpectrumPeaks


@dataclass(frozen=True)
class PairwiseAccuracyResult:
    """逐对谱打分精度评测结果。"""

    n_pairs: int
    tolerance_da: float
    bin_width: float
    mae: float                  # BLINK vs matchms MAE
    rmse: float                 # BLINK vs matchms RMSE
    pearson_r: float            # BLINK vs matchms Pearson 相关系数
    spearman_rho: float         # BLINK vs matchms Spearman 秩相关系数
    discrepancy_rate: float     # BLINK vs matchms 得分差异 > 0.001 比例
    match_agreement_rate: float # BLINK vs matchms 匹配峰数完全相等比例
    bias: float                 # BLINK vs matchms 均值偏差 (mean diff)
    jetf_mae: float             # JETF vs matchms MAE
    jetf_rmse: float            # JETF vs matchms RMSE
    jetf_discrepancy_rate: float# JETF vs matchms 差异 > 0.001 比例
    jetf_match_agreement_rate: float
    jetf_bias: float
    jetf_pearson_r: float = 1.0
    jetf_spearman_rho: float = 1.0
    # 附带打分数组供绘图
    mms_scores: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    blink_scores: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    jetf_scores: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class RetrievalAccuracyResult:
    """全库检索一致性评测结果。"""

    n_queries: int
    k: int
    mean_jaccard_jetf_blink: float   # JETF 与 BLINK Top-K Jaccard 相似度
    mean_recall_jetf_mms: float      # JETF 相对 matchms Ground Truth 的 Recall@K
    mean_recall_blink_mms: float     # BLINK 相对 matchms Ground Truth 的 Recall@K
    details: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ConfusionMatrixResult:
    """2x2 混淆矩阵与评估指标。"""

    method_name: str
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float


def _prepare_sqrt_peaks(peaks: SpectrumPeaks) -> SpectrumPeaks:
    """将峰强度预变换为平方根强度 (intensity_power=0.5) 并归一化，使得 JETF 算子与 matchms 严格对齐。"""
    if len(peaks.mass) == 0:
        return peaks
    raw_i = peaks.intensity * peaks.norm if peaks.norm > 0.0 else peaks.intensity
    sqrt_i = np.sqrt(np.maximum(0.0, raw_i))
    norm_sqrt = float(np.linalg.norm(sqrt_i))
    if norm_sqrt > 0.0:
        norm_i = sqrt_i / norm_sqrt
    else:
        norm_i = np.zeros_like(sqrt_i)
    return SpectrumPeaks._create_unchecked(
        peaks.mass, norm_i, norm_i**2, peaks.peak_id, 1.0
    )


def evaluate_blink_pairwise_accuracy(
    pairs: Sequence[tuple[SpectrumPeaks, SpectrumPeaks]],
    tolerance: float = 0.01,
    bin_width: float = 0.001,
) -> PairwiseAccuracyResult:
    """评测采样谱对在 JETF、BLINK 与 matchms (Ground Truth) 之间的一致性指标。"""
    check_blink_available()
    from matchms.similarity import CosineGreedy

    mms_scorer = CosineGreedy(tolerance=tolerance, mz_power=0.0, intensity_power=0.5)

    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("pairs 序列不能为空")

    mms_scores_list: list[float] = []
    blink_scores_list: list[float] = []
    jetf_scores_list: list[float] = []

    mms_matches_list: list[int] = []
    blink_matches_list: list[int] = []
    jetf_matches_list: list[int] = []

    for p1, p2 in pairs:
        # 1. matchms Ground Truth 打分
        s1 = jetf_peaks_to_matchms(p1, use_normalized_intensity=False)
        s2 = jetf_peaks_to_matchms(p2, use_normalized_intensity=False)
        mms_res = mms_scorer.pair(s1, s2)
        m_score = float(mms_res["score"])
        m_matches = int(round(float(mms_res["matches"])))
        mms_scores_list.append(m_score)
        mms_matches_list.append(m_matches)

        # 2. BLINK 打分
        b_score, b_matches = score_blink_pair(
            p1, p2, tolerance=tolerance, bin_width=bin_width, intensity_power=0.5
        )
        blink_scores_list.append(b_score)
        blink_matches_list.append(b_matches)

        # 3. JETF 打分 (使用平方根归一化以对齐 CosineGreedy intensity_power=0.5)
        p1_sqrt = _prepare_sqrt_peaks(p1)
        p2_sqrt = _prepare_sqrt_peaks(p2)
        j_res = score_greedy_cosine(p1_sqrt, p2_sqrt, tolerance_da=tolerance)
        jetf_scores_list.append(float(j_res.score))
        jetf_matches_list.append(int(j_res.n_matched))

    mms_arr = np.array(mms_scores_list, dtype=np.float64)
    blink_arr = np.array(blink_scores_list, dtype=np.float64)
    jetf_arr = np.array(jetf_scores_list, dtype=np.float64)

    mms_m = np.array(mms_matches_list, dtype=np.int64)
    blink_m = np.array(blink_matches_list, dtype=np.int64)
    jetf_m = np.array(jetf_matches_list, dtype=np.int64)

    # BLINK 指标
    blink_diff = blink_arr - mms_arr
    blink_mae = float(np.mean(np.abs(blink_diff)))
    blink_rmse = float(np.sqrt(np.mean(blink_diff**2)))
    blink_disc_rate = float(np.mean(np.abs(blink_diff) > 0.001))
    blink_match_rate = float(np.mean(blink_m == mms_m))
    blink_bias = float(np.mean(blink_diff))

    if np.all(blink_arr == mms_arr):
        blink_pearson = 1.0
        blink_spearman = 1.0
    elif np.std(blink_arr) < 1e-12 or np.std(mms_arr) < 1e-12:
        blink_pearson = 0.0
        blink_spearman = 0.0
    else:
        blink_pearson = float(scipy.stats.pearsonr(blink_arr, mms_arr)[0])
        blink_spearman = float(scipy.stats.spearmanr(blink_arr, mms_arr)[0])

    # JETF 指标
    jetf_diff = jetf_arr - mms_arr
    jetf_mae = float(np.mean(np.abs(jetf_diff)))
    jetf_rmse = float(np.sqrt(np.mean(jetf_diff**2)))
    jetf_disc_rate = float(np.mean(np.abs(jetf_diff) > 0.001))
    jetf_match_rate = float(np.mean(jetf_m == mms_m))
    jetf_bias = float(np.mean(jetf_diff))

    if np.all(jetf_arr == mms_arr):
        jetf_pearson = 1.0
        jetf_spearman = 1.0
    elif np.std(jetf_arr) < 1e-12 or np.std(mms_arr) < 1e-12:
        jetf_pearson = 0.0
        jetf_spearman = 0.0
    else:
        jetf_pearson = float(scipy.stats.pearsonr(jetf_arr, mms_arr)[0])
        jetf_spearman = float(scipy.stats.spearmanr(jetf_arr, mms_arr)[0])

    return PairwiseAccuracyResult(
        n_pairs=n_pairs,
        tolerance_da=tolerance,
        bin_width=bin_width,
        mae=blink_mae,
        rmse=blink_rmse,
        pearson_r=blink_pearson,
        spearman_rho=blink_spearman,
        discrepancy_rate=blink_disc_rate,
        match_agreement_rate=blink_match_rate,
        bias=blink_bias,
        jetf_mae=jetf_mae,
        jetf_rmse=jetf_rmse,
        jetf_discrepancy_rate=jetf_disc_rate,
        jetf_match_agreement_rate=jetf_match_rate,
        jetf_bias=jetf_bias,
        jetf_pearson_r=jetf_pearson,
        jetf_spearman_rho=jetf_spearman,
        mms_scores=mms_arr,
        blink_scores=blink_arr,
        jetf_scores=jetf_arr,
    )


def evaluate_blink_retrieval_accuracy(
    dataset: BenchmarkDataset,
    n_queries: int = 100,
    k: int = 10,
    tolerance: float = 0.01,
    bin_width: float = 0.001,
    exclude_self: bool = True,
) -> RetrievalAccuracyResult:
    """评测全库检索中 JETF 与 BLINK 相对 matchms 真实基准的召回率与 Top-K 重合度 (Jaccard)。"""
    check_blink_available()
    from matchms.similarity import CosineGreedy

    library = dataset.library
    n_lib = library.n_spectra
    effective_k = min(k, n_lib)

    # 预准备 BLINK 引擎
    engine = BlinkBenchmarkEngine(
        library, tolerance=tolerance, bin_width=bin_width, intensity_power=0.5
    )

    # 预准备 matchms 谱列表供基准打分
    from jetf.benchmarks.adapter import jetf_library_to_matchms

    # 为 JETF 准备与其自身预处理空间完全对齐的 Ground Truth (已做过 alpha 变换及归一化，故 intensity_power 固定为 1.0)
    mms_scorer_jetf = CosineGreedy(tolerance=tolerance, mz_power=0.0, intensity_power=1.0)
    mms_library_jetf = jetf_library_to_matchms(library, use_normalized_intensity=True)

    # 为 BLINK 准备其原生平方根强度空间完全对齐的 Ground Truth (intensity_power=0.5)
    mms_scorer_blink = CosineGreedy(tolerance=tolerance, mz_power=0.0, intensity_power=0.5)
    mms_library_blink = jetf_library_to_matchms(library, use_normalized_intensity=False)

    # 抽样查询谱
    sample_items = sample_query_spectra(library, n_queries=n_queries)
    if not sample_items:
        raise ValueError("无法从数据集中抽样到有效查询谱")

    jaccard_list: list[float] = []
    recall_jetf_list: list[float] = []
    recall_blink_list: list[float] = []
    details: list[dict] = []

    k_fetch = min(n_lib, effective_k + 1 if exclude_self else effective_k)

    for q_idx, (orig_row, q_peaks) in enumerate(sample_items):
        meta = library.spectra[orig_row] if (library.spectra and orig_row < len(library.spectra)) else None
        exclude_id = meta.external_id if (meta and meta.external_id is not None) else None

        query_config = QueryConfig(
            mode=SearchMode.TOP_K,
            k=k_fetch,
            fragment_tolerance_da=tolerance,
            min_matched_peaks=1,
            ion_mode_policy=IonModePolicy.ANY,
            exclude_spectrum_id=exclude_id if exclude_self else None,
        )

        # 1. JETF 检索与 Ground Truth 评测 (在 library 原生规格空间内严格对齐)
        s_q_jetf = jetf_peaks_to_matchms(q_peaks, use_normalized_intensity=True)
        mms_scores_jetf = np.array([
            float(mms_scorer_jetf.pair(s_q_jetf, s_lib)["score"]) for s_lib in mms_library_jetf
        ], dtype=np.float64)

        if exclude_self:
            mms_scores_jetf[orig_row] = -1.0

        if n_lib <= k_fetch:
            mms_topk_jetf = np.argsort(mms_scores_jetf)[::-1]
        else:
            part = np.argpartition(mms_scores_jetf, -k_fetch)[-k_fetch:]
            mms_topk_jetf = part[np.argsort(mms_scores_jetf[part])[::-1]]

        if exclude_self:
            mms_cand_jetf = [idx for idx in mms_topk_jetf if idx != orig_row]
        else:
            mms_cand_jetf = list(mms_topk_jetf)
        mms_set_jetf = set(int(idx) for idx in mms_cand_jetf[:effective_k])

        # JETF 检索
        j_outcome = search_forest(q_peaks, dataset.forest, config=query_config, library=library)
        if exclude_self:
            jetf_hits = [h.spectrum_index for h in j_outcome.hits if h.spectrum_index != orig_row]
        else:
            jetf_hits = [h.spectrum_index for h in j_outcome.hits]
        jetf_set = set(int(idx) for idx in jetf_hits[:effective_k])

        # 2. BLINK 检索与 Ground Truth 评测 (在 BLINK 原生 sqrt 强度空间内评测)
        s_q_blink = jetf_peaks_to_matchms(q_peaks, use_normalized_intensity=False)
        mms_scores_blink = np.array([
            float(mms_scorer_blink.pair(s_q_blink, s_lib)["score"]) for s_lib in mms_library_blink
        ], dtype=np.float64)

        if exclude_self:
            mms_scores_blink[orig_row] = -1.0

        if n_lib <= k_fetch:
            mms_topk_blink = np.argsort(mms_scores_blink)[::-1]
        else:
            part = np.argpartition(mms_scores_blink, -k_fetch)[-k_fetch:]
            mms_topk_blink = part[np.argsort(mms_scores_blink[part])[::-1]]

        if exclude_self:
            mms_cand_blink = [idx for idx in mms_topk_blink if idx != orig_row]
        else:
            mms_cand_blink = list(mms_topk_blink)
        mms_set_blink = set(int(idx) for idx in mms_cand_blink[:effective_k])

        # BLINK 检索
        b_res = engine.search_single(q_peaks, top_k=k_fetch)
        if exclude_self:
            b_hits = [idx for idx in b_res.indices if idx != orig_row]
        else:
            b_hits = list(b_res.indices)
        blink_set = set(int(idx) for idx in b_hits[:effective_k])

        # 指标计算: 仅在真实有效正样本集上评估 Recall@K
        gt_pos_jetf = set(int(idx) for idx in np.where(mms_scores_jetf > 0.0)[0])
        relevant_gt_jetf = mms_set_jetf & gt_pos_jetf
        rec_jetf = (
            len(jetf_set & relevant_gt_jetf) / len(relevant_gt_jetf)
            if relevant_gt_jetf
            else 1.0
        )

        gt_pos_blink = set(int(idx) for idx in np.where(mms_scores_blink > 0.0)[0])
        relevant_gt_blink = mms_set_blink & gt_pos_blink
        rec_blink = (
            len(blink_set & relevant_gt_blink) / len(relevant_gt_blink)
            if relevant_gt_blink
            else 1.0
        )

        recall_jetf_list.append(rec_jetf)
        recall_blink_list.append(rec_blink)

        # Top-K Jaccard (JETF vs BLINK): 考虑有效检索集合，避免 0 分随机排列造成的噪声
        b_pos_hits = [
            int(idx) for idx, sc in zip(b_res.indices, b_res.scores)
            if (not exclude_self or idx != orig_row) and sc > 0.0
        ][:effective_k]
        b_pos_set = set(b_pos_hits)

        if jetf_set or b_pos_set:
            union_jb = jetf_set | b_pos_set
            inter_jb = jetf_set & b_pos_set
            jaccard = len(inter_jb) / len(union_jb) if union_jb else 1.0
        else:
            jaccard = 1.0
        jaccard_list.append(jaccard)

        details.append({
            "query_index": orig_row,
            "jaccard_jetf_blink": jaccard,
            "recall_jetf": rec_jetf,
            "recall_blink": rec_blink,
        })

    return RetrievalAccuracyResult(
        n_queries=len(sample_items),
        k=effective_k,
        mean_jaccard_jetf_blink=float(np.mean(jaccard_list)),
        mean_recall_jetf_mms=float(np.mean(recall_jetf_list)),
        mean_recall_blink_mms=float(np.mean(recall_blink_list)),
        details=details,
    )


def evaluate_blink_confusion_matrix(
    pairs: Sequence[tuple[SpectrumPeaks, SpectrumPeaks]],
    score_thresh: float = 0.7,
    match_thresh: int = 6,
    tolerance: float = 0.01,
    bin_width: float = 0.001,
) -> tuple[ConfusionMatrixResult, ConfusionMatrixResult]:
    """生成 2x2 混淆矩阵 (Similar vs Dissimilar) 对比 BLINK 与 JETF 相对 matchms 真实基准的分类指标。

    定义:
    - 相似 (Similar / Positive): score >= score_thresh 且 matches >= match_thresh
    - 不相似 (Dissimilar / Negative): 否则
    返回: (blink_cm, jetf_cm)
    """
    check_blink_available()
    from matchms.similarity import CosineGreedy

    mms_scorer = CosineGreedy(tolerance=tolerance, mz_power=0.0, intensity_power=0.5)

    gt_positive: list[bool] = []
    blink_positive: list[bool] = []
    jetf_positive: list[bool] = []

    for p1, p2 in pairs:
        # 1. matchms Ground Truth
        s1 = jetf_peaks_to_matchms(p1, use_normalized_intensity=False)
        s2 = jetf_peaks_to_matchms(p2, use_normalized_intensity=False)
        mms_res = mms_scorer.pair(s1, s2)
        m_s = float(mms_res["score"])
        m_c = int(round(float(mms_res["matches"])))
        gt_sim = (m_s >= score_thresh) and (m_c >= match_thresh)
        gt_positive.append(gt_sim)

        # 2. BLINK
        b_s, b_c = score_blink_pair(p1, p2, tolerance=tolerance, bin_width=bin_width)
        blink_sim = (b_s >= score_thresh) and (b_c >= match_thresh)
        blink_positive.append(blink_sim)

        # 3. JETF
        p1_sqrt = _prepare_sqrt_peaks(p1)
        p2_sqrt = _prepare_sqrt_peaks(p2)
        j_res = score_greedy_cosine(p1_sqrt, p2_sqrt, tolerance_da=tolerance)
        jetf_sim = (j_res.score >= score_thresh) and (j_res.n_matched >= match_thresh)
        jetf_positive.append(jetf_sim)

    gt_arr = np.array(gt_positive, dtype=bool)
    b_arr = np.array(blink_positive, dtype=bool)
    j_arr = np.array(jetf_positive, dtype=bool)

    def _calc_cm(method_name: str, pred: NDArray[np.bool_]) -> ConfusionMatrixResult:
        tp = int(np.sum(pred & gt_arr))
        fp = int(np.sum(pred & ~gt_arr))
        fn = int(np.sum(~pred & gt_arr))
        tn = int(np.sum(~pred & ~gt_arr))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else 1.0

        return ConfusionMatrixResult(
            method_name=method_name,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=prec,
            recall=rec,
            f1=f1,
            accuracy=acc,
        )

    blink_cm = _calc_cm("BLINK", b_arr)
    jetf_cm = _calc_cm("JET-Forest", j_arr)
    return blink_cm, jetf_cm
