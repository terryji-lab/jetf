"""JET-Forest 与 FlashEntropySearch 精度与排序一致性评测模块。

涵盖：
1. 逐对谱相似度打分相关性评测 (Spectral Entropy vs Greedy Cosine 相关性，Pearson r, Spearman rho)
2. 全库检索 Top-K 候选交叠度与一致性评测 (Top-1 Agreement, Jaccard Index, Overlap Coefficient)
3. 窄窗检索 (Identity Search) 与全库开放检索 (Open Search) 下的命中一致性对比
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import scipy.stats

from jetf.benchmarks.dataset import BenchmarkDataset, sample_query_spectra
from jetf.benchmarks.flashentropy_adapter import (
    FlashEntropyBenchmarkEngine,
    check_flashentropy_available,
    score_flashentropy_pair,
)
from jetf.query import IonModePolicy, QueryConfig, SearchMode
from jetf.scoring import score_greedy_cosine
from jetf.search import search_forest
from jetf.types import SpectrumPeaks


@dataclass(frozen=True)
class EntropyCosineCorrelationResult:
    """光谱信息熵 (Spectral Entropy) 与贪心余弦 (Greedy Cosine) 逐对谱打分相关性评测结果。"""

    n_pairs: int
    ms2_tolerance_da: float
    pearson_r: float                # 线性相关系数
    spearman_rho: float             # 单调秩相关系数
    mean_entropy_score: float       # 光谱熵平均分
    mean_cosine_score: float        # 余弦平均分
    entropy_scores: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    cosine_scores: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class EntropyRetrievalAgreementResult:
    """全库检索场景下 JET-Forest (Cosine) 与 FlashEntropy (Entropy) 的候选一致性评测结果。"""

    mode_name: str
    n_queries: int
    k: int
    top1_agreement_rate: float      # Top-1 最优候选完全一致比例
    mean_jaccard: float             # Top-K 集合平均 Jaccard 相似度 (|A ∩ B| / |A ∪ B|)
    mean_overlap_coefficient: float # Top-K 平均交叠系数 (|A ∩ B| / min(|A|, |B|))
    details: list[dict] = field(default_factory=list)


def evaluate_entropy_cosine_correlation(
    pairs: Sequence[tuple[SpectrumPeaks, SpectrumPeaks]],
    ms2_tolerance_da: float = 0.02,
) -> EntropyCosineCorrelationResult:
    """计算成对谱在 FlashEntropy 与 JETF (Greedy Cosine) 打分下的数值相关性与分布。"""
    check_flashentropy_available()
    n = len(pairs)
    if n == 0:
        raise ValueError("pairs 列表不能为空")

    fe_scores = np.empty(n, dtype=np.float64)
    jetf_scores = np.empty(n, dtype=np.float64)

    for i, (p1, p2) in enumerate(pairs):
        fe_scores[i] = score_flashentropy_pair(p1, p2, ms2_tolerance_da=ms2_tolerance_da)
        jetf_scores[i] = score_greedy_cosine(p1, p2, tolerance_da=ms2_tolerance_da).score

    # 统计相关性 (处理常量/全零边界避免 NaN)
    std_fe = float(np.std(fe_scores))
    std_jetf = float(np.std(jetf_scores))
    if std_fe > 1e-12 and std_jetf > 1e-12:
        pr, _ = scipy.stats.pearsonr(fe_scores, jetf_scores)
        sr, _ = scipy.stats.spearmanr(fe_scores, jetf_scores)
    else:
        pr = 1.0 if np.allclose(fe_scores, jetf_scores) else 0.0
        sr = 1.0 if np.allclose(fe_scores, jetf_scores) else 0.0

    return EntropyCosineCorrelationResult(
        n_pairs=n,
        ms2_tolerance_da=ms2_tolerance_da,
        pearson_r=float(pr),
        spearman_rho=float(sr),
        mean_entropy_score=float(np.mean(fe_scores)),
        mean_cosine_score=float(np.mean(jetf_scores)),
        entropy_scores=fe_scores,
        cosine_scores=jetf_scores,
    )


def evaluate_entropy_retrieval_agreement(
    dataset: BenchmarkDataset,
    engine: FlashEntropyBenchmarkEngine,
    queries_info: Sequence[tuple[int, SpectrumPeaks]],
    k: int = 10,
    tolerance: float = 0.02,
    mode: str = "open",
    precursor_window_da: float = 0.02,
    exclude_self: bool = True,
) -> EntropyRetrievalAgreementResult:
    """评测 JET-Forest 与 FlashEntropy 在全库检索或窄窗检索场景下的 Top-K 候选交叠度与 Top-1 一致性。"""
    top1_matches = 0
    jaccard_sum = 0.0
    overlap_sum = 0.0
    details = []

    for row_idx, q_peaks in queries_info:
        meta = dataset.library.spectra[row_idx]
        pmz = float(meta.precursor_mz) if meta.precursor_mz is not None else None

        # 1. FlashEntropy 检索 (提取 k+1 个候选以防排除自身后不足 k 个)
        fe_res = engine.search_single(
            query_peaks=q_peaks,
            precursor_mz=pmz,
            top_k=k + 1 if exclude_self else k,
            mode=mode,
            ms1_tolerance_da=precursor_window_da,
            ms2_tolerance_da=tolerance,
        )
        if exclude_self:
            fe_top_indices = [idx for idx in fe_res.indices if idx != row_idx][:k]
        else:
            fe_top_indices = list(fe_res.indices[:k])

        # 2. JET-Forest 检索
        from jetf.query import PrecursorWindow
        p_win = None
        if mode == "identity" and pmz is not None:
            p_win = PrecursorWindow(center=pmz, tolerance_da=precursor_window_da)

        q_cfg = QueryConfig(
            mode=SearchMode.TOP_K,
            k=k + 1 if exclude_self else k,
            fragment_tolerance_da=tolerance,
            ion_mode=meta.ion_mode,
            precursor_window=p_win,
            ion_mode_policy=IonModePolicy.EXACT,
            exclude_spectrum_id=meta.external_id if exclude_self else None,
        )
        jetf_res = search_forest(q_peaks, dataset.forest, dataset.library, q_cfg)
        if exclude_self:
            jetf_top_indices = [h.spectrum_index for h in jetf_res.hits if h.spectrum_index != row_idx][:k]
        else:
            jetf_top_indices = [h.spectrum_index for h in jetf_res.hits[:k]]

        # 计算一致性指标
        top1_match = False
        if fe_top_indices and jetf_top_indices:
            top1_match = (fe_top_indices[0] == jetf_top_indices[0])
        elif not fe_top_indices and not jetf_top_indices:
            top1_match = True

        if top1_match:
            top1_matches += 1

        set_fe = set(fe_top_indices)
        set_jetf = set(jetf_top_indices)
        intersection = len(set_fe & set_jetf)
        union = len(set_fe | set_jetf)
        jaccard = float(intersection / union) if union > 0 else 1.0
        min_len = min(len(set_fe), len(set_jetf))
        overlap = float(intersection / min_len) if min_len > 0 else (1.0 if union == 0 else 0.0)

        jaccard_sum += jaccard
        overlap_sum += overlap

        details.append({
            "query_row": row_idx,
            "external_id": meta.external_id,
            "top1_match": top1_match,
            "jaccard": jaccard,
            "overlap_coefficient": overlap,
            "jetf_hits": jetf_top_indices,
            "fe_hits": fe_top_indices,
        })

    n_q = len(queries_info)
    mode_label = "全库开放检索 (Open Search)" if mode == "open" else f"窄前体窗检索 (Identity Search ±{precursor_window_da} Da)"
    return EntropyRetrievalAgreementResult(
        mode_name=mode_label,
        n_queries=n_q,
        k=k,
        top1_agreement_rate=float(top1_matches / n_q) if n_q > 0 else 0.0,
        mean_jaccard=float(jaccard_sum / n_q) if n_q > 0 else 0.0,
        mean_overlap_coefficient=float(overlap_sum / n_q) if n_q > 0 else 0.0,
        details=details,
    )
