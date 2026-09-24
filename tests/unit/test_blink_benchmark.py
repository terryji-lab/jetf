"""BLINK 精度与吞吐评测模块单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from jetf.benchmarks.blink_accuracy import (
    evaluate_blink_confusion_matrix,
    evaluate_blink_pairwise_accuracy,
)
from jetf.benchmarks.blink_adapter import check_blink_available
from jetf.benchmarks.blink_throughput import extrapolate_blink_throughput
from jetf.types import SpectrumPeaks


@pytest.fixture(autouse=True)
def ensure_blink():
    check_blink_available()


def make_test_peaks(mz_list: list[float], int_list: list[float]) -> SpectrumPeaks:
    mass = np.array(mz_list, dtype=np.float64)
    raw_intensity = np.array(int_list, dtype=np.float64)
    norm = float(np.linalg.norm(raw_intensity))
    normalized = raw_intensity / norm if norm > 0 else np.zeros_like(raw_intensity)
    energy = normalized**2
    peak_id = np.arange(len(mass), dtype=np.int64)
    return SpectrumPeaks._create_unchecked(mass, normalized, energy, peak_id, norm)


def test_pairwise_accuracy_metrics():
    """测试逐对打分精度指标的正确计算。"""
    p1 = make_test_peaks([100.0, 200.0, 300.0], [10.0, 50.0, 100.0])
    p2 = make_test_peaks([100.005, 200.0, 300.002], [10.0, 50.0, 100.0])
    p3 = make_test_peaks([500.0, 600.0], [20.0, 40.0])

    pairs = [(p1, p2), (p1, p3), (p2, p3)]
    res = evaluate_blink_pairwise_accuracy(pairs, tolerance=0.01, bin_width=0.001)

    assert res.n_pairs == 3
    assert 0.0 <= res.mae <= 1.0
    assert 0.0 <= res.rmse <= 1.0
    assert -1.0 <= res.pearson_r <= 1.0
    assert -1.0 <= res.spearman_rho <= 1.0
    assert 0.0 <= res.discrepancy_rate <= 1.0
    assert 0.0 <= res.match_agreement_rate <= 1.0

    # JETF 与 matchms 应当完美一致 (MAE 极小，异动率为 0)
    assert res.jetf_mae < 1e-6
    assert res.jetf_discrepancy_rate == 0.0
    assert res.jetf_match_agreement_rate == 1.0


def test_confusion_matrix_metrics():
    """测试混淆矩阵计算与 Precision / Recall / F1。"""
    p1 = make_test_peaks([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0], [10.0] * 7)
    p2 = make_test_peaks([100.002, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0], [10.0] * 7)
    p3 = make_test_peaks([50.0, 60.0], [1.0, 1.0])

    pairs = [(p1, p2), (p1, p3), (p2, p3)]
    b_cm, j_cm = evaluate_blink_confusion_matrix(pairs, score_thresh=0.7, match_thresh=6)

    assert b_cm.tp + b_cm.fp + b_cm.fn + b_cm.tn == 3
    assert j_cm.tp + j_cm.fp + j_cm.fn + j_cm.tn == 3

    assert 0.0 <= b_cm.precision <= 1.0
    assert 0.0 <= b_cm.recall <= 1.0
    assert 0.0 <= b_cm.f1 <= 1.0
    assert 0.0 <= b_cm.accuracy <= 1.0

    assert 0.0 <= j_cm.precision <= 1.0
    assert 0.0 <= j_cm.recall <= 1.0
    assert 0.0 <= j_cm.f1 <= 1.0
    assert 0.0 <= j_cm.accuracy <= 1.0


def test_extrapolate_blink_throughput():
    """测试对数线性回归拟合与加速比外推。"""
    scales = [100, 500, 1000, 2000, 5000, 10000]
    # 模拟线性时延: T(ms) = 0.005 * N
    latencies = [0.005 * s for s in scales]

    res = extrapolate_blink_throughput(
        measured_scales=scales,
        measured_latencies_ms=latencies,
        target_scales=[50000, 100000, 2003310],
        jetf_2m_latency_ms=25.0,
    )

    assert res.r_squared > 0.99
    assert abs(res.beta - 1.0) < 0.05  # O(N) 线性阶数接近 1.0
    assert len(res.blink_extrapolated_latency_ms) == 3
    assert len(res.blink_extrapolated_qps) == 3

    # 验证 2M 外推时延约为 0.005 * 2003310 ≈ 10016 ms
    idx_2m = res.target_scales.index(2003310)
    blink_2m_lat = res.blink_extrapolated_latency_ms[idx_2m]
    assert 9000.0 < blink_2m_lat < 11000.0

    # 验证加速比
    assert res.jetf_speedup_at_2m is not None
    assert res.jetf_speedup_at_2m > 300.0


def test_extrapolate_blink_throughput_edge_cases():
    """测试外推模型在少于 2 点、非正数耗时及任意目标规模下的健壮性。"""
    # 1. 少于 2 点应当抛出 ValueError
    with pytest.raises(ValueError, match="至少需要 2 个"):
        extrapolate_blink_throughput([100], [5.0])

    # 2. 存在非正数耗时，过滤后若不足 2 点抛出 ValueError
    with pytest.raises(ValueError, match="至少需要 2 个"):
        extrapolate_blink_throughput([100, 200], [0.0, 5.0])

    # 3. 目标规模未显式包含 2003310 时，若传入 jetf_2m_latency_ms 仍能正确算出 2M 加速比
    scales = [100, 500, 1000]
    latencies = [0.01 * s for s in scales]
    res = extrapolate_blink_throughput(
        measured_scales=scales,
        measured_latencies_ms=latencies,
        target_scales=[50000, 100000],  # 不包含 2003310
        jetf_2m_latency_ms=20.0,
    )
    assert res.jetf_speedup_at_2m is not None
    # 理论 2M 时延约为 0.01 * 2003310 = 20033 ms，加速比约 1000x
    assert res.jetf_speedup_at_2m > 800.0


def test_pairwise_accuracy_with_constant_and_identical_scores():
    """测试逐对打分在包含恒定得分或零方差时的相关系数计算与 JETF 指标字段。"""
    p1 = make_test_peaks([100.0, 200.0], [1.0, 2.0])
    p2 = make_test_peaks([100.005, 200.0], [1.0, 2.0])

    res = evaluate_blink_pairwise_accuracy([(p1, p2)], tolerance=0.01, bin_width=0.001)
    # 单对样本，所有方差为 0，两者均等，相关系数应返回 1.0
    assert res.pearson_r == 1.0
    assert res.spearman_rho == 1.0
    assert res.jetf_pearson_r == 1.0
    assert res.jetf_spearman_rho == 1.0
