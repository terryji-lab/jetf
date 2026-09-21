"""Unit tests verifying mathematical consistency between JETF scoring and matchms CosineGreedy."""

from __future__ import annotations

import numpy as np
import pytest

from jetf.benchmarks.adapter import jetf_peaks_to_matchms
from jetf.scoring import score_greedy_cosine
from jetf.types import SpectrumPeaks

try:
    import matchms
    from matchms.similarity import CosineGreedy

    HAS_MATCHMS = True
except ImportError:
    HAS_MATCHMS = False

pytestmark = pytest.mark.skipif(not HAS_MATCHMS, reason="matchms 未安装，跳过对比单元测试")


def _make_spectrum_peaks(masses: list[float], intensities: list[float]) -> SpectrumPeaks:
    m = np.array(masses, dtype=np.float64)
    it = np.array(intensities, dtype=np.float64)
    l2_sum = float(np.sum(it * it))
    norm = float(np.sqrt(l2_sum)) if l2_sum > 0 else 1.0
    v = it / norm if l2_sum > 0 else it
    return SpectrumPeaks(
        mass=m,
        intensity=v,
        energy=v * v,
        peak_id=np.arange(len(m), dtype=np.int64),
        norm=norm,
    )


def test_matchms_self_match_exact():
    """验证自匹配时 JETF 与 matchms 得分均为 1.0 且匹配峰数一致。"""
    p = _make_spectrum_peaks([100.0, 200.0, 300.0, 400.0], [10.0, 20.0, 5.0, 50.0])
    s = jetf_peaks_to_matchms(p)

    jetf_res = score_greedy_cosine(p, p, tolerance_da=0.02)
    mms_scorer = CosineGreedy(tolerance=0.02, mz_power=0.0, intensity_power=1.0)
    mms_res = mms_scorer.pair(s, s)

    assert pytest.approx(jetf_res.score, abs=1e-10) == 1.0
    assert pytest.approx(float(mms_res["score"]), abs=1e-10) == 1.0
    assert jetf_res.n_matched == int(mms_res["matches"]) == 4


def test_matchms_disjoint_match():
    """验证互不相交谱打分为 0.0。"""
    p1 = _make_spectrum_peaks([100.0, 200.0], [1.0, 2.0])
    p2 = _make_spectrum_peaks([300.0, 400.0], [1.0, 2.0])

    s1 = jetf_peaks_to_matchms(p1)
    s2 = jetf_peaks_to_matchms(p2)

    jetf_res = score_greedy_cosine(p1, p2, tolerance_da=0.02)
    mms_scorer = CosineGreedy(tolerance=0.02, mz_power=0.0, intensity_power=1.0)
    mms_res = mms_scorer.pair(s1, s2)

    assert jetf_res.score == 0.0
    assert float(mms_res["score"]) == 0.0
    assert jetf_res.n_matched == int(mms_res["matches"]) == 0


def test_matchms_partial_overlap_numerical_equivalence():
    """验证部分重叠谱对在微观打分上的严格数值等价性 (误差 < 1e-10)。"""
    rng = np.random.default_rng(42)
    # 随机生成 50 对具有部分重叠与噪声的谱对
    mms_scorer = CosineGreedy(tolerance=0.02, mz_power=0.0, intensity_power=1.0)

    for _ in range(50):
        n_common = rng.integers(5, 20)
        n_unique1 = rng.integers(5, 20)
        n_unique2 = rng.integers(5, 20)

        common_m = np.sort(rng.uniform(100.0, 800.0, size=n_common))
        # 加入微小 m/z 扰动 (在 0.02 Da 容差内)
        m1 = np.sort(
            np.concatenate([common_m, rng.uniform(100.0, 800.0, size=n_unique1)])
        )
        m2 = np.sort(
            np.concatenate(
                [
                    common_m + rng.uniform(-0.015, 0.015, size=n_common),
                    rng.uniform(100.0, 800.0, size=n_unique2),
                ]
            )
        )

        it1 = rng.exponential(scale=10.0, size=len(m1)) + 1.0
        it2 = rng.exponential(scale=10.0, size=len(m2)) + 1.0

        p1 = _make_spectrum_peaks(m1.tolist(), it1.tolist())
        p2 = _make_spectrum_peaks(m2.tolist(), it2.tolist())

        s1 = jetf_peaks_to_matchms(p1)
        s2 = jetf_peaks_to_matchms(p2)

        jetf_res = score_greedy_cosine(p1, p2, tolerance_da=0.02)
        mms_res = mms_scorer.pair(s1, s2)

        score_diff = abs(jetf_res.score - float(mms_res["score"]))
        assert score_diff < 1e-10, (
            f"JETF ({jetf_res.score}) 与 matchms ({mms_res['score']}) 分差过大: {score_diff}"
        )
        assert jetf_res.n_matched == int(mms_res["matches"])
