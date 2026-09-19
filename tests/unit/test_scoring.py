"""Unit tests for Greedy Cosine scoring and Uind bound."""

from __future__ import annotations

import numpy as np
import pytest

from jetf.scoring import score_greedy_cosine, single_spectrum_bound
from jetf.types import SpectrumPeaks


def _make_peaks(masses: list[float], intensities: list[float]) -> SpectrumPeaks:
    m = np.array(masses, dtype=np.float64)
    it = np.array(intensities, dtype=np.float64)
    # L2 normalize
    norm = float(np.sqrt(np.sum(it * it)))
    v = it / norm if norm > 0 else it
    return SpectrumPeaks(
        mass=m,
        intensity=v,
        energy=v * v,
        peak_id=np.arange(len(m), dtype=np.int64),
        norm=norm,
    )


def test_greedy_cosine_self_match():
    p = _make_peaks([100.0, 200.0, 300.0], [1.0, 2.0, 3.0])
    res = score_greedy_cosine(p, p, tolerance_da=0.02)
    assert pytest.approx(res.score, abs=1e-12) == 1.0
    assert res.n_matched == 3


def test_greedy_cosine_disjoint():
    p1 = _make_peaks([100.0, 200.0], [1.0, 1.0])
    p2 = _make_peaks([300.0, 400.0], [1.0, 1.0])
    res = score_greedy_cosine(p1, p2, tolerance_da=0.02)
    assert res.score == 0.0
    assert res.n_matched == 0


def test_single_spectrum_bound_dominates():
    p1 = _make_peaks([100.0, 200.0, 300.0], [1.0, 2.0, 3.0])
    p2 = _make_peaks([100.01, 199.99, 300.05], [2.0, 1.0, 4.0])

    res = score_greedy_cosine(p1, p2, tolerance_da=0.02)
    u_ind = single_spectrum_bound(p1, p2, tolerance_da=0.02)

    assert u_ind >= res.score - 1e-12
