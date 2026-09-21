"""Unit tests for U_ind Numba JIT acceleration and NumPy fallback equivalency."""

from __future__ import annotations

import numpy as np
import pytest

import jetf.scoring as scoring
from jetf.scoring import (
    _HAVE_NUMBA,
    _single_spectrum_bound_numba,
    _single_spectrum_bound_numpy,
    score_greedy_cosine,
    single_spectrum_bound,
)
from jetf.types import SpectrumPeaks


def _make_peaks(masses: list[float], intensities: list[float]) -> SpectrumPeaks:
    m = np.array(masses, dtype=np.float64)
    it = np.array(intensities, dtype=np.float64)
    norm = float(np.sqrt(np.sum(it * it)))
    v = it / norm if norm > 0 else it
    return SpectrumPeaks(
        mass=m,
        intensity=v,
        energy=v * v,
        peak_id=np.arange(len(m), dtype=np.int64),
        norm=norm,
    )


def test_empty_spectra():
    empty = _make_peaks([], [])
    p = _make_peaks([100.0, 200.0], [1.0, 1.0])

    assert single_spectrum_bound(empty, p, 0.02) == 0.0
    assert single_spectrum_bound(p, empty, 0.02) == 0.0
    assert single_spectrum_bound(empty, empty, 0.02) == 0.0

    assert _single_spectrum_bound_numpy(empty, p, 0.02) == 0.0
    assert _single_spectrum_bound_numpy(p, empty, 0.02) == 0.0


@pytest.mark.skipif(not _HAVE_NUMBA, reason="Numba not available")
def test_single_peak_matches():
    q = _make_peaks([250.0], [2.0])
    l_match = _make_peaks([250.01], [3.0])
    l_no_match = _make_peaks([250.1], [3.0])

    u_nb = _single_spectrum_bound_numba(q.mass, q.intensity, l_match.mass, l_match.intensity, 0.02)
    u_np = _single_spectrum_bound_numpy(q, l_match, 0.02)
    assert abs(u_nb - u_np) < 1e-12
    assert u_nb > 0.0

    u_no_nb = _single_spectrum_bound_numba(q.mass, q.intensity, l_no_match.mass, l_no_match.intensity, 0.02)
    u_no_np = _single_spectrum_bound_numpy(q, l_no_match, 0.02)
    assert u_no_nb == 0.0
    assert u_no_np == 0.0


@pytest.mark.skipif(not _HAVE_NUMBA, reason="Numba not available")
def test_overlapping_peaks_picks_max():
    # One query peak at 200.0 with tol=0.02 matches [199.98, 200.02]
    # Library has multiple peaks in window: 199.99 (int=1.0), 200.00 (int=5.0), 200.01 (int=2.0)
    q = _make_peaks([200.0], [1.0])
    lib = _make_peaks([199.99, 200.00, 200.01], [1.0, 5.0, 2.0])

    u_nb = _single_spectrum_bound_numba(q.mass, q.intensity, lib.mass, lib.intensity, 0.02)
    u_np = _single_spectrum_bound_numpy(q, lib, 0.02)

    assert abs(u_nb - u_np) < 1e-12
    # The picked intensity from library should be the max (which is 200.00 with normalized intensity)
    assert u_nb > 0.0


@pytest.mark.skipif(not _HAVE_NUMBA, reason="Numba not available")
def test_numba_vs_numpy_random_trials():
    rng = np.random.default_rng(12345)
    for _ in range(50):
        n_q = rng.integers(1, 80)
        n_l = rng.integers(1, 100)

        # Generate realistic peaks clustered in 50-1000 Da
        q_m = np.sort(rng.uniform(50.0, 1000.0, n_q))
        q_i = rng.uniform(0.01, 100.0, n_q)
        q_p = _make_peaks(q_m.tolist(), q_i.tolist())

        # Some library peaks aligned with query, some random
        l_m = np.sort(np.concatenate([q_m[: min(10, n_q)] + rng.uniform(-0.015, 0.015, min(10, n_q)), rng.uniform(50.0, 1000.0, n_l)]))
        l_i = rng.uniform(0.01, 100.0, len(l_m))
        l_p = _make_peaks(l_m.tolist(), l_i.tolist())

        tol = float(rng.choice([0.01, 0.02, 0.05, 0.1]))

        val_nb = _single_spectrum_bound_numba(q_p.mass, q_p.intensity, l_p.mass, l_p.intensity, tol)
        val_np = _single_spectrum_bound_numpy(q_p, l_p, tol)

        assert abs(val_nb - val_np) < 1e-12, f"Discrepancy: Numba={val_nb}, NumPy={val_np}"

        # U_ind must dominate score_greedy_cosine
        greedy_res = score_greedy_cosine(q_p, l_p, tol)
        assert val_nb >= greedy_res.score - 1e-12


def test_dispatch_fallback(monkeypatch):
    q = _make_peaks([100.0, 200.0], [1.0, 2.0])
    lib = _make_peaks([100.01, 200.01], [2.0, 1.0])

    val_normal = single_spectrum_bound(q, lib, 0.02)

    # Monkeypatch _HAVE_NUMBA to False
    monkeypatch.setattr(scoring, "_HAVE_NUMBA", False)
    val_fallback = single_spectrum_bound(q, lib, 0.02)

    assert abs(val_normal - val_fallback) < 1e-12
