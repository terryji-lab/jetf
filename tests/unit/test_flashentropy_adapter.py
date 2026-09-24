"""Unit tests for FlashEntropySearch adapter, benchmark engine, and accuracy evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from jetf.benchmarks.flashentropy_accuracy import (
    evaluate_entropy_cosine_correlation,
    evaluate_entropy_retrieval_agreement,
)
from jetf.benchmarks.flashentropy_adapter import (
    FlashEntropyBenchmarkEngine,
    check_flashentropy_available,
    flashentropy_to_jetf_peaks,
    jetf_library_to_flashentropy,
    jetf_peaks_to_flashentropy,
    score_flashentropy_pair,
)
from jetf.types import IonMode, SpectrumMeta, SpectrumPeaks


def _create_mock_peaks(
    masses: list[float], intensities: list[float]
) -> SpectrumPeaks:
    m = np.asarray(masses, dtype=np.float64)
    raw_i = np.asarray(intensities, dtype=np.float64)
    norm = float(np.linalg.norm(raw_i))
    norm_i = (raw_i / norm) if norm > 0 else np.zeros_like(raw_i)
    energy = norm_i**2
    peak_id = np.arange(len(m), dtype=np.int64)
    return SpectrumPeaks._create_unchecked(m, norm_i, energy, peak_id, norm)


def test_check_flashentropy_available():
    check_flashentropy_available()


def test_jetf_peaks_to_flashentropy_conversion():
    peaks = _create_mock_peaks([100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
    from jetf.types import SourceRef

    meta = SpectrumMeta(
        external_id="SPEC_001",
        precursor_mz=350.5,
        charge=1,
        ion_mode=IonMode.POSITIVE,
        source=SourceRef("test.mgf", 0),
    )
    d = jetf_peaks_to_flashentropy(peaks, meta=meta, use_raw_intensity=True)
    assert d["precursor_mz"] == pytest.approx(350.5)
    assert d["id"] == "SPEC_001"
    assert d["peaks"].shape == (3, 2)
    assert np.allclose(d["peaks"][:, 0], [100.0, 200.0, 300.0])
    assert np.allclose(d["peaks"][:, 1], [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
    assert float(np.sum(d["peaks"][:, 1])) == pytest.approx(1.0)


def test_flashentropy_to_jetf_peaks_roundtrip():
    arr = np.array([[100.0, 1.0], [200.0, 2.0]], dtype=np.float32)
    p = flashentropy_to_jetf_peaks(arr)
    assert len(p.mass) == 2
    assert p.norm > 0
    assert np.allclose(p.mass, [100.0, 200.0])

    # 边界检测：空数组与非有限值
    empty_p = flashentropy_to_jetf_peaks(np.empty((0, 2), dtype=np.float32))
    assert len(empty_p.mass) == 0
    assert empty_p.norm == 0.0


def test_score_flashentropy_pair():
    p1 = _create_mock_peaks([100.0, 200.0], [1.0, 2.0])
    p2 = _create_mock_peaks([100.0, 200.0], [1.0, 2.0])
    score = score_flashentropy_pair(p1, p2, ms2_tolerance_da=0.02)
    assert score == pytest.approx(1.0, abs=1e-3)

    # 不相交谱
    p3 = _create_mock_peaks([500.0, 600.0], [1.0, 2.0])
    score_disjoint = score_flashentropy_pair(p1, p3, ms2_tolerance_da=0.02)
    assert score_disjoint == pytest.approx(0.0, abs=1e-5)


def test_flashentropy_benchmark_engine_search():
    from jetf.mgf import ParsedLibrary
    from jetf.preprocessing import preprocess_library
    from jetf.types import SourceRef

    # 构造小型测试库
    mass_list = [100.0, 200.0, 150.0, 250.0, 100.0, 250.0]
    int_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    offsets = [0, 2, 4, 6]

    spectra = (
        SpectrumMeta("ID_A", 250.0, 1, IonMode.POSITIVE, SourceRef("test.mgf", 0)),
        SpectrumMeta("ID_B", 300.0, 1, IonMode.POSITIVE, SourceRef("test.mgf", 1)),
        SpectrumMeta("ID_C", 350.0, 1, IonMode.POSITIVE, SourceRef("test.mgf", 2)),
    )
    parsed = ParsedLibrary(
        source_path="test.mgf",
        spectra=spectra,
        mass=np.asarray(mass_list, dtype=np.float64),
        intensity=np.asarray(int_list, dtype=np.float64),
        peak_id=np.arange(6, dtype=np.int64),
        spectrum_offsets=np.asarray(offsets, dtype=np.int64),
    )
    lib = preprocess_library(parsed)

    engine = FlashEntropyBenchmarkEngine(lib, clean_spectra=False)
    assert engine.n_indexed == 3
    assert engine.build_time_s >= 0.0

    # 1. Open Search
    q = lib.peaks.spectrum_at(0)
    res = engine.search_single(q, top_k=2, mode="open")
    assert len(res.indices) == 2
    assert res.indices[0] == 0  # 最相似应当是自身
    assert res.scores[0] == pytest.approx(1.0, abs=1e-3)
    assert res.total_time_s >= 0.0

    # 2. Identity Search
    res_id = engine.search_single(
        q, precursor_mz=250.0, top_k=2, mode="identity", ms1_tolerance_da=0.02
    )
    assert len(res_id.indices) >= 1
    assert res_id.indices[0] == 0
    assert res_id.scores[0] == pytest.approx(1.0, abs=1e-3)

    # 3. Batch Search
    batch_res = engine.score_batch([q, lib.peaks.spectrum_at(1)], top_k=2)
    assert batch_res.n_queries == 2
    assert len(batch_res.indices) == 2
    assert batch_res.indices[0][0] == 0
    assert batch_res.indices[1][0] == 1


def test_evaluate_entropy_cosine_correlation():
    p1 = _create_mock_peaks([100.0, 200.0], [1.0, 2.0])
    p2 = _create_mock_peaks([100.0, 200.0], [1.0, 2.0])
    p3 = _create_mock_peaks([100.0, 250.0], [1.0, 2.0])

    pairs = [(p1, p2), (p1, p3)]
    corr = evaluate_entropy_cosine_correlation(pairs, ms2_tolerance_da=0.02)
    assert corr.n_pairs == 2
    assert corr.pearson_r >= 0.0
    assert corr.spearman_rho >= 0.0
