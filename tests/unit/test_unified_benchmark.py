"""Unit tests for MultiEngineBenchmarkRunner and Unified Benchmark Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from jetf.benchmarks.dataset import BenchmarkDataset
from jetf.benchmarks.unified_runner import (
    MultiEngineBenchmarkRunner,
    UnifiedBenchmarkReport,
    detect_available_engines,
)
from jetf.builder import build_forest_index
from jetf.mgf import ParsedLibrary
from jetf.preprocessing import preprocess_library
from jetf.structure import DEFAULT_FOREST_SPEC
from jetf.types import IonMode, SourceRef, SpectrumMeta


@pytest.fixture
def mini_dataset() -> BenchmarkDataset:
    """构造微型合成质谱库数据集 (20 条谱)。"""
    rng = np.random.default_rng(42)
    n_spectra = 20
    spectra = []
    masses = []
    intensities = []
    offsets = [0]

    for i in range(n_spectra):
        n_p = rng.integers(5, 15)
        m = np.sort(rng.uniform(50.0, 500.0, size=n_p))
        raw_it = rng.uniform(1.0, 100.0, size=n_p)
        masses.extend(m)
        intensities.extend(raw_it)
        offsets.append(offsets[-1] + n_p)
        pmz = float(np.max(m) + rng.uniform(10.0, 50.0))
        meta = SpectrumMeta(
            external_id=f"SYNTH_{i:03d}",
            precursor_mz=pmz,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("synth.mgf", i),
        )
        spectra.append(meta)

    mass_arr = np.asarray(masses, dtype=np.float64)
    int_arr = np.asarray(intensities, dtype=np.float64)
    pid_arr = np.arange(len(mass_arr), dtype=np.int64)
    off_arr = np.asarray(offsets, dtype=np.int64)

    parsed = ParsedLibrary(
        source_path="synth.mgf",
        spectra=tuple(spectra),
        mass=mass_arr,
        intensity=int_arr,
        peak_id=pid_arr,
        spectrum_offsets=off_arr,
    )
    lib = preprocess_library(parsed)
    forest = build_forest_index(lib, DEFAULT_FOREST_SPEC)
    return BenchmarkDataset(parsed=parsed, library=lib, forest=forest)


def test_detect_available_engines():
    avail = detect_available_engines()
    assert isinstance(avail, dict)
    assert avail.get("jetf") is True
    assert avail.get("blink") is True
    assert avail.get("flashentropy") is True


def test_multi_engine_benchmark_runner_pipeline(mini_dataset: BenchmarkDataset):
    avail = detect_available_engines()
    active_engines = [e for e in ["jetf", "blink", "flashentropy"] if avail.get(e, False)]

    runner = MultiEngineBenchmarkRunner(
        dataset=mini_dataset,
        engines=active_engines,
        tolerance_da=0.02,
    )

    # 1. 验证索引构建记录
    for eng in active_engines:
        assert eng in runner.index_stats
        assert runner.index_stats[eng].n_spectra == 20
        assert runner.index_stats[eng].build_time_s >= 0.0

    # 2. 检索延迟与吞吐评测
    queries = [(i, mini_dataset.library.peaks.spectrum_at(i)) for i in range(5)]
    lat_stats = runner.run_latency_benchmark(queries, mode="open", top_k=5, warmup_queries=1)

    for eng in active_engines:
        assert eng in lat_stats
        stat = lat_stats[eng]
        assert stat.n_queries == 5
        assert stat.mean_ms >= 0.0
        assert stat.p50_ms >= 0.0
        assert stat.qps > 0.0

    # 3. 伸缩性扩展评测
    scaling = runner.run_scaling_benchmark(scales=[5, 10], n_queries=3, mode="open", top_k=5)
    assert len(scaling) == 2
    assert scaling[0]["scale"] == 5
    assert scaling[1]["scale"] == 10

    # 4. 全景报告生成与控制台格式化
    report = runner.generate_report(n_queries=5, mode="open", top_k=5, run_scaling=False)
    assert isinstance(report, UnifiedBenchmarkReport)
    assert report.library_size == 20

    rep_dict = report.to_dict()
    assert "latency_stats" in rep_dict
    assert "index_stats" in rep_dict
    json_str = json.dumps(rep_dict)
    assert len(json_str) > 0

    table_str = report.format_console_table()
    assert "JET-Forest vs BLINK vs FlashEntropy" in table_str
    assert "Online Retrieval Latency" in table_str
