"""测试统一多后端多引擎调度器 (MultiEngineBenchmarkRunner) 在脱机快照与 GPU 异构流水线下的表现。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from jetf.benchmarks.unified_runner import (
    MultiEngineBenchmarkRunner,
    UnifiedBenchmarkReport,
    detect_available_engines,
)
from jetf.builder import build_forest_index
from jetf.cleaning import clean_parsed_library
from jetf.mgf import ParsedLibrary
from jetf.preprocessing import preprocess_library
from jetf.serialization import save_forest_snapshot
from jetf.structure import DEFAULT_FOREST_SPEC
from jetf.types import IonMode, SourceRef, SpectrumMeta


@pytest.fixture
def temp_snapshot_path(tmp_path: Path) -> Path:
    """生成一个微型快照用于脱机测试。"""
    n_spectra = 16
    spectra = []
    masses = []
    intensities = []
    peak_ids = []
    offsets = [0]

    for i in range(n_spectra):
        spectra.append(
            SpectrumMeta(
                external_id=f"SNAP_{i:03d}",
                ion_mode=IonMode.POSITIVE,
                precursor_mz=150.0 + i * 20.0,
                charge=1,
                source=SourceRef(path="synth.mgf", record_index=i),
            )
        )
        mz = np.sort(np.linspace(50.0 + i, 500.0 + i, 10, dtype=np.float64))
        it = np.ones(10, dtype=np.float64)
        masses.append(mz)
        intensities.append(it)
        peak_ids.append(np.arange(10, dtype=np.int64))
        offsets.append(offsets[-1] + 10)

    parsed = ParsedLibrary(
        source_path="synth.mgf",
        spectra=tuple(spectra),
        mass=np.concatenate(masses),
        intensity=np.concatenate(intensities),
        peak_id=np.concatenate(peak_ids),
        spectrum_offsets=np.array(offsets, dtype=np.int64),
        rejected=(),
    )
    lib = preprocess_library(parsed)
    forest = build_forest_index(lib, DEFAULT_FOREST_SPEC)

    snap_file = tmp_path / "test_synth_forest.npz"
    save_forest_snapshot(forest, snap_file)
    return snap_file


def test_unified_snapshot_gpu_runner(temp_snapshot_path: Path):
    avail = detect_available_engines()
    engines = ["jetf-cpu-mt", "jetf-cpu-1t"]
    if avail.get("jetf-gpu", False):
        engines.insert(0, "jetf-gpu")

    runner = MultiEngineBenchmarkRunner(
        dataset=temp_snapshot_path,
        engines=engines,
        batch_size=8,
        tolerance_da=0.02,
        clean_matchms=True,
    )

    assert runner.library_size == 16
    assert "jetf-cpu-1t" in runner.index_stats

    if avail.get("jetf-gpu", False):
        assert "jetf-gpu" in runner.index_stats
        gpu_stat = runner.index_stats["jetf-gpu"]
        assert gpu_stat.vram_mb >= 0.0
        assert gpu_stat.upload_time_s >= 0.0

    report = runner.generate_report(n_queries=4, mode="open", top_k=5, batch_size=8)
    assert isinstance(report, UnifiedBenchmarkReport)
    assert report.library_size == 16
    assert report.batch_size == 8

    # 验证加速比与时延
    rep_dict = report.to_dict()
    assert "speedups_vs_1t" in rep_dict
    assert "latency_stats" in rep_dict
    assert "jetf-cpu-1t" in rep_dict["latency_stats"]

    table_str = report.format_console_table()
    assert "JET-Forest vs BLINK vs FlashEntropy" in table_str
    assert "Batch Throughput QPS" in table_str
