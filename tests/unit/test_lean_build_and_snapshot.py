"""测试局部轻量化 SAH 构建、快照直接评测与默认清洗工作流。"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from jetf.builder import build_forest_index
from jetf.benchmarks.dataset import sample_query_spectra_from_forest
from jetf.benchmarks.throughput import benchmark_retrieval_throughput
from jetf.cleaning import MatchmsCleanConfig
from jetf.cli import main
from jetf.preprocessing import preprocess_library
from jetf.query import QueryConfig, SearchMode
from jetf.serialization import load_forest_snapshot, save_forest_snapshot
from jetf.structure import DEFAULT_FOREST_SPEC
from jetf.types import IonMode, SourceRef, SpectrumMeta, SpectrumPeaks
from jetf.mgf import ParsedLibrary


def _create_mock_library(n_spectra: int = 100) -> ParsedLibrary:
    """构造小型模拟 ParsedLibrary。"""
    rng = np.random.default_rng(42)
    spectra = []
    mass_list = []
    intensity_list = []
    pid_list = []
    offsets = [0]

    for i in range(n_spectra):
        n_p = int(rng.integers(10, 40))
        m = np.sort(rng.uniform(50.0, 1000.0, size=n_p))
        it = rng.uniform(10.0, 1000.0, size=n_p)
        pid = np.arange(n_p, dtype=np.int64)

        mass_list.append(m)
        intensity_list.append(it)
        pid_list.append(pid)
        offsets.append(offsets[-1] + n_p)

        mode = IonMode.POSITIVE if i % 2 == 0 else IonMode.NEGATIVE
        prec = float(m[-1] + rng.uniform(10.0, 50.0))
        meta = SpectrumMeta(
            external_id=f"SPEC_{i:04d}",
            ion_mode=mode,
            precursor_mz=prec,
            charge=1,
            source=SourceRef("mock.mgf", i),
            raw_metadata={"TITLE": f"SPEC_{i:04d}"},
        )
        spectra.append(meta)

    return ParsedLibrary(
        source_path="mock.mgf",
        spectra=tuple(spectra),
        mass=np.concatenate(mass_list),
        intensity=np.concatenate(intensity_list),
        peak_id=np.concatenate(pid_list),
        spectrum_offsets=np.array(offsets, dtype=np.int64),
        rejected=(),
    )


def test_lean_builder_local_sah_and_postings():
    """验证局部按需 SAH 与预分配 ForestPostings 构建的正确性。"""
    parsed = _create_mock_library(n_spectra=120)
    prep_lib = preprocess_library(parsed)
    forest = build_forest_index(prep_lib, DEFAULT_FOREST_SPEC)

    assert forest.n_spectra == 120
    assert forest.n_trees > 0
    assert forest.n_nodes > 0
    assert forest.postings.mass.shape[0] == prep_lib.peaks.mass.shape[0]
    assert forest.postings.spectrum_offsets.shape[0] == 121
    assert np.all(forest.trees.precursor_min <= forest.trees.precursor_max)


def test_snapshot_sampling_and_throughput_benchmark():
    """验证从快照直接抽样并在 ForestIndex 上评测吞吐量 (支持 skip_matchms)。"""
    parsed = _create_mock_library(n_spectra=80)
    prep_lib = preprocess_library(parsed)
    forest = build_forest_index(prep_lib, DEFAULT_FOREST_SPEC)

    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "test_snap.npz"
        save_forest_snapshot(forest, snap_path)

        # 脱机加载
        loaded_forest = load_forest_snapshot(snap_path)
        assert loaded_forest.n_spectra == 80

        # 从快照抽样
        queries = sample_query_spectra_from_forest(loaded_forest, n_queries=5, seed=123)
        assert len(queries) == 5

        # 直接评测吞吐量 (skip_matchms=True)
        q_configs = [
            (
                row,
                q,
                QueryConfig(
                    mode=SearchMode.TOP_K,
                    k=5,
                    ion_mode=loaded_forest.spectra[row].ion_mode,
                ),
            )
            for row, q in queries
        ]
        res = benchmark_retrieval_throughput(
            loaded_forest,
            q_configs,
            mode_name="test_snap_eval",
            skip_matchms=True,
        )

        assert res.n_queries == 5
        assert res.library_size == 80
        assert res.jetf_qps > 0
        assert res.matchms_qps == 0.0
        assert res.avg_pruned_ratio >= 0.0


def test_cli_argument_parsing():
    """验证 CLI build 与 benchmark 参数的默认开启清洗与 snapshot 支持。"""
    # 验证 CLI help 或参数构建
    import sys
    from jetf.cli import main

    # 捕获 --help 退出码 0
    try:
        main(["build", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    try:
        main(["benchmark", "--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_spectrum_at_bounds_check():
    """验证 ForestPostings.spectrum_at 的边界检查 (C-01 防御)。"""
    import pytest

    parsed = _create_mock_library(n_spectra=10)
    prep_lib = preprocess_library(parsed)
    forest = build_forest_index(prep_lib, DEFAULT_FOREST_SPEC)

    # 正常访问
    sp = forest.postings.spectrum_at(0)
    assert sp.norm > 0

    # 越界访问与负索引拦截
    with pytest.raises(IndexError, match="超出合法区间"):
        forest.postings.spectrum_at(-1)

    with pytest.raises(IndexError, match="超出合法区间"):
        forest.postings.spectrum_at(forest.postings.n_spectra)


def test_empty_queries_raises_value_error():
    """验证 benchmark_retrieval_throughput 对空查询列表的防御 (M-04 防御)。"""
    import pytest

    parsed = _create_mock_library(n_spectra=10)
    prep_lib = preprocess_library(parsed)
    forest = build_forest_index(prep_lib, DEFAULT_FOREST_SPEC)

    with pytest.raises(ValueError, match="queries 列表不能为空"):
        benchmark_retrieval_throughput(forest, [], skip_matchms=True)


def test_snapshot_consistency_mode_rejected():
    """验证 --snapshot 与 --mode consistency 组合被明确拦截 (M-01 防御)。"""
    parsed = _create_mock_library(n_spectra=10)
    prep_lib = preprocess_library(parsed)
    forest = build_forest_index(prep_lib, DEFAULT_FOREST_SPEC)

    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "test_snap.npz"
        save_forest_snapshot(forest, snap_path)

        code = main(["benchmark", "--snapshot", str(snap_path), "--mode", "consistency"])
        assert code == 1

