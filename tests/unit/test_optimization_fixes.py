"""针对百万级谱图构建优化（matchms 日志静音、parse_mgf 进度与 keep_rejected、建树进度、基准流式清洗）的单元测试。"""

import io
import logging
from pathlib import Path
import tempfile
import numpy as np
import pytest

from jetf.builder import build_forest_index
from jetf.cleaning import MatchmsCleanConfig, silence_matchms_logging
from jetf.mgf import ParsedLibrary, RejectReason, parse_mgf
from jetf.benchmarks.dataset import load_benchmark_dataset
from jetf.preprocessing import preprocess_library
from jetf.structure import ForestSpec


_SAMPLE_MGF_CONTENT = """BEGIN IONS
TITLE=spec_1
PEPMASS=100.0
CHARGE=1+
IONMODE=Positive
10.0 100.0
20.0 200.0
30.0 300.0
40.0 400.0
END IONS

BEGIN IONS
TITLE=spec_bad_nan
PEPMASS=200.0
CHARGE=1+
IONMODE=Positive
10.0 nan
END IONS

BEGIN IONS
TITLE=spec_few_peaks
PEPMASS=300.0
CHARGE=1+
IONMODE=Positive
10.0 100.0
END IONS
"""


def test_silence_matchms_logging() -> None:
    silence_matchms_logging()
    logger = logging.getLogger("matchms")
    assert logger.level == logging.ERROR
    for h in logger.handlers:
        assert h.level == logging.ERROR


def test_parse_mgf_keep_rejected_true_and_false() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mgf", delete=False, encoding="utf-8") as f:
        f.write(_SAMPLE_MGF_CONTENT)
        tmp_path = Path(f.name)

    try:
        # 1. keep_rejected=True (默认)
        p_keep = parse_mgf(tmp_path, keep_rejected=True, clean_config=MatchmsCleanConfig(min_peaks=3))
        assert p_keep.n_spectra == 1
        assert p_keep.rejected_count == 2
        assert len(p_keep.rejected) == 2
        assert p_keep.n_rejected == 2

        # 2. keep_rejected=False (节省内存)
        p_drop = parse_mgf(tmp_path, keep_rejected=False, clean_config=MatchmsCleanConfig(min_peaks=3))
        assert p_drop.n_spectra == 1
        assert p_drop.rejected_count == 2
        assert len(p_drop.rejected) == 0  # 列表不存对象
        assert p_drop.n_rejected == 2
    finally:
        tmp_path.unlink(missing_ok=True)


def test_parse_mgf_progress_interval(capsys: pytest.CaptureFixture[str]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mgf", delete=False, encoding="utf-8") as f:
        f.write(_SAMPLE_MGF_CONTENT)
        tmp_path = Path(f.name)

    try:
        # progress_interval=1 确保每条记录都触发打印
        parse_mgf(tmp_path, progress_interval=1)
        captured = capsys.readouterr()
        assert "[解析进度]" in captured.out
        assert "spec/s" in captured.out
    finally:
        tmp_path.unlink(missing_ok=True)


def test_build_forest_index_progress(capsys: pytest.CaptureFixture[str]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mgf", delete=False, encoding="utf-8") as f:
        f.write(_SAMPLE_MGF_CONTENT)
        tmp_path = Path(f.name)

    try:
        parsed = parse_mgf(tmp_path, keep_rejected=False)
        lib = preprocess_library(parsed)
        # progress_interval=1
        spec = ForestSpec(tree_capacity=4, leaf_capacity=2)
        forest = build_forest_index(lib, spec=spec, progress_interval=1)
        captured = capsys.readouterr()
        assert "[建树进度]" in captured.out
        assert "trees/s" in captured.out
        assert forest.n_trees >= 1
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_benchmark_dataset_streaming_clean() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mgf", delete=False, encoding="utf-8") as f:
        f.write(_SAMPLE_MGF_CONTENT)
        tmp_path = Path(f.name)

    try:
        ds = load_benchmark_dataset(
            mgf_path=tmp_path,
            library_size=10,
            clean_config=MatchmsCleanConfig(min_peaks=3),
        )
        # spec_1 有 4 个有效峰，保留；spec_bad_nan 语法错误；spec_few_peaks 仅 1 峰，流式清洗淘汰
        assert ds.n_spectra == 1
        assert ds.forest.n_spectra == 1
    finally:
        tmp_path.unlink(missing_ok=True)
