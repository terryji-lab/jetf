"""Unit tests for adaptive concurrency and search_forest_batch in JET-Forest."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    IonMode,
    QueryConfig,
    SearchMode,
    SpectrumPeaks,
    adaptive_numba_threads,
    build_forest_index,
    parse_mgf,
    preprocess_library,
    search_forest,
    search_forest_batch,
)
from jetf.bounds import _HAVE_NUMBA

LIBRARY_PATH = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"

pytestmark = pytest.mark.skipif(
    not LIBRARY_PATH.is_file(),
    reason=f"缺少 {LIBRARY_PATH.name}：请确认 MGF 文件存在于根目录",
)


def test_adaptive_numba_threads_context():
    """测试 adaptive_numba_threads 上下文管理器行为及现场恢复。"""
    if not _HAVE_NUMBA:
        pytest.skip("环境无 Numba 依赖")

    import numba

    orig = numba.get_num_threads()

    # 1. concurrency <= 1 时不应改动线程数
    with adaptive_numba_threads(1) as inner:
        assert inner is None
        assert numba.get_num_threads() == orig

    # 2. concurrency > 1 时自适应调整
    total_cores = os.cpu_count() or 1
    expected_inner = max(1, total_cores // 16)
    with adaptive_numba_threads(16) as inner:
        assert inner == expected_inner
        assert numba.get_num_threads() == expected_inner

    # 退出后必须恢复
    assert numba.get_num_threads() == orig

    # 3. 异常退出测试
    with pytest.raises(RuntimeError):
        with adaptive_numba_threads(8):
            raise RuntimeError("故意抛出测试异常")

    assert numba.get_num_threads() == orig


def test_search_forest_batch_equivalence():
    """验证 search_forest_batch 并发检索结果与单线程串行完全逐项一致。"""
    parsed = parse_mgf(LIBRARY_PATH, max_records=200)
    library = preprocess_library(parsed, CORRECTNESS_V1)
    forest = build_forest_index(library)

    sample_rows = [0, 5, 10, 20, 30]
    queries = [library.peaks.spectrum_at(r) for r in sample_rows]
    cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=5,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        ion_mode=library.spectra[sample_rows[0]].ion_mode,
    )

    # 1. 串行基准
    seq_results = [search_forest(q, forest, library, cfg) for q in queries]

    # 2. 批量并发检索 (4 线程)
    batch_results = search_forest_batch(
        queries, forest, library, config=cfg, concurrency=4
    )

    assert len(batch_results) == len(seq_results)
    for b_out, s_out in zip(batch_results, seq_results):
        assert b_out.complete is True
        assert len(b_out.hits) == len(s_out.hits)
        for bh, sh in zip(b_out.hits, s_out.hits):
            assert bh.external_id == sh.external_id
            assert abs(bh.score - sh.score) <= 1e-12
            assert bh.n_matched == sh.n_matched

    # 3. 参数校验边界测试
    assert search_forest_batch([], forest, library, cfg) == []

    with pytest.raises(ValueError, match="长度.*不匹配"):
        search_forest_batch(queries, forest, library, config=[cfg])

    with pytest.raises(ValueError, match="必须提供 config 参数"):
        search_forest_batch(queries, forest, library, config=None)
