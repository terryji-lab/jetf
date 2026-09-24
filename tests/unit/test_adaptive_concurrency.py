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
    ParsedLibrary,
    QueryConfig,
    SearchMode,
    SourceRef,
    SpectrumMeta,
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
skip_if_no_mgf = pytest.mark.skipif(
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
    base_threads = numba.get_num_threads()
    expected_inner = max(1, min(base_threads, base_threads // 16))
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

    # 4. target_threads 显式设置测试
    with adaptive_numba_threads(target_threads=1) as inner:
        assert inner == 1
        assert numba.get_num_threads() == 1
    assert numba.get_num_threads() == orig

    # target_threads <= 0 应收敛为 1
    with adaptive_numba_threads(target_threads=0) as inner:
        assert inner == 1
        assert numba.get_num_threads() == 1
    assert numba.get_num_threads() == orig


def test_adaptive_numba_threads_restricted_environment(monkeypatch):
    """测试受限线程环境下 adaptive_numba_threads 不会抛出 ValueError，且防御性降级生效。"""
    if not _HAVE_NUMBA:
        pytest.skip("环境无 Numba 依赖")

    import numba

    orig = numba.get_num_threads()
    try:
        # 1. 模拟已有环境被限制为 2 线程，外部请求 4 并发
        numba.set_num_threads(2)
        assert numba.get_num_threads() == 2

        with adaptive_numba_threads(4) as inner:
            # base_threads=2, 2 // 4 = 0 -> max(1, min(2, 0)) = 1
            assert inner == 1
            assert numba.get_num_threads() == 1

        assert numba.get_num_threads() == 2

        # 2. 模拟单线程受限环境，请求 8 并发
        numba.set_num_threads(1)
        with adaptive_numba_threads(8) as inner:
            assert inner == 1
            assert numba.get_num_threads() == 1

        assert numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(orig)

    # 3. 防御性降级测试：模拟 set_num_threads 发生未知异常，确保不影响主调用
    def _mock_raise(*args, **kwargs):
        raise RuntimeError("底层运行时禁止修改线程数")

    monkeypatch.setattr(numba, "set_num_threads", _mock_raise)
    with adaptive_numba_threads(4) as inner:
        assert inner is None


@pytest.fixture
def synthetic_index():
    """轻量内存测试索引，无外部 MGF 文件依赖。"""
    metas = [
        SpectrumMeta(
            external_id=f"lib_{i}",
            precursor_mz=200.0 + i * 50.0,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("synth.mgf", i),
            raw_metadata={},
        )
        for i in range(10)
    ]
    mass_list = []
    intensity_list = []
    peak_id_list = []
    offsets = [0]
    for i in range(10):
        m = np.linspace(50.0 + i * 10.0, 300.0 + i * 10.0, 5, dtype=np.float64)
        it = np.ones(5, dtype=np.float64) * 50.0
        pid = np.arange(5, dtype=np.int64)
        mass_list.append(m)
        intensity_list.append(it)
        peak_id_list.append(pid)
        offsets.append(offsets[-1] + 5)

    parsed = ParsedLibrary(
        source_path="synth.mgf",
        spectra=tuple(metas),
        mass=np.concatenate(mass_list),
        intensity=np.concatenate(intensity_list),
        peak_id=np.concatenate(peak_id_list),
        spectrum_offsets=np.array(offsets, dtype=np.int64),
    )
    lib = preprocess_library(parsed, CORRECTNESS_V1)
    forest = build_forest_index(lib)

    queries = [
        lib.peaks.spectrum_at(0),
        lib.peaks.spectrum_at(3),
        lib.peaks.spectrum_at(7),
    ]
    return lib, forest, queries


def test_search_forest_batch_positional_config(synthetic_index):
    """测试 search_forest_batch(queries, forest, cfg) 位置参数调用（省略 library）。"""
    lib, forest, queries = synthetic_index
    cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=3,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        ion_mode=IonMode.POSITIVE,
    )

    # 1. 第 3 位置参数为单个 QueryConfig
    res_pos = search_forest_batch(queries, forest, cfg, concurrency=2)
    # 基准：显式关键字参数调用
    res_kw = search_forest_batch(queries, forest, config=cfg, concurrency=2)

    assert len(res_pos) == len(res_kw) == len(queries)
    for r_p, r_k in zip(res_pos, res_kw):
        assert len(r_p.hits) == len(r_k.hits)
        for hp, hk in zip(r_p.hits, r_k.hits):
            assert hp.external_id == hk.external_id
            assert abs(hp.score - hk.score) <= 1e-12

    # 2. 第 3 位置参数为 Sequence[QueryConfig]
    cfgs = [cfg] * len(queries)
    res_seq_pos = search_forest_batch(queries, forest, cfgs, concurrency=2)
    assert len(res_seq_pos) == len(queries)
    for r_sp, r_k in zip(res_seq_pos, res_kw):
        assert len(r_sp.hits) == len(r_k.hits)


def test_search_forest_batch_heterogeneous_configs(synthetic_index):
    """测试针对每个查询提供独立 Sequence[QueryConfig] 的异构配置并发检索。"""
    lib, forest, queries = synthetic_index

    cfgs = [
        QueryConfig(
            mode=SearchMode.TOP_K,
            k=1,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            ion_mode=IonMode.POSITIVE,
        ),
        QueryConfig(
            mode=SearchMode.TOP_K,
            k=2,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            ion_mode=IonMode.POSITIVE,
        ),
        QueryConfig(
            mode=SearchMode.THRESHOLD,
            threshold=0.1,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
            ion_mode=IonMode.POSITIVE,
        ),
    ]

    # 并发执行 (concurrency=3)
    batch_res = search_forest_batch(queries, forest, library=lib, config=cfgs, concurrency=3)

    # 串行逐项执行作为基准对照
    seq_res = [search_forest(queries[i], forest, lib, cfgs[i]) for i in range(len(queries))]

    assert len(batch_res) == len(seq_res) == 3
    assert len(batch_res[0].hits) <= 1
    assert len(batch_res[1].hits) <= 2
    assert batch_res[2].mode == SearchMode.THRESHOLD

    for b_out, s_out in zip(batch_res, seq_res):
        assert b_out.mode == s_out.mode
        assert len(b_out.hits) == len(s_out.hits)
        for bh, sh in zip(b_out.hits, s_out.hits):
            assert bh.external_id == sh.external_id
            assert abs(bh.score - sh.score) <= 1e-12


def test_search_forest_batch_edge_cases(synthetic_index):
    """测试边界情况：concurrency <= 0、concurrency > len(queries)、uind=False。"""
    lib, forest, queries = synthetic_index
    cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=3,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        ion_mode=IonMode.POSITIVE,
    )

    # 1. concurrency <= 0 自动收敛为 1
    res_zero = search_forest_batch(queries, forest, library=lib, config=cfg, concurrency=0)
    res_neg = search_forest_batch(queries, forest, library=lib, config=cfg, concurrency=-5)
    res_one = search_forest_batch(queries, forest, library=lib, config=cfg, concurrency=1)

    assert len(res_zero) == len(queries)
    assert len(res_neg) == len(queries)
    for r_z, r_1 in zip(res_zero, res_one):
        assert [h.external_id for h in r_z.hits] == [h.external_id for h in r_1.hits]

    # 2. concurrency > len(queries) 自动收敛有效并发
    res_oversub = search_forest_batch(queries, forest, library=lib, config=cfg, concurrency=100)
    assert len(res_oversub) == len(queries)
    for r_o, r_1 in zip(res_oversub, res_one):
        assert [h.external_id for h in r_o.hits] == [h.external_id for h in r_1.hits]

    # 3. uind=False 参数透传测试
    res_no_uind = search_forest_batch(
        queries, forest, library=lib, config=cfg, concurrency=2, uind=False
    )
    seq_no_uind = [search_forest(q, forest, lib, cfg, uind=False) for q in queries]
    assert len(res_no_uind) == len(queries)
    for r_b, r_s in zip(res_no_uind, seq_no_uind):
        assert [h.external_id for h in r_b.hits] == [h.external_id for h in r_s.hits]
        for hb, hs in zip(r_b.hits, r_s.hits):
            assert abs(hb.score - hs.score) <= 1e-12


@skip_if_no_mgf
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


def test_search_forest_batch_parallel_root_bounds_flag(monkeypatch, synthetic_index):
    """测试 search_forest_batch 在多并发时将 parallel_root_bounds=False 透传给 worker。"""
    lib, forest, queries = synthetic_index
    cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=3,
        fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        ion_mode=IonMode.POSITIVE,
    )
    import jetf.search as search_mod

    observed_flags = []
    orig_search_forest = search_mod.search_forest

    def _mock_search_forest(*args, **kwargs):
        flag = kwargs.get("parallel_root_bounds", True)
        observed_flags.append(flag)
        return orig_search_forest(*args, **kwargs)

    monkeypatch.setattr(search_mod, "search_forest", _mock_search_forest)

    # 1. 单线程串行执行时，parallel_root_bounds 默认为 True
    observed_flags.clear()
    search_forest_batch(queries, forest, library=lib, config=cfg, concurrency=1)
    assert len(observed_flags) == len(queries)
    assert all(f is True for f in observed_flags)

    # 2. 多线程并发执行时，parallel_root_bounds 应被设为 False 防止嵌套过度订阅
    observed_flags.clear()
    search_forest_batch(queries, forest, library=lib, config=cfg, concurrency=2)
    assert len(observed_flags) == len(queries)
    assert all(f is False for f in observed_flags)

