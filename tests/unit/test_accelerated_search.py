"""Unit tests for accelerated node bounds (JIT) and search optimizations in JET-Forest.

验证：
1. JIT 节点上界求值（_batch_node_bounds_numba, _single_node_bound_numba, batch_node_bounds）
   与原 Python build_query_context + peak_bound 计算结果数学完全等价（误差 <= 1e-12，单次浮点上偏一致）；
2. 优化后的 search_forest 与 search_exhaustive 在各类场景（Top-5, Top-10, Threshold）下
   100% 逐项逐位完全一致，零漏检（Zero False Dismissals）；
3. 边界用例（空查询、空树、非法索引防御）。
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _subset import stratified_subset_indices, subset_parsed_library

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    IonMode,
    QueryConfig,
    SearchMode,
    SpectrumPeaks,
    batch_node_bounds,
    build_forest_index,
    build_query_context,
    parse_mgf,
    peak_bound,
    preprocess_library,
    search_exhaustive,
    search_forest,
)
from jetf.bounds import (
    _HAVE_NUMBA,
    _batch_node_bounds_numpy,
    window_cells,
)

if _HAVE_NUMBA:
    from jetf.bounds import (
        _batch_node_bounds_numba,
        _leaf_bound_numba,
        _single_node_bound_numba,
    )


LIBRARY_PATH = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"

pytestmark = pytest.mark.skipif(
    not LIBRARY_PATH.is_file(),
    reason=f"缺少 {LIBRARY_PATH.name}：请确认 MGF 文件存在于根目录",
)


@pytest.fixture(scope="module")
def shared_fixture():
    parsed = parse_mgf(LIBRARY_PATH)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    library = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(library)
    return library, forest


def test_batch_node_bounds_exact_equivalence(shared_fixture):
    """验证批量节点求值与原 Python build_query_context + peak_bound 误差 <= 1e-12。"""
    library, forest = shared_fixture

    env_offsets = forest.envelopes.node_envelope_offsets
    cell_index = forest.envelopes.cell_index
    max_peak_amplitude = forest.envelopes.max_peak_amplitude

    # 抽样 3 条查询谱
    for q_row in [0, 15, 30]:
        q_peaks = library.peaks.spectrum_at(q_row)
        q_lower, q_upper = window_cells(
            q_peaks.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.spec.summary_grid_da
        )

        # 抽查前 10 棵小树的全部叶节点
        sample_leaf_ids = []
        for t_id in range(min(10, forest.n_trees)):
            sample_leaf_ids.extend(forest.trees.leaves_of_tree(t_id).tolist())

        leaf_ids_arr = np.array(sample_leaf_ids, dtype=np.int64)

        # 1. 逐个独立计算基准 (原 Python 实现)
        expected_bounds = []
        for nid in sample_leaf_ids:
            leaf_env = forest.envelope_of(nid)
            ctx = build_query_context(
                q_peaks, leaf_env, DEFAULT_FRAGMENT_TOLERANCE_DA, q_lower, q_upper
            )
            expected_bounds.append(peak_bound(ctx, leaf_env))

        expected_arr = np.array(expected_bounds, dtype=np.float64)

        # 2. NumPy 向量化降级实现
        q_intensity = np.ascontiguousarray(q_peaks.intensity, dtype=np.float64)
        numpy_bounds = _batch_node_bounds_numpy(
            leaf_ids_arr,
            env_offsets,
            cell_index,
            max_peak_amplitude,
            q_intensity,
            q_lower,
            q_upper,
        )
        np.testing.assert_allclose(
            numpy_bounds,
            expected_arr,
            atol=1e-12,
            rtol=1e-12,
            err_msg=f"查询 {q_row} NumPy 节点上界与原 Python 实现不一致",
        )

        # 3. 统一派发函数 batch_node_bounds
        public_bounds = batch_node_bounds(
            q_peaks, forest, leaf_ids_arr, q_lower, q_upper
        )
        np.testing.assert_allclose(
            public_bounds,
            expected_arr,
            atol=1e-12,
            rtol=1e-12,
            err_msg=f"查询 {q_row} batch_node_bounds 与原 Python 实现不一致",
        )

        # 4. 若环境支持 Numba，验证 JIT 内核与标量内核
        if _HAVE_NUMBA:
            numba_bounds = _batch_node_bounds_numba(
                leaf_ids_arr,
                env_offsets,
                cell_index,
                max_peak_amplitude,
                q_intensity,
                q_lower,
                q_upper,
            )
            np.testing.assert_allclose(
                numba_bounds,
                expected_arr,
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"查询 {q_row} Numba JIT 节点上界与原 Python 实现不一致",
            )

            # 验证标量内核
            for idx, nid in enumerate(sample_leaf_ids):
                single_b = _single_node_bound_numba(
                    int(nid),
                    env_offsets,
                    cell_index,
                    max_peak_amplitude,
                    q_intensity,
                    q_lower,
                    q_upper,
                )
                assert abs(single_b - expected_arr[idx]) <= 1e-12


def test_accelerated_search_forest_exhaustive_equivalence(shared_fixture):
    """验证加速后的 search_forest 与穷举检索 100% 逐项完全一致（零漏检）。"""
    library, forest = shared_fixture

    test_queries = [0, 10, 25]
    for row in test_queries:
        meta = library.spectra[row]
        q_peaks = library.peaks.spectrum_at(row)

        # 场景 A: Top-10
        cfg_top10 = QueryConfig(
            mode=SearchMode.TOP_K,
            k=10,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        )
        res_forest_10 = search_forest(q_peaks, forest, library, cfg_top10)
        res_exh_10 = search_exhaustive(q_peaks, library, cfg_top10)

        assert len(res_forest_10.hits) == len(res_exh_10.hits), (
            f"Top-10 命中数不一致: 森林 {len(res_forest_10.hits)} vs 穷举 {len(res_exh_10.hits)}"
        )
        for i, (hf, he) in enumerate(zip(res_forest_10.hits, res_exh_10.hits)):
            assert hf.external_id == he.external_id
            assert hf.spectrum_index == he.spectrum_index
            assert abs(hf.score - he.score) <= 1e-12
            assert hf.n_matched == he.n_matched

        # 场景 B: Top-5
        cfg_top5 = QueryConfig(
            mode=SearchMode.TOP_K,
            k=5,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        )
        res_forest_5 = search_forest(q_peaks, forest, library, cfg_top5)
        res_exh_5 = search_exhaustive(q_peaks, library, cfg_top5)
        assert len(res_forest_5.hits) == len(res_exh_5.hits)
        for hf, he in zip(res_forest_5.hits, res_exh_5.hits):
            assert hf.external_id == he.external_id
            assert abs(hf.score - he.score) <= 1e-12

        # 场景 C: Threshold 0.50
        cfg_thresh = QueryConfig(
            mode=SearchMode.THRESHOLD,
            threshold=0.50,
            ion_mode=meta.ion_mode,
            fragment_tolerance_da=DEFAULT_FRAGMENT_TOLERANCE_DA,
        )
        res_forest_t = search_forest(q_peaks, forest, library, cfg_thresh)
        res_exh_t = search_exhaustive(q_peaks, library, cfg_thresh)
        assert len(res_forest_t.hits) == len(res_exh_t.hits)
        for hf, he in zip(res_forest_t.hits, res_exh_t.hits):
            assert hf.external_id == he.external_id
            assert abs(hf.score - he.score) <= 1e-12


def test_accelerated_search_edge_cases(shared_fixture):
    """边缘用例测试：空查询、空节点列表、越界索引防御。"""
    library, forest = shared_fixture

    q_peaks = library.peaks.spectrum_at(0)
    q_lower, q_upper = window_cells(
        q_peaks.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.spec.summary_grid_da
    )

    # 1. 空节点列表
    empty_b = batch_node_bounds(q_peaks, forest, [], q_lower, q_upper)
    assert len(empty_b) == 0

    # 2. 空查询谱
    empty_q = SpectrumPeaks._create_unchecked(
        mass=np.empty(0, dtype=np.float64),
        intensity=np.empty(0, dtype=np.float64),
        energy=np.empty(0, dtype=np.float64),
        peak_id=np.empty(0, dtype=np.int64),
        norm=0.0,
    )
    empty_res = search_forest(
        empty_q,
        forest,
        library,
        QueryConfig(mode=SearchMode.TOP_K, k=5, ion_mode=IonMode.POSITIVE),
    )
    assert empty_res.complete is True

    # 3. 非法越界节点索引防御
    with pytest.raises(IndexError):
        batch_node_bounds(q_peaks, forest, [-1], q_lower, q_upper)
    with pytest.raises(IndexError):
        batch_node_bounds(q_peaks, forest, [forest.envelopes.n_nodes + 10], q_lower, q_upper)
