"""Unit tests for batch_root_bounds SIMD/vectorized root bound evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    batch_root_bounds,
    build_forest_index,
    build_query_context,
    parse_mgf,
    peak_bound,
    preprocess_library,
)
from jetf.bounds import window_cells
from jetf.types import SpectrumPeaks


def test_batch_root_bounds_exact_equivalence():
    """验证批量根求值结果与单个 build_query_context + peak_bound 计算结果完全一致。"""
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _subset import stratified_subset_indices, subset_parsed_library

    lib_path = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"
    parsed = parse_mgf(lib_path)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    library = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(library)

    # 抽样 3 条真实库谱进行查询比对
    for q_row in [0, 10, 20]:
        q_peaks = library.peaks.spectrum_at(q_row)
        q_lower, q_upper = window_cells(
            q_peaks.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.spec.summary_grid_da
        )

        all_tree_ids = list(range(forest.n_trees))

        # 逐个独立计算基准
        expected_bounds = []
        for t_id in all_tree_ids:
            root_id = int(forest.trees.root_node_id[t_id])
            root_env = forest.envelope_of(root_id)
            ctx = build_query_context(
                q_peaks, root_env, DEFAULT_FRAGMENT_TOLERANCE_DA, q_lower, q_upper
            )
            expected_bounds.append(peak_bound(ctx, root_env))

        # 批量向量化计算
        actual_bounds = batch_root_bounds(
            q_peaks, forest, all_tree_ids, q_lower, q_upper
        )

        np.testing.assert_allclose(
            actual_bounds,
            expected_bounds,
            atol=1e-12,
            rtol=1e-12,
            err_msg=f"查询 {q_row} 的批量根上界与逐根求值不一致",
        )


def test_batch_root_bounds_empty_and_corner_cases():
    """边界情况：空树列表、空查询峰等。"""
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _subset import stratified_subset_indices, subset_parsed_library

    lib_path = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"
    parsed = parse_mgf(lib_path)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    library = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(library)

    # 空树列表
    q_peaks = library.peaks.spectrum_at(0)
    q_lower, q_upper = window_cells(
        q_peaks.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.spec.summary_grid_da
    )
    b_empty = batch_root_bounds(q_peaks, forest, [], q_lower, q_upper)
    assert len(b_empty) == 0

    # 空查询
    empty_peaks = SpectrumPeaks._create_unchecked(
        mass=np.empty(0, dtype=np.float64),
        intensity=np.empty(0, dtype=np.float64),
        energy=np.empty(0, dtype=np.float64),
        peak_id=np.empty(0, dtype=np.int64),
        norm=0.0,
    )
    b_no_peaks = batch_root_bounds(empty_peaks, forest, [0, 1], np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    assert np.all(b_no_peaks == 0.0)


def test_batch_root_bounds_numpy_vs_numba_cross_check():
    """显式测试 NumPy fallback 实现，并验证与 Numba JIT 内核的严格数值等价性。"""
    from pathlib import Path
    import sys
    import jetf.bounds as b
    from jetf.bounds import _batch_root_bounds_numpy

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _subset import stratified_subset_indices, subset_parsed_library

    lib_path = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"
    parsed = parse_mgf(lib_path)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    library = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(library)

    for q_row in [0, 5, 15]:
        q_peaks = library.peaks.spectrum_at(q_row)
        q_lower, q_upper = window_cells(
            q_peaks.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.spec.summary_grid_da
        )
        tree_ids = np.arange(forest.n_trees, dtype=np.int64)

        # 1. 直接调用 NumPy fallback 函数
        res_numpy_direct = _batch_root_bounds_numpy(
            tree_ids,
            forest.trees.root_node_id,
            forest.envelopes.node_envelope_offsets,
            forest.envelopes.cell_index,
            forest.envelopes.max_peak_amplitude,
            q_peaks.intensity,
            q_lower,
            q_upper,
        )

        # 2. 模拟无 Numba 环境 (_HAVE_NUMBA = False) 调用公共入口
        orig_have_numba = b._HAVE_NUMBA
        try:
            b._HAVE_NUMBA = False
            res_numpy_entry = batch_root_bounds(q_peaks, forest, tree_ids, q_lower, q_upper)

            # 3. 恢复 Numba 调用公共入口
            b._HAVE_NUMBA = True
            res_numba = batch_root_bounds(q_peaks, forest, tree_ids, q_lower, q_upper)
        finally:
            b._HAVE_NUMBA = orig_have_numba

        np.testing.assert_allclose(res_numpy_direct, res_numpy_entry, atol=1e-15, rtol=1e-15)
        np.testing.assert_allclose(
            res_numba,
            res_numpy_direct,
            atol=1e-12,
            rtol=1e-12,
            err_msg=f"查询 {q_row} Numba 与 NumPy fallback 结果差异超出容差",
        )


def test_batch_root_bounds_defensive_validation():
    """验证非法参数与越界索引时的防御性异常抛出。"""
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _subset import stratified_subset_indices, subset_parsed_library

    lib_path = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"
    parsed = parse_mgf(lib_path)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    library = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(library)

    q_peaks = library.peaks.spectrum_at(0)
    q_lower, q_upper = window_cells(
        q_peaks.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.spec.summary_grid_da
    )

    # 1. 越界 tree_ids（负数与溢出）
    with pytest.raises(IndexError, match="tree_ids 存在超出合法范围"):
        batch_root_bounds(q_peaks, forest, [-1], q_lower, q_upper)

    with pytest.raises(IndexError, match="tree_ids 存在超出合法范围"):
        batch_root_bounds(q_peaks, forest, [forest.n_trees], q_lower, q_upper)

    # 2. 向量长度不匹配
    with pytest.raises(ValueError, match="查询向量长度不匹配"):
        batch_root_bounds(q_peaks, forest, [0], q_lower[:-1], q_upper)

    with pytest.raises(ValueError, match="查询向量长度不匹配"):
        batch_root_bounds(q_peaks, forest, [0], q_lower, q_upper[:-1])

