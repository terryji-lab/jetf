"""Unit tests for ForestIndex structure, builder, invariants, and serialization."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _subset import stratified_subset_indices, subset_parsed_library
from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FOREST_SPEC,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestIndex,
    ForestSpec,
    build_forest_index,
    build_query_context,
    check_forest_index,
    load_forest_snapshot,
    parse_mgf,
    peak_bound,
    preprocess_library,
    save_forest_snapshot,
    score_greedy_cosine,
)

LIBRARY_PATH = Path(__file__).resolve().parents[2] / "GNPS-LIBRARY.mgf"


@pytest.fixture(scope="module")
def subset_library():
    parsed = parse_mgf(LIBRARY_PATH)
    indices = stratified_subset_indices(parsed)
    sub_parsed = subset_parsed_library(parsed, indices)
    return preprocess_library(sub_parsed, CORRECTNESS_V1)


def test_build_forest_index_invariants(subset_library):
    spec = ForestSpec(tree_capacity=64, leaf_capacity=16, summary_grid_da=0.02)
    forest = build_forest_index(subset_library, spec)

    # 1. 不变量自检
    check_forest_index(forest, subset_library)

    assert forest.n_spectra == subset_library.n_spectra
    assert forest.n_trees > 0
    assert forest.n_nodes > forest.n_trees

    # 2. 验证覆盖性：森林内部 ID 覆盖的行序号无重无漏（加上零分侧车恰好等于全谱）
    all_indexed_rows = set(forest.internal_to_row)
    all_zero_rows = set(forest.zero_energy_members.member)
    assert len(all_indexed_rows.intersection(all_zero_rows)) == 0
    assert len(all_indexed_rows) + len(all_zero_rows) == subset_library.n_spectra

    # 3. 验证叶节点容量约束
    for node_id in range(forest.n_nodes):
        if forest.nodes.is_leaf[node_id]:
            count = int(forest.nodes.member_count[node_id])
            assert 4 <= count <= 24, f"叶节点容量异常: {count}"


def test_forest_envelope_safety(subset_library):
    forest = build_forest_index(subset_library, DEFAULT_FOREST_SPEC)

    # 抽样 5 条查询和 10 棵小树，验证根上界和叶上界均不低估真实分数
    np.random.seed(42)
    sample_queries = np.random.choice(forest.internal_to_row, size=5, replace=False)
    sample_trees = np.random.choice(forest.n_trees, size=min(10, forest.n_trees), replace=False)

    for q_row in sample_queries:
        q_peaks = subset_library.peaks.spectrum_at(q_row)
        for t_id in sample_trees:
            root_id = int(forest.trees.root_node_id[t_id])
            root_env = forest.envelope_of(root_id)
            root_ctx = build_query_context(q_peaks, root_env, DEFAULT_FRAGMENT_TOLERANCE_DA)
            u_root = peak_bound(root_ctx, root_env)

            # 遍历该树下的叶子
            leaf_ids = forest.trees.leaves_of_tree(t_id)
            for lid in leaf_ids:
                leaf_env = forest.envelope_of(lid)
                leaf_ctx = build_query_context(q_peaks, leaf_env, DEFAULT_FRAGMENT_TOLERANCE_DA)
                u_leaf = peak_bound(leaf_ctx, leaf_env)

                # 叶界必须被根界支配（或在其极小数值误差内）
                assert u_leaf <= u_root + 1e-9, f"叶界 ({u_leaf}) 超过根界 ({u_root})"

                # 成员真实分数检验
                leaf_rows = forest.rows_of(lid)
                for member_row in leaf_rows:
                    m_peaks = subset_library.peaks.spectrum_at(member_row)
                    real_score = score_greedy_cosine(
                        q_peaks, m_peaks, DEFAULT_FRAGMENT_TOLERANCE_DA
                    ).score
                    assert u_leaf >= real_score - 1e-9, (
                        f"叶界低估真实分数! u_leaf={u_leaf}, real_score={real_score}"
                    )
                    assert u_root >= real_score - 1e-9, (
                        f"根界低估真实分数! u_root={u_root}, real_score={real_score}"
                    )


def test_forest_snapshot_roundtrip(subset_library):
    forest = build_forest_index(subset_library, DEFAULT_FOREST_SPEC)

    with tempfile.TemporaryDirectory() as tmpdir:
        snap_path = Path(tmpdir) / "forest_snapshot.npz"
        save_forest_snapshot(forest, snap_path)
        loaded = load_forest_snapshot(snap_path)

        check_forest_index(loaded, subset_library)

        assert loaded.n_spectra == forest.n_spectra
        assert loaded.n_trees == forest.n_trees
        assert loaded.n_nodes == forest.n_nodes
        np.testing.assert_array_equal(loaded.internal_to_row, forest.internal_to_row)
        np.testing.assert_array_equal(loaded.row_to_internal, forest.row_to_internal)
        np.testing.assert_array_equal(loaded.trees.root_node_id, forest.trees.root_node_id)
        np.testing.assert_array_equal(loaded.nodes.id_start, forest.nodes.id_start)
        np.testing.assert_array_equal(loaded.envelopes.cell_index, forest.envelopes.cell_index)
