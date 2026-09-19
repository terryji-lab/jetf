"""Unit tests for BVH-SAH leaf partitioning algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from jetf.bvh_sah import sah_cost, split_sah_bvh


def test_sah_cost_empty():
    assert sah_cost([], []) == 0.0
    assert sah_cost([set()], [{}]) == 0.0


def test_sah_cost_simple():
    cells = [{10, 20}, {20, 30}]
    amps = [{10: 0.5, 20: 0.8}, {20: 0.6, 30: 0.7}]
    cost = sah_cost(cells, amps)
    assert pytest.approx(cost) == 6.0


def test_split_sah_bvh_small():
    indices = list(range(10))
    cells = [{i} for i in range(10)]
    amps = [{i: 1.0} for i in range(10)]
    leaves = split_sah_bvh(indices, cells, amps, target_leaf_size=16)
    assert len(leaves) == 1
    assert leaves[0] == indices


def test_split_sah_bvh_clean_separation():
    # 32 spectra with peaks at cell 100, and 32 spectra with peaks at cell 200
    indices = list(range(64))
    cells = []
    amps = []
    for i in range(32):
        cells.append({100, 1000 + i})
        amps.append({100: 0.9, 1000 + i: 0.1})
    for i in range(32):
        cells.append({200, 2000 + i})
        amps.append({200: 0.9, 2000 + i: 0.1})

    leaves = split_sah_bvh(indices, cells, amps, target_leaf_size=16)
    flat = [idx for leaf in leaves for idx in leaf]
    assert sorted(flat) == list(range(64))

    for leaf in leaves:
        assert 12 <= len(leaf) <= 20

    for leaf in leaves:
        is_g1 = all(x < 32 for x in leaf)
        is_g2 = all(x >= 32 for x in leaf)
        assert is_g1 or is_g2, "SAH should cleanly separate orthogonal clusters!"


def test_split_sah_bvh_determinism():
    indices = list(range(64))
    np.random.seed(42)
    cells = [set(np.random.choice(500, size=20, replace=False)) for _ in range(64)]
    amps = [{c: float(np.random.uniform(0.1, 1.0)) for c in s} for s in cells]

    leaves1 = split_sah_bvh(indices, cells, amps, target_leaf_size=16)
    leaves2 = split_sah_bvh(indices, cells, amps, target_leaf_size=16)

    assert leaves1 == leaves2
    flat = [idx for leaf in leaves1 for idx in leaf]
    assert len(flat) == 64
    assert len(set(flat)) == 64
