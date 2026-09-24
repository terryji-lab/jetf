"""Unit tests and benchmarks for GPU batch envelope bounds kernels (K1 & K2).

Validates numerical correctness, zero-false-dismissal conservative inflation,
boundary corner cases, and high-throughput evaluation performance.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np
import pytest
from numba import cuda

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestSpec,
    ParsedLibrary,
    SpectrumPeaks,
    build_forest_index,
    parse_mgf,
    preprocess_library,
)
from jetf.bounds import (
    _batch_node_bounds_numba,
    _batch_root_bounds_numba,
    window_cells,
)
from jetf.gpu import (
    BatchQueryDevice,
    GpuForestIndex,
    batch_node_bounds_gpu,
    batch_root_bounds_gpu,
    is_cuda_available,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _subset import stratified_subset_indices, subset_parsed_library
from tests.unit.gpu.test_device import make_synthetic_library

GNPS_PATH = Path(__file__).resolve().parents[3] / "GNPS-LIBRARY.mgf"


@pytest.fixture(scope="module")
def synthetic_forest_fixture():
    parsed = make_synthetic_library(n_spectra=48)
    prep = preprocess_library(parsed)
    # Using small capacity to generate multiple trees and nodes
    spec = ForestSpec(tree_capacity=8, leaf_capacity=4)
    forest = build_forest_index(prep, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)
    return forest, gpu_forest


@pytest.fixture(scope="module")
def gnps_forest_fixture():
    if not GNPS_PATH.exists():
        pytest.skip(f"GNPS dataset not found at {GNPS_PATH}")
    parsed = parse_mgf(GNPS_PATH)
    indices = stratified_subset_indices(parsed, n_per_stratum=20)
    sub = subset_parsed_library(parsed, indices)
    lib = preprocess_library(sub, CORRECTNESS_V1)
    spec = ForestSpec(tree_capacity=8, leaf_capacity=4)
    forest = build_forest_index(lib, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)
    return lib, forest, gpu_forest


def _make_single_query(masses: list[float], intensities: list[float]) -> SpectrumPeaks:
    m = np.asarray(masses, dtype=np.float64)
    it = np.asarray(intensities, dtype=np.float64)
    order = np.argsort(m)
    m = np.ascontiguousarray(m[order])
    it = np.ascontiguousarray(it[order])
    n = len(m)
    return SpectrumPeaks._create_unchecked(
        mass=m,
        intensity=it,
        energy=it**2,
        peak_id=np.arange(n, dtype=np.int64),
        norm=1.0,
    )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_batch_query_device_properties(synthetic_forest_fixture):
    """Verify BatchQueryDevice structure, properties, and memory metrics."""
    _, gpu_forest = synthetic_forest_fixture
    q1 = _make_single_query([100.0, 200.0], [1.0, 2.0])
    q2 = _make_single_query([150.0, 250.0, 350.0], [0.5, 1.5, 2.5])

    batch_q = BatchQueryDevice.from_queries(
        [q1, q2],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )

    assert batch_q.n_queries == 2
    assert batch_q.total_peaks == 5
    assert batch_q.frag_tau == DEFAULT_FRAGMENT_TOLERANCE_DA
    assert batch_q.grid_da == gpu_forest.grid_da

    offsets = batch_q.q_offsets.copy_to_host()
    np.testing.assert_array_equal(offsets, [0, 2, 5])

    intensities = batch_q.q_intensity.copy_to_host()
    np.testing.assert_allclose(intensities, [1.0, 2.0, 0.5, 1.5, 2.5], rtol=1e-6)

    assert batch_q.q_mass is not None
    masses = batch_q.q_mass.copy_to_host()
    assert masses.dtype == np.float64
    np.testing.assert_allclose(masses, [100.0, 200.0, 150.0, 250.0, 350.0], rtol=1e-9)

    assert batch_q.memory_bytes() > 0
    assert "queries=2" in repr(batch_q)
    assert "total_peaks=5" in repr(batch_q)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_k1_root_bounds_synthetic_single_and_multi_query(synthetic_forest_fixture):
    """对拍测试 1（合成数据）：验证单查询与 Q=8 多查询下 GPU K1 与 CPU 的对拍。

    - 断言严格零漏检保守性：gpu_bounds >= cpu_bounds
    - 断言紧致性：|gpu - cpu| / (cpu + 1e-9) < 2e-3 (误差 < 0.2%)
    """
    forest, gpu_forest = synthetic_forest_fixture
    n_trees = forest.n_trees
    all_tree_ids = np.arange(n_trees, dtype=np.int64)

    # 构造 Q=8 条具有代表性的合成查询
    queries: list[SpectrumPeaks] = []
    for q_i in range(8):
        n_p = 4 + q_i * 2
        m = np.linspace(60.0 + q_i * 5.0, 180.0 + q_i * 10.0, n_p)
        it = np.linspace(0.2, 2.0, n_p)
        queries.append(_make_single_query(list(m), list(it)))

    # 1. 验证单查询 Q=1
    single_q = queries[:1]
    batch_q1 = BatchQueryDevice.from_queries(
        single_q,
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=forest.envelopes.grid_da,
    )
    gpu_d1 = batch_root_bounds_gpu(batch_q1, gpu_forest)
    gpu_bounds1 = gpu_d1.copy_to_host()
    assert gpu_bounds1.shape == (1, n_trees)

    c_low1, c_high1 = window_cells(
        single_q[0].mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.envelopes.grid_da
    )
    cpu_bounds1 = _batch_root_bounds_numba(
        all_tree_ids,
        forest.trees.root_node_id,
        forest.envelopes.node_envelope_offsets,
        forest.envelopes.cell_index,
        forest.envelopes.max_peak_amplitude,
        single_q[0].intensity,
        c_low1,
        c_high1,
    ).reshape(1, n_trees)

    # 检验严格零漏检保守性与紧致性
    assert np.all(gpu_bounds1 >= cpu_bounds1), "单查询未能满足严格零漏检 U_GPU >= U_CPU"
    rel_err1 = np.abs(gpu_bounds1 - cpu_bounds1) / (cpu_bounds1 + 1e-9)
    assert np.all(rel_err1 < 2e-3), f"单查询相对误差超出 2e-3 上限: max={np.max(rel_err1)}"

    # 2. 验证多查询 Q=8
    batch_q8 = BatchQueryDevice.from_queries(
        queries,
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=forest.envelopes.grid_da,
    )
    gpu_d8 = batch_root_bounds_gpu(batch_q8, gpu_forest)
    gpu_bounds8 = gpu_d8.copy_to_host()
    assert gpu_bounds8.shape == (8, n_trees)

    cpu_bounds8 = np.zeros((8, n_trees), dtype=np.float64)
    for i, q in enumerate(queries):
        c_low, c_high = window_cells(
            q.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.envelopes.grid_da
        )
        cpu_bounds8[i] = _batch_root_bounds_numba(
            all_tree_ids,
            forest.trees.root_node_id,
            forest.envelopes.node_envelope_offsets,
            forest.envelopes.cell_index,
            forest.envelopes.max_peak_amplitude,
            q.intensity,
            c_low,
            c_high,
        )

    # 检验严格零漏检保守性与紧致性
    assert np.all(gpu_bounds8 >= cpu_bounds8), "多查询 Q=8 未能满足严格零漏检 U_GPU >= U_CPU"
    rel_err8 = np.abs(gpu_bounds8 - cpu_bounds8) / (cpu_bounds8 + 1e-9)
    assert np.all(rel_err8 < 2e-3), f"多查询 Q=8 相对误差超出 2e-3 上限: max={np.max(rel_err8)}"

    # 验证非全零有效性
    assert np.any(gpu_bounds8 > 0.0), "GPU 求界结果全为 0"


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_k1_and_k2_gnps_subset_correctness(gnps_forest_fixture):
    """对拍测试 2（GNPS 子集数据）：抽样真实库谱验证 K1 根上界与 K2 叶上界对拍。

    - K1 根上界与 CPU 对拍（严格零漏检保守性与紧致性）
    - K2 叶上界与 CPU _batch_node_bounds_numba 对拍（严格零漏检保守性与紧致性）
    - 树包络单调性：对每棵树和查询，U_leaf <= U_root + 1e-5
    """
    lib, forest, gpu_forest = gnps_forest_fixture
    n_trees = forest.n_trees
    n_nodes = forest.n_nodes
    all_tree_ids = np.arange(n_trees, dtype=np.int64)

    # 选取 5 条真实库谱作为查询
    sample_indices = [0, 5, 15, 25, 35]
    queries = [lib.peaks.spectrum_at(idx) for idx in sample_indices]
    batch_q = BatchQueryDevice.from_queries(
        queries,
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=forest.envelopes.grid_da,
    )

    # 1. K1 根上界验证
    gpu_d_roots = batch_root_bounds_gpu(batch_q, gpu_forest)
    gpu_root_bounds = gpu_d_roots.copy_to_host()

    cpu_root_bounds = np.zeros((len(queries), n_trees), dtype=np.float64)
    for i, q in enumerate(queries):
        c_low, c_high = window_cells(
            q.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.envelopes.grid_da
        )
        cpu_root_bounds[i] = _batch_root_bounds_numba(
            all_tree_ids,
            forest.trees.root_node_id,
            forest.envelopes.node_envelope_offsets,
            forest.envelopes.cell_index,
            forest.envelopes.max_peak_amplitude,
            q.intensity,
            c_low,
            c_high,
        )

    assert np.all(gpu_root_bounds >= cpu_root_bounds), "GNPS K1 未满足严格零漏检 U_GPU >= U_CPU"
    rel_err_root = np.abs(gpu_root_bounds - cpu_root_bounds) / (cpu_root_bounds + 1e-9)
    assert np.all(rel_err_root < 2e-3), f"GNPS K1 相对误差超限: max={np.max(rel_err_root)}"

    # 2. K2 叶上界验证：收集所有叶节点 ID
    leaf_node_ids = np.ascontiguousarray(forest.trees.leaf_node_ids, dtype=np.int64)
    gpu_d_leaves = batch_node_bounds_gpu(batch_q, gpu_forest, leaf_node_ids)
    gpu_leaf_bounds = gpu_d_leaves.copy_to_host()
    assert gpu_leaf_bounds.shape == (len(queries), len(leaf_node_ids))

    cpu_leaf_bounds = np.zeros((len(queries), len(leaf_node_ids)), dtype=np.float64)
    for i, q in enumerate(queries):
        c_low, c_high = window_cells(
            q.mass, DEFAULT_FRAGMENT_TOLERANCE_DA, forest.envelopes.grid_da
        )
        cpu_leaf_bounds[i] = _batch_node_bounds_numba(
            leaf_node_ids,
            forest.envelopes.node_envelope_offsets,
            forest.envelopes.cell_index,
            forest.envelopes.max_peak_amplitude,
            q.intensity,
            c_low,
            c_high,
        )

    assert np.all(gpu_leaf_bounds >= cpu_leaf_bounds), "GNPS K2 未满足严格零漏检 U_GPU >= U_CPU"
    rel_err_leaf = np.abs(gpu_leaf_bounds - cpu_leaf_bounds) / (cpu_leaf_bounds + 1e-9)
    assert np.all(rel_err_leaf < 2e-3), f"GNPS K2 相对误差超限: max={np.max(rel_err_leaf)}"

    # 3. 验证层次剪枝单调性：对每棵树，其各叶节点的上界 <= 该树根节点的上界
    for t_id in range(n_trees):
        l_start = forest.trees.leaf_offsets[t_id]
        l_end = forest.trees.leaf_offsets[t_id + 1]
        for q_idx in range(len(queries)):
            u_root = gpu_root_bounds[q_idx, t_id]
            for leaf_idx in range(l_start, l_end):
                u_leaf = gpu_leaf_bounds[q_idx, leaf_idx]
                assert u_leaf <= u_root + 1e-5, (
                    f"层次单调性违例: tree={t_id}, query={q_idx}, "
                    f"leaf_bound={u_leaf} > root_bound={u_root}"
                )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_bounds_kernels_corner_and_edge_cases(synthetic_forest_fixture):
    """边界条件测试：0 峰空查询、空树列表、空节点列表、空查询集、极端不相交等。"""
    forest, gpu_forest = synthetic_forest_fixture

    # 1. 0 峰空查询
    empty_peaks = SpectrumPeaks._create_unchecked(
        mass=np.empty(0, dtype=np.float64),
        intensity=np.empty(0, dtype=np.float64),
        energy=np.empty(0, dtype=np.float64),
        peak_id=np.empty(0, dtype=np.int64),
        norm=0.0,
    )
    batch_empty = BatchQueryDevice.from_queries(
        [empty_peaks],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )
    assert batch_empty.n_queries == 1
    assert batch_empty.total_peaks == 0
    bounds_empty = batch_root_bounds_gpu(batch_empty, gpu_forest).copy_to_host()
    assert bounds_empty.shape == (1, gpu_forest.n_trees)
    np.testing.assert_array_equal(bounds_empty, 0.0)

    # 2. 空查询列表 (Q=0)
    batch_q0 = BatchQueryDevice.from_queries(
        [],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )
    assert batch_q0.n_queries == 0
    bounds_q0 = batch_root_bounds_gpu(batch_q0, gpu_forest).copy_to_host()
    assert bounds_q0.shape == (0, gpu_forest.n_trees)

    # 3. 空树列表
    normal_q = _make_single_query([100.0, 150.0], [1.0, 2.0])
    batch_norm = BatchQueryDevice.from_queries(
        [normal_q],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )
    bounds_t0 = batch_root_bounds_gpu(batch_norm, gpu_forest, tree_ids=[]).copy_to_host()
    assert bounds_t0.shape == (1, 0)

    # 4. 空候选节点列表 (K2)
    bounds_n0 = batch_node_bounds_gpu(batch_norm, gpu_forest, node_ids=[]).copy_to_host()
    assert bounds_n0.shape == (1, 0)

    # 5. 单峰查询
    single_peak_q = _make_single_query([100.0], [1.0])
    batch_single = BatchQueryDevice.from_queries(
        [single_peak_q],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )
    bounds_single = batch_root_bounds_gpu(batch_single, gpu_forest).copy_to_host()
    assert bounds_single.shape == (1, gpu_forest.n_trees)
    assert np.all(bounds_single >= 0.0)

    # 6. 极端质量区间完全不重叠查询 (远超森林峰区间)
    disjoint_q = _make_single_query([50000.0, 50001.0], [1.0, 1.0])
    batch_disjoint = BatchQueryDevice.from_queries(
        [disjoint_q],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )
    bounds_disjoint = batch_root_bounds_gpu(batch_disjoint, gpu_forest).copy_to_host()
    np.testing.assert_array_equal(bounds_disjoint, 0.0)

    # 7. 指定子集 tree_ids 评估
    sub_trees = [0, 1] if gpu_forest.n_trees > 1 else [0]
    bounds_sub = batch_root_bounds_gpu(batch_norm, gpu_forest, tree_ids=sub_trees).copy_to_host()
    assert bounds_sub.shape == (1, len(sub_trees))


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_custom_cuda_stream_execution(synthetic_forest_fixture):
    """Verify asynchronous execution with a custom CUDA stream."""
    _, gpu_forest = synthetic_forest_fixture
    q = _make_single_query([120.0, 160.0], [1.0, 2.0])

    stream = cuda.stream()
    batch_q = BatchQueryDevice.from_queries(
        [q],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
        stream=stream,
    )
    d_bounds = batch_root_bounds_gpu(batch_q, gpu_forest, stream=stream)
    host_bounds = d_bounds.copy_to_host(stream=stream)
    stream.synchronize()

    assert host_bounds.shape == (1, gpu_forest.n_trees)
    assert np.all(host_bounds >= 0.0)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_bounds_kernel_benchmark(synthetic_forest_fixture):
    """性能基准验证：批量 Q=32 根求界耗时统计，计算并打印 GPU 耗时与吞吐（树求界次/秒）。"""
    _, gpu_forest = synthetic_forest_fixture
    n_trees = gpu_forest.n_trees

    # 构造 Q=32 批量查询
    queries: list[SpectrumPeaks] = []
    for i in range(32):
        n_p = 10 + (i % 10) * 2
        m = np.linspace(50.0 + (i % 5) * 10.0, 200.0 + (i % 5) * 15.0, n_p)
        it = np.linspace(0.1, 1.0, n_p)
        queries.append(_make_single_query(list(m), list(it)))

    batch_q = BatchQueryDevice.from_queries(
        queries,
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )

    # Warm-up (JIT compilation and memory caching)
    for _ in range(5):
        d_out = batch_root_bounds_gpu(batch_q, gpu_forest)
    cuda.synchronize()

    # Benchmark run
    n_repeats = 200
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        d_out = batch_root_bounds_gpu(batch_q, gpu_forest)
    cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_latency_ms = (total_time / n_repeats) * 1000.0
    total_evals = 32 * n_trees * n_repeats
    throughput = total_evals / total_time

    print(
        f"\n[GPU Benchmark] Q=32, trees={n_trees}: "
        f"Avg latency = {avg_latency_ms:.3f} ms, "
        f"Throughput = {throughput:,.1f} tree-bounds/sec"
    )

    assert avg_latency_ms > 0
    assert throughput > 0
