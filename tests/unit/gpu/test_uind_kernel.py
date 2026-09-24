"""Unit tests and benchmarks for GPU single-spectrum upper bound kernels (K3a).

Validates numerical correctness, zero-false-dismissal conservative inflation,
dense vs. paired evaluation equivalence, edge/corner cases, and throughput.
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
    IonMode,
    ParsedLibrary,
    SourceRef,
    SpectrumMeta,
    SpectrumPeaks,
    build_forest_index,
    parse_mgf,
    preprocess_library,
)
from jetf.scoring import _single_spectrum_bound_numba, single_spectrum_bound
from jetf.gpu import (
    BatchQueryDevice,
    GpuForestIndex,
    batch_uind_dense_gpu,
    batch_uind_pairs_gpu,
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
def test_uind_dense_synthetic_correctness(synthetic_forest_fixture):
    """对拍测试 1（合成数据 - 稠密模式）：验证 GPU U_ind 与 CPU _single_spectrum_bound_numba 对拍。

    - 断言严格零漏检保守性：uind_gpu >= uind_cpu
    - 断言相对误差紧致性：abs(uind_gpu - uind_cpu) / (uind_cpu + 1e-9) < 2e-3 (误差 < 0.2%)
    """
    forest, gpu_forest = synthetic_forest_fixture
    n_spectra = gpu_forest.n_spectra
    candidate_iids = np.arange(n_spectra, dtype=np.int64)

    # 构造 Q=6 条具有代表性的合成查询谱
    queries: list[SpectrumPeaks] = []
    for q_i in range(6):
        n_p = 5 + q_i * 3
        m = np.linspace(60.0 + q_i * 10.0, 140.0 + q_i * 15.0, n_p, dtype=np.float64)
        it = np.linspace(0.1, 1.0, n_p, dtype=np.float64)
        it = it / np.linalg.norm(it)
        queries.append(_make_single_query(list(m), list(it)))

    frag_tau = DEFAULT_FRAGMENT_TOLERANCE_DA
    batch_q = BatchQueryDevice.from_queries(
        queries,
        frag_tau=frag_tau,
        grid_da=gpu_forest.grid_da,
    )

    d_out = batch_uind_dense_gpu(batch_q, gpu_forest, candidate_iids)
    gpu_bounds = d_out.copy_to_host()

    assert gpu_bounds.shape == (len(queries), n_spectra)

    for q_idx, q in enumerate(queries):
        for c_pos, iid in enumerate(candidate_iids):
            lib_peaks = forest.postings.spectrum_at(int(iid))
            cpu_val = _single_spectrum_bound_numba(
                q.mass, q.intensity, lib_peaks.mass, lib_peaks.intensity, frag_tau
            )
            gpu_val = float(gpu_bounds[q_idx, c_pos])

            # 1. 严格零漏检保守性
            assert gpu_val >= cpu_val, (
                f"Conservative bound failed: gpu={gpu_val:.8f} < cpu={cpu_val:.8f} "
                f"for query {q_idx}, spectrum {iid}"
            )

            # 2. 紧致性保证：相对误差 < 0.2% (2e-3)
            rel_err = abs(gpu_val - cpu_val) / (cpu_val + 1e-9)
            assert rel_err < 2e-3, (
                f"Tightness bound failed: rel_err={rel_err:.6e} >= 2e-3, "
                f"gpu={gpu_val:.8f}, cpu={cpu_val:.8f} for query {q_idx}, spectrum {iid}"
            )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_uind_pairs_synthetic_correctness(synthetic_forest_fixture):
    """对拍测试 1（合成数据 - 稀疏配对模式）：验证 GPU 配对模式与 CPU 及稠密结果的一致性。"""
    forest, gpu_forest = synthetic_forest_fixture
    n_spectra = gpu_forest.n_spectra

    queries: list[SpectrumPeaks] = []
    for q_i in range(4):
        n_p = 6 + q_i * 2
        m = np.linspace(50.0 + q_i * 12.0, 130.0 + q_i * 10.0, n_p, dtype=np.float64)
        it = np.linspace(0.2, 0.9, n_p, dtype=np.float64)
        it = it / np.linalg.norm(it)
        queries.append(_make_single_query(list(m), list(it)))

    frag_tau = DEFAULT_FRAGMENT_TOLERANCE_DA
    batch_q = BatchQueryDevice.from_queries(
        queries,
        frag_tau=frag_tau,
        grid_da=gpu_forest.grid_da,
    )

    # 构造配对列表：混合采样
    q_indices_list: list[int] = []
    c_iids_list: list[int] = []
    for q_idx in range(len(queries)):
        for iid in range(0, n_spectra, 2):
            q_indices_list.append(q_idx)
            c_iids_list.append(iid)

    q_indices = np.array(q_indices_list, dtype=np.int64)
    c_iids = np.array(c_iids_list, dtype=np.int64)
    n_pairs = len(q_indices)

    # 1. 运行配对内核
    d_pairs_out = batch_uind_pairs_gpu(batch_q, gpu_forest, q_indices, c_iids)
    pairs_bounds = d_pairs_out.copy_to_host()
    assert pairs_bounds.shape == (n_pairs,)

    # 2. 运行稠密内核用于交叉核对
    d_dense_out = batch_uind_dense_gpu(batch_q, gpu_forest, np.arange(n_spectra, dtype=np.int64))
    dense_bounds = d_dense_out.copy_to_host()

    for tid in range(n_pairs):
        q_idx = int(q_indices[tid])
        iid = int(c_iids[tid])
        gpu_val = float(pairs_bounds[tid])
        dense_val = float(dense_bounds[q_idx, iid])

        # 配对与稠密数值应完全一致
        assert gpu_val == pytest.approx(dense_val, rel=1e-6, abs=1e-6)

        # 与 CPU 对拍
        lib_peaks = forest.postings.spectrum_at(iid)
        cpu_val = _single_spectrum_bound_numba(
            queries[q_idx].mass,
            queries[q_idx].intensity,
            lib_peaks.mass,
            lib_peaks.intensity,
            frag_tau,
        )

        assert gpu_val >= cpu_val
        rel_err = abs(gpu_val - cpu_val) / (cpu_val + 1e-9)
        assert rel_err < 2e-3


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_uind_gnps_subset_correctness(gnps_forest_fixture):
    """对拍测试 2（GNPS 子集数据）：使用真实质谱数据验证 GPU 与 CPU 对拍。

    - 断言严格零漏检保守性：uind_gpu >= uind_cpu (零漏检)
    - 断言紧致性：abs(uind_gpu - uind_cpu) / (uind_cpu + 1e-9) < 2e-3 (误差 < 0.2%)
    """
    lib, forest, gpu_forest = gnps_forest_fixture
    n_spectra = gpu_forest.n_spectra
    assert n_spectra > 0

    # 选取前 6 个真实库谱作为查询谱
    n_queries = min(6, n_spectra)
    queries: list[SpectrumPeaks] = []
    for i in range(n_queries):
        sp = lib.peaks.spectrum_at(i)
        queries.append(sp)

    frag_tau = DEFAULT_FRAGMENT_TOLERANCE_DA
    batch_q = BatchQueryDevice.from_queries(
        queries,
        frag_tau=frag_tau,
        grid_da=gpu_forest.grid_da,
    )

    candidate_iids = np.arange(n_spectra, dtype=np.int64)
    d_out = batch_uind_dense_gpu(batch_q, gpu_forest, candidate_iids)
    gpu_bounds = d_out.copy_to_host()

    for q_idx in range(n_queries):
        q = queries[q_idx]
        for c_pos in range(n_spectra):
            iid = int(candidate_iids[c_pos])
            lib_peaks = forest.postings.spectrum_at(iid)
            cpu_val = single_spectrum_bound(q, lib_peaks, frag_tau)
            gpu_val = float(gpu_bounds[q_idx, c_pos])

            # 验证严格零漏检保守性
            assert gpu_val >= cpu_val, (
                f"GNPS Conservative bound violated: gpu={gpu_val:.8f} < cpu={cpu_val:.8f} "
                f"for q={q_idx}, iid={iid}"
            )

            # 验证紧致性
            rel_err = abs(gpu_val - cpu_val) / (cpu_val + 1e-9)
            assert rel_err < 2e-3, (
                f"GNPS Tightness bound violated: rel_err={rel_err:.6e} >= 2e-3, "
                f"gpu={gpu_val:.8f}, cpu={cpu_val:.8f} for q={q_idx}, iid={iid}"
            )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_uind_edge_and_corner_cases(synthetic_forest_fixture):
    """边界条件测试：空谱、无交集谱、单峰谱、空批次、越界保护等。"""
    _, gpu_forest = synthetic_forest_fixture

    # 1. 空查询谱
    q_empty = _make_single_query([], [])
    # 2. 完全无质量交集的查询谱 (远远超出合成谱 50~150 Da 范围)
    q_disjoint = _make_single_query([5000.0, 5010.0, 5020.0], [1.0, 1.0, 1.0])
    # 3. 单峰查询谱
    q_single = _make_single_query([110.0], [1.0])

    batch_edge = BatchQueryDevice.from_queries(
        [q_empty, q_disjoint, q_single],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )

    candidate_iids = np.arange(gpu_forest.n_spectra, dtype=np.int64)
    d_out = batch_uind_dense_gpu(batch_edge, gpu_forest, candidate_iids)
    bounds = d_out.copy_to_host()

    # 空查询谱上界必须恒为 0.0
    np.testing.assert_allclose(bounds[0], 0.0, atol=1e-8)

    # 质量不交集谱上界必须恒为 0.0
    np.testing.assert_allclose(bounds[1], 0.0, atol=1e-8)

    # 单峰谱上界非负
    assert np.all(bounds[2] >= 0.0)

    # 4. 越界保护测试：candidate_iids 包含负数与超出 n_spectra 的 ID
    invalid_iids = np.array([-1, gpu_forest.n_spectra, gpu_forest.n_spectra + 100], dtype=np.int64)
    d_invalid = batch_uind_dense_gpu(batch_edge, gpu_forest, invalid_iids)
    invalid_bounds = d_invalid.copy_to_host()
    np.testing.assert_allclose(invalid_bounds, 0.0, atol=1e-8)

    # 配对内核中的越界保护
    q_idx_invalid = np.array([0, -1, 100], dtype=np.int64)
    c_iid_invalid = np.array([0, 0, -5], dtype=np.int64)
    d_pairs_invalid = batch_uind_pairs_gpu(batch_edge, gpu_forest, q_idx_invalid, c_iid_invalid)
    pairs_invalid_bounds = d_pairs_invalid.copy_to_host()
    assert pairs_invalid_bounds.shape == (3,)
    assert pairs_invalid_bounds[1] == 0.0
    assert pairs_invalid_bounds[2] == 0.0

    # 5. 空查询批次 (n_queries == 0)
    batch_zero = BatchQueryDevice.from_queries(
        [],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )
    d_zero_dense = batch_uind_dense_gpu(batch_zero, gpu_forest, candidate_iids)
    assert d_zero_dense.shape == (0, len(candidate_iids))

    d_zero_pairs = batch_uind_pairs_gpu(
        batch_zero,
        gpu_forest,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    assert d_zero_pairs.shape == (0,)

    # 6. 配对长度不匹配异常
    with pytest.raises(ValueError, match="does not match"):
        batch_uind_pairs_gpu(
            batch_edge,
            gpu_forest,
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )

    # 7. candidate_iids 为 None 时全库默认稠密评估
    d_all = batch_uind_dense_gpu(batch_edge, gpu_forest, candidate_iids=None)
    assert d_all.shape == (3, gpu_forest.n_spectra)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_uind_cuda_stream_execution(synthetic_forest_fixture):
    """验证自定义 CUDA Stream 异步流执行的正确性。"""
    forest, gpu_forest = synthetic_forest_fixture
    q = _make_single_query([70.0, 90.0, 110.0], [0.5, 0.8, 1.0])
    batch_q = BatchQueryDevice.from_queries(
        [q],
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )

    custom_stream = cuda.stream()
    candidate_iids = np.arange(gpu_forest.n_spectra, dtype=np.int64)

    d_dense = batch_uind_dense_gpu(batch_q, gpu_forest, candidate_iids, stream=custom_stream)
    d_pairs = batch_uind_pairs_gpu(
        batch_q,
        gpu_forest,
        np.zeros(gpu_forest.n_spectra, dtype=np.int64),
        candidate_iids,
        stream=custom_stream,
    )
    custom_stream.synchronize()

    dense_res = d_dense.copy_to_host()
    pairs_res = d_pairs.copy_to_host()

    np.testing.assert_allclose(dense_res[0], pairs_res, rtol=1e-6)
    assert np.all(dense_res >= 0.0)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_uind_throughput_benchmark(synthetic_forest_fixture):
    """吞吐量性能测试：在批量候选（如 10,000+ 对谱图）上测试耗时并打印对/秒吞吐。

    预期纯寄存器评估吞吐超千万次/秒 (> 10M evals/sec)。
    """
    _, gpu_forest = synthetic_forest_fixture
    n_spectra = gpu_forest.n_spectra

    # 构造 16 条合成查询谱
    queries: list[SpectrumPeaks] = []
    for i in range(16):
        n_p = 10 + (i % 8) * 2
        m = np.linspace(50.0 + (i % 4) * 10.0, 150.0 + (i % 4) * 15.0, n_p)
        it = np.linspace(0.1, 1.0, n_p)
        it = it / np.linalg.norm(it)
        queries.append(_make_single_query(list(m), list(it)))

    batch_q = BatchQueryDevice.from_queries(
        queries,
        frag_tau=DEFAULT_FRAGMENT_TOLERANCE_DA,
        grid_da=gpu_forest.grid_da,
    )

    # 构造 20,000 对候选配对
    n_pairs_target = 20_000
    rng = np.random.default_rng(42)
    q_indices = rng.integers(0, len(queries), size=n_pairs_target, dtype=np.int64)
    c_iids = rng.integers(0, n_spectra, size=n_pairs_target, dtype=np.int64)

    d_q_indices = cuda.to_device(q_indices)
    d_c_iids = cuda.to_device(c_iids)

    # Warm-up (JIT compilation and caching)
    for _ in range(5):
        _ = batch_uind_pairs_gpu(batch_q, gpu_forest, d_q_indices, d_c_iids)
        _ = batch_uind_dense_gpu(batch_q, gpu_forest, None)
    cuda.synchronize()

    # Benchmark 配对模式
    n_repeats = 100
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = batch_uind_pairs_gpu(batch_q, gpu_forest, d_q_indices, d_c_iids)
    cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_latency_ms = (total_time / n_repeats) * 1000.0
    total_evals = n_pairs_target * n_repeats
    throughput = total_evals / total_time

    print(
        f"\n[GPU K3a U_ind Pairs Benchmark] N_pairs={n_pairs_target:,}: "
        f"Avg latency = {avg_latency_ms:.3f} ms, "
        f"Throughput = {throughput:,.1f} U_ind pairs/sec"
    )

    assert avg_latency_ms > 0
    assert throughput > 10_000_000, f"Throughput {throughput:,.1f} < 10M evals/sec!"

    # Benchmark 稠密模式
    n_dense_tasks = len(queries) * n_spectra
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        _ = batch_uind_dense_gpu(batch_q, gpu_forest, None)
    cuda.synchronize()
    t1 = time.perf_counter()

    dense_time = t1 - t0
    dense_latency_ms = (dense_time / n_repeats) * 1000.0
    dense_throughput = (n_dense_tasks * n_repeats) / dense_time

    print(
        f"[GPU K3a U_ind Dense Benchmark] Q={len(queries)}, Lib={n_spectra} (Tasks={n_dense_tasks}): "
        f"Avg latency = {dense_latency_ms:.3f} ms, "
        f"Throughput = {dense_throughput:,.1f} U_ind dense/sec"
    )

    assert dense_throughput > 0


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_uind_adversarial_high_dynamic_range():
    """对抗样本测试：验证 300 峰高动态范围（强度跨越 1e-4 到 1e4 达 8 个数量级）下 GPU FP32 严格大于等于 CPU FP64。

    在极端峰数（300 峰）和巨大动态范围（10^8 跨度）下，FP32 单精度浮点由于尾数有效位数有限（24 bits），
    在大小数交错累加或重排序累加时极易产生下溢/舍入误差。
    此测试验证在 SAFETY_MARGIN_FP32 (1000 ppm / 0.1%) 安全膨胀因子加固下，
    GPU FP32 计算的 U_ind 上界严格大于等于 CPU FP64 参考基准（零漏检），且相对误差紧致受控（< 0.2%）。
    """
    n_peaks = 300
    rng = np.random.default_rng(2026)
    frag_tau = DEFAULT_FRAGMENT_TOLERANCE_DA

    # 1. 构造具有 300 峰、跨越 1e-4 到 1e4 高动态范围的库谱
    lib_mass = np.sort(rng.uniform(60.0, 1200.0, size=n_peaks))
    lib_raw_int = np.geomspace(1e-4, 1e4, n_peaks)
    rng.shuffle(lib_raw_int)  # 随机打乱强度形成大小交错的极端对抗排列

    meta = SpectrumMeta(
        external_id="adv_library_spec_300",
        precursor_mz=650.0,
        charge=1,
        ion_mode=IonMode.POSITIVE,
        source=SourceRef("adv_test.mgf", 0),
    )
    parsed = ParsedLibrary(
        source_path="adv_test.mgf",
        spectra=(meta,),
        mass=lib_mass,
        intensity=lib_raw_int,
        peak_id=np.arange(n_peaks, dtype=np.int64),
        spectrum_offsets=np.array([0, n_peaks], dtype=np.int64),
    )
    prep = preprocess_library(parsed, CORRECTNESS_V1)
    spec = ForestSpec(tree_capacity=4, leaf_capacity=2)
    forest = build_forest_index(prep, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)
    assert gpu_forest.n_spectra == 1

    # 2. 构造 3 组具有代表性的 300 峰对抗查询谱：
    # 查询 1：质量完全对齐库谱，强度随机交替排列（高动态范围对抗）
    q1_mass = np.copy(lib_mass)
    q1_raw_int = np.geomspace(1e-4, 1e4, n_peaks)
    rng.shuffle(q1_raw_int)
    q1_int = q1_raw_int / np.linalg.norm(q1_raw_int)
    q1 = _make_single_query(list(q1_mass), list(q1_int))

    # 查询 2：质量在 frag_tau 窗口内施加连续微小扰动 (在 +/- 0.5*frag_tau 内)，强度逆序排列（大数在前）
    q2_mass = lib_mass + rng.uniform(-0.5 * frag_tau, 0.5 * frag_tau, size=n_peaks)
    q2_raw_int = np.geomspace(1e-4, 1e4, n_peaks)[::-1]
    q2_order = np.argsort(q2_mass)
    q2_mass = q2_mass[q2_order]
    q2_raw_int = q2_raw_int[q2_order]
    q2_int = q2_raw_int / np.linalg.norm(q2_raw_int)
    q2 = _make_single_query(list(q2_mass), list(q2_int))

    # 查询 3：部分峰在容差内对齐（前 200 峰），其余 100 峰完全错开 (偏移 20 Da)，测试部分匹配场景
    q3_mass = np.copy(lib_mass)
    q3_mass[200:] += 20.0
    q3_raw_int = np.geomspace(1e-4, 1e4, n_peaks)
    rng.shuffle(q3_raw_int)
    q3_order = np.argsort(q3_mass)
    q3_mass = q3_mass[q3_order]
    q3_raw_int = q3_raw_int[q3_order]
    q3_int = q3_raw_int / np.linalg.norm(q3_raw_int)
    q3 = _make_single_query(list(q3_mass), list(q3_int))

    adversarial_queries = [q1, q2, q3]

    batch_q = BatchQueryDevice.from_queries(
        adversarial_queries,
        frag_tau=frag_tau,
        grid_da=gpu_forest.grid_da,
    )

    # 3. 执行 GPU 稠密与配对评估
    candidate_iids = np.array([0], dtype=np.int64)
    d_dense = batch_uind_dense_gpu(batch_q, gpu_forest, candidate_iids)
    gpu_dense_bounds = d_dense.copy_to_host()

    pair_q_indices = np.arange(len(adversarial_queries), dtype=np.int64)
    pair_c_iids = np.zeros(len(adversarial_queries), dtype=np.int64)
    d_pairs = batch_uind_pairs_gpu(batch_q, gpu_forest, pair_q_indices, pair_c_iids)
    gpu_pairs_bounds = d_pairs.copy_to_host()

    lib_peaks = forest.postings.spectrum_at(0)

    # 4. 对抗验证：逐一断言 GPU FP32 严格大于等于 CPU FP64
    for q_idx, q in enumerate(adversarial_queries):
        cpu_val = _single_spectrum_bound_numba(
            q.mass, q.intensity, lib_peaks.mass, lib_peaks.intensity, frag_tau
        )
        gpu_dense_val = float(gpu_dense_bounds[q_idx, 0])
        gpu_pair_val = float(gpu_pairs_bounds[q_idx])

        # 确保 GPU 稠密与配对模式计算结果一致
        assert gpu_dense_val == pytest.approx(gpu_pair_val, rel=1e-6, abs=1e-6)

        # 必须有效匹配（> 0.0）
        assert gpu_dense_val > 0.0, f"Query {q_idx} GPU bound must be positive"
        assert cpu_val > 0.0, f"Query {q_idx} CPU bound must be positive"

        # 核心断言：严格零漏检保守性 (GPU FP32 >= CPU FP64)
        assert gpu_dense_val >= cpu_val, (
            f"Adversarial 300-peak HDR check failed on query {q_idx}: "
            f"gpu_val={gpu_dense_val:.8f} < cpu_val={cpu_val:.8f}, "
            f"underflow gap={cpu_val - gpu_dense_val:.6e}"
        )

        # 紧致性断言：相对误差必须在 0.2% (2e-3) 范围内，未发生过度膨胀
        rel_err = abs(gpu_dense_val - cpu_val) / (cpu_val + 1e-9)
        assert rel_err < 2e-3, (
            f"Adversarial 300-peak HDR tightness violated on query {q_idx}: "
            f"rel_err={rel_err:.6e} >= 2e-3, gpu={gpu_dense_val:.8f}, cpu={cpu_val:.8f}"
        )
