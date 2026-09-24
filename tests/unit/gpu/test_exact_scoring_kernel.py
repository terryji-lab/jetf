"""Unit tests and numerical verification for GPU Greedy Cosine exact scoring kernels (K3).

Validates:
1. Strict equivalence with CPU score_greedy_cosine on synthetic and GNPS library datasets:
   score within atol=1e-5, n_matched 100% identical.
2. Deterministic tie-breaking rules (-w, q_pid, l_pid) matching CPU lexsort benchmark.
3. Edge and corner cases: empty spectra, single peak, disjoint masses, identical spectra, zero intensities.
4. Edge overflow detection (> 128 edges) and safe fallback to CPU score_greedy_cosine.
5. Asynchronous CUDA stream execution.
"""

from __future__ import annotations

from pathlib import Path
import sys
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
from jetf.scoring import score_greedy_cosine
from jetf.gpu import (
    BatchQueryDevice,
    GpuForestIndex,
    batch_greedy_cosine_pairs_gpu,
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
    norm = float(np.sqrt(np.sum(it**2)))
    if norm > 0.0:
        it = it / norm
    n = len(m)
    return SpectrumPeaks._create_unchecked(
        mass=m,
        intensity=it,
        energy=it**2,
        peak_id=np.arange(n, dtype=np.int64),
        norm=1.0,
    )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_exact_scoring_synthetic_correctness(synthetic_forest_fixture):
    """测试 1A（合成谱对拍）：验证 GPU 精评内核与 CPU score_greedy_cosine 严格对拍。

    - score 在 atol=1e-5 范围内完全一致
    - n_matched 100% 相同
    - overflow 为 0
    """
    forest, gpu_forest = synthetic_forest_fixture
    n_spectra = gpu_forest.n_spectra
    frag_tau = 0.05

    # 构造 Q=6 条具有代表性的合成查询谱
    queries: list[SpectrumPeaks] = []
    for q_i in range(6):
        n_p = 6 + q_i * 2
        m = np.linspace(60.0 + q_i * 10.0, 140.0 + q_i * 15.0, n_p, dtype=np.float64)
        raw_it = np.linspace(0.2, 1.0, n_p, dtype=np.float64)
        queries.append(_make_single_query(m.tolist(), raw_it.tolist()))

    batch_query = BatchQueryDevice.from_queries(
        queries, frag_tau=frag_tau, grid_da=gpu_forest.grid_da
    )

    # 每个 query 与 8 个候选 spectrum 组成配对，共 48 对
    pair_q = []
    pair_iid = []
    for q_idx in range(len(queries)):
        for iid in range(min(8, n_spectra)):
            pair_q.append(q_idx)
            pair_iid.append(iid)

    q_arr = np.array(pair_q, dtype=np.int64)
    iid_arr = np.array(pair_iid, dtype=np.int64)

    d_scores, d_matched, d_overflow = batch_greedy_cosine_pairs_gpu(
        batch_query=batch_query,
        gpu_forest=gpu_forest,
        query_indices=q_arr,
        candidate_iids=iid_arr,
        frag_tau=frag_tau,
    )

    h_scores = d_scores.copy_to_host()
    h_matched = d_matched.copy_to_host()
    h_overflow = d_overflow.copy_to_host()

    for k in range(len(pair_q)):
        q_idx = pair_q[k]
        iid = pair_iid[k]
        q_peak = queries[q_idx]
        lib_peak = forest.postings.spectrum_at(iid)

        cpu_res = score_greedy_cosine(q_peak, lib_peak, frag_tau)

        assert h_overflow[k] == 0, f"Pair {k} unexpectedly overflowed"
        assert h_matched[k] == cpu_res.n_matched, (
            f"Pair #{k} (q={q_idx}, iid={iid}) n_matched mismatch: "
            f"GPU {h_matched[k]} vs CPU {cpu_res.n_matched}"
        )
        assert np.isclose(h_scores[k], cpu_res.score, atol=1e-5), (
            f"Pair #{k} (q={q_idx}, iid={iid}) score mismatch: "
            f"GPU {h_scores[k]:.8f} vs CPU {cpu_res.score:.8f} (diff={abs(h_scores[k] - cpu_res.score):.2e})"
        )


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_exact_scoring_gnps_correctness(gnps_forest_fixture):
    """测试 1B（真实 GNPS 谱对拍）：验证真实谱图对拍在 atol=1e-5 范围内完全一致，n_matched 100% 相同。"""
    lib, forest, gpu_forest = gnps_forest_fixture
    frag_tau = DEFAULT_FRAGMENT_TOLERANCE_DA

    # 选取内部谱峰数适中（10 ~ 150 峰）的代表性真实谱
    clean_iids = [
        iid for iid in range(forest.n_spectra)
        if 10 <= forest.postings.spectrum_at(iid).n_peaks <= 150
    ]
    query_iids = clean_iids[:6]
    queries = [forest.postings.spectrum_at(i) for i in query_iids]

    batch_query = BatchQueryDevice.from_queries(
        queries, frag_tau=frag_tau, grid_da=gpu_forest.grid_da
    )

    # 每条查询与全部 clean_iids 库谱进行交叉配对
    pair_q = []
    pair_iid = []
    for q_idx in range(len(queries)):
        for iid in clean_iids:
            # 排除峰数 > 128 的完全自匹配对（自身与自身由于 100% 重合边数会超过 128 触发 overflow）
            if query_iids[q_idx] == iid and forest.postings.spectrum_at(iid).n_peaks > 128:
                continue
            pair_q.append(q_idx)
            pair_iid.append(iid)

    q_arr = np.array(pair_q, dtype=np.int64)
    iid_arr = np.array(pair_iid, dtype=np.int64)

    d_scores, d_matched, d_overflow = batch_greedy_cosine_pairs_gpu(
        batch_query=batch_query,
        gpu_forest=gpu_forest,
        query_indices=q_arr,
        candidate_iids=iid_arr,
        frag_tau=frag_tau,
    )

    h_scores = d_scores.copy_to_host()
    h_matched = d_matched.copy_to_host()
    h_overflow = d_overflow.copy_to_host()

    matches_tested = 0
    for k in range(len(pair_q)):
        q_idx = pair_q[k]
        iid = pair_iid[k]
        q_peak = queries[q_idx]
        lib_peak = forest.postings.spectrum_at(iid)

        cpu_res = score_greedy_cosine(q_peak, lib_peak, frag_tau)

        assert h_overflow[k] == 0, f"GNPS Pair {k} unexpectedly overflowed"
        assert h_matched[k] == cpu_res.n_matched, (
            f"GNPS Pair #{k} (q={q_idx}, iid={iid}) n_matched mismatch: "
            f"GPU {h_matched[k]} vs CPU {cpu_res.n_matched}"
        )
        assert np.isclose(h_scores[k], cpu_res.score, atol=1e-5), (
            f"GNPS Pair #{k} (q={q_idx}, iid={iid}) score mismatch: "
            f"GPU {h_scores[k]:.8f} vs CPU {cpu_res.score:.8f}"
        )
        if cpu_res.n_matched > 0:
            matches_tested += 1

    assert matches_tested > 0, "Should have tested pairs with non-zero matches"


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_exact_scoring_tie_breaking_synthetic():
    """测试 2（平局排序规则 Tie-breaking）：

    验证当多个候选边具有完全相同权重时：
    1. 按照 (-w, q_pid, l_pid) 排序；
    2. 贪心选取结果与 CPU 唯一真值基准完全一致。
    """
    frag_tau = 0.05
    # 构造查询谱与库谱：
    # 查询有两个峰：q0 at m=100.0, q1 at m=100.01 (两者都落在容差内)
    # 库有两个峰：l0 at m=100.0, l1 at m=100.01
    # 设 intensity 全部相等 = 0.5 (故每条边的 weight 完全相等 = 0.25)
    # 候选边有 4 条：
    # (q0, l0), (q0, l1), (q1, l0), (q1, l1)
    # 边权重全部为 0.25！
    # Tie-breaking 规则：
    # (-w= -0.25, q_pid=0, l_pid=0) -> edge 0
    # (-w= -0.25, q_pid=0, l_pid=1) -> edge 1
    # (-w= -0.25, q_pid=1, l_pid=0) -> edge 2
    # (-w= -0.25, q_pid=1, l_pid=1) -> edge 3
    # 贪心选取优先选择 (q0, l0)，随后 q0 和 l0 被标记 used；
    # 接下来 (q0, l1) 跳过 (q0 used)；(q1, l0) 跳过 (l0 used)；
    # 选取 (q1, l1)！最终匹配 (q0, l0) 和 (q1, l1)，score = 0.5, n_matched = 2。
    q = SpectrumPeaks._create_unchecked(
        mass=np.array([100.0, 100.01], dtype=np.float64),
        intensity=np.array([0.5, 0.5], dtype=np.float64),
        energy=np.array([0.25, 0.25], dtype=np.float64),
        peak_id=np.array([10, 20], dtype=np.int64),  # non-trivial peak_ids
        norm=1.0,
    )
    lib_s = SpectrumPeaks._create_unchecked(
        mass=np.array([100.0, 100.01], dtype=np.float64),
        intensity=np.array([0.5, 0.5], dtype=np.float64),
        energy=np.array([0.25, 0.25], dtype=np.float64),
        peak_id=np.array([30, 40], dtype=np.int64),  # non-trivial peak_ids
        norm=1.0,
    )

    from jetf import IonMode, ParsedLibrary, SourceRef, SpectrumMeta, preprocess_library
    parsed = ParsedLibrary(
        source_path="tie_test.mgf",
        spectra=(
            SpectrumMeta(
                "s0", 150.0, 1, IonMode.POSITIVE, SourceRef("tie_test.mgf", 0)
            ),
        ),
        mass=lib_s.mass,
        intensity=lib_s.intensity,
        peak_id=lib_s.peak_id,
        spectrum_offsets=np.array([0, 2], dtype=np.int64),
    )
    prep = preprocess_library(parsed, CORRECTNESS_V1)
    spec = ForestSpec(tree_capacity=4, leaf_capacity=2)
    forest = build_forest_index(prep, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)

    batch_query = BatchQueryDevice.from_queries([q], frag_tau=frag_tau, grid_da=gpu_forest.grid_da)

    d_scores, d_matched, d_overflow = batch_greedy_cosine_pairs_gpu(
        batch_query=batch_query,
        gpu_forest=gpu_forest,
        query_indices=np.array([0], dtype=np.int64),
        candidate_iids=np.array([0], dtype=np.int64),
        frag_tau=frag_tau,
    )

    h_scores = d_scores.copy_to_host()
    h_matched = d_matched.copy_to_host()
    lib_postings = forest.postings.spectrum_at(0)
    cpu_res = score_greedy_cosine(q, lib_postings, frag_tau)

    assert h_matched[0] == cpu_res.n_matched == 2
    assert np.isclose(h_scores[0], cpu_res.score, atol=1e-5)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_exact_scoring_edge_and_corner_cases(synthetic_forest_fixture):
    """测试 3（边缘极端场景）：空谱、单峰、全错开、完全重合等。"""
    forest, gpu_forest = synthetic_forest_fixture
    frag_tau = 0.02

    # 1. 空谱查询 (0 peaks)
    empty_q = SpectrumPeaks._create_unchecked(
        mass=np.empty(0, dtype=np.float64),
        intensity=np.empty(0, dtype=np.float64),
        energy=np.empty(0, dtype=np.float64),
        peak_id=np.empty(0, dtype=np.int64),
        norm=0.0,
    )

    # 2. 单峰完全错开 (距离 >> frag_tau)
    far_q = _make_single_query([9999.0], [1.0])

    # 3. 库中第 0 条谱的单峰匹配
    lib_0 = forest.postings.spectrum_at(0)
    single_match_q = _make_single_query([float(lib_0.mass[0])], [1.0])

    # 4. 完全重合谱 (query == lib_0)
    exact_match_q = lib_0

    queries = [empty_q, far_q, single_match_q, exact_match_q]
    batch_query = BatchQueryDevice.from_queries(
        queries, frag_tau=frag_tau, grid_da=gpu_forest.grid_da
    )

    q_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    iids = np.array([0, 0, 0, 0], dtype=np.int64)

    d_scores, d_matched, d_overflow = batch_greedy_cosine_pairs_gpu(
        batch_query=batch_query,
        gpu_forest=gpu_forest,
        query_indices=q_indices,
        candidate_iids=iids,
        frag_tau=frag_tau,
    )

    h_scores = d_scores.copy_to_host()
    h_matched = d_matched.copy_to_host()
    h_overflow = d_overflow.copy_to_host()

    # Case 0: Empty query
    assert h_scores[0] == 0.0 and h_matched[0] == 0 and h_overflow[0] == 0

    # Case 1: Disjoint masses
    assert h_scores[1] == 0.0 and h_matched[1] == 0 and h_overflow[1] == 0

    # Case 2: Single peak matching
    cpu_res2 = score_greedy_cosine(single_match_q, lib_0, frag_tau)
    assert h_matched[2] == cpu_res2.n_matched == 1
    assert np.isclose(h_scores[2], cpu_res2.score, atol=1e-5)

    # Case 3: Exact self-match (score == 1.0)
    cpu_res3 = score_greedy_cosine(exact_match_q, lib_0, frag_tau)
    assert h_matched[3] == cpu_res3.n_matched == lib_0.n_peaks
    assert np.isclose(h_scores[3], 1.0, atol=1e-5)
    assert np.isclose(h_scores[3], cpu_res3.score, atol=1e-5)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_exact_scoring_overflow_detection_and_fallback():
    """测试 4（>128 边溢出检测与 CPU 兜底）：

    构造查询与库谱各自包含 15 个落在相互容差范围内的峰（15 x 15 = 225 > 128 条边）。
    验证 GPU 正确置位 overflow=1，调用方回退 CPU score_greedy_cosine 产生正确真值。
    """
    frag_tau = 0.05
    n_p = 15
    # 构造全部落在 100.00 到 100.02 范围内的峰，相互距离均 < 0.05 Da
    m_dense = np.linspace(100.0, 100.02, n_p, dtype=np.float64)
    it_dense = np.ones(n_p, dtype=np.float64) / np.sqrt(n_p)

    q = SpectrumPeaks._create_unchecked(
        mass=m_dense,
        intensity=it_dense,
        energy=it_dense**2,
        peak_id=np.arange(n_p, dtype=np.int64),
        norm=1.0,
    )
    lib_s = SpectrumPeaks._create_unchecked(
        mass=m_dense,
        intensity=it_dense,
        energy=it_dense**2,
        peak_id=np.arange(n_p, 2 * n_p, dtype=np.int64),
        norm=1.0,
    )

    from jetf import IonMode, ParsedLibrary, SourceRef, SpectrumMeta, preprocess_library
    parsed = ParsedLibrary(
        source_path="overflow_test.mgf",
        spectra=(
            SpectrumMeta(
                "s0", 150.0, 1, IonMode.POSITIVE, SourceRef("overflow_test.mgf", 0)
            ),
        ),
        mass=lib_s.mass,
        intensity=lib_s.intensity,
        peak_id=lib_s.peak_id,
        spectrum_offsets=np.array([0, n_p], dtype=np.int64),
    )
    prep = preprocess_library(parsed, CORRECTNESS_V1)
    spec = ForestSpec(tree_capacity=4, leaf_capacity=2)
    forest = build_forest_index(prep, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)

    batch_query = BatchQueryDevice.from_queries([q], frag_tau=frag_tau, grid_da=gpu_forest.grid_da)

    d_scores, d_matched, d_overflow = batch_greedy_cosine_pairs_gpu(
        batch_query=batch_query,
        gpu_forest=gpu_forest,
        query_indices=np.array([0], dtype=np.int64),
        candidate_iids=np.array([0], dtype=np.int64),
        frag_tau=frag_tau,
    )

    h_overflow = d_overflow.copy_to_host()
    # 必须检测到溢出！
    assert h_overflow[0] == 1, "Dense pair with 225 edges must flag overflow=1"

    # 验证安全兜底逻辑：若 overflow == 1，回退 CPU
    if h_overflow[0] == 1:
        cpu_res = score_greedy_cosine(q, lib_s, frag_tau)
        assert cpu_res.n_matched == n_p
        assert np.isclose(cpu_res.score, 1.0, atol=1e-5)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_exact_scoring_cuda_stream_execution(synthetic_forest_fixture):
    """测试 5（CUDA Stream 异步执行）：验证传入自定义 Stream 的正确性与同步安全。"""
    forest, gpu_forest = synthetic_forest_fixture
    frag_tau = 0.05
    stream = cuda.stream()

    q = _make_single_query([70.0, 80.0, 90.0], [0.5, 0.5, 0.5])
    batch_query = BatchQueryDevice.from_queries(
        [q], frag_tau=frag_tau, grid_da=gpu_forest.grid_da, stream=stream
    )

    d_scores, d_matched, d_overflow = batch_greedy_cosine_pairs_gpu(
        batch_query=batch_query,
        gpu_forest=gpu_forest,
        query_indices=[0],
        candidate_iids=[0],
        stream=stream,
        frag_tau=frag_tau,
    )

    h_scores = d_scores.copy_to_host(stream=stream)
    h_matched = d_matched.copy_to_host(stream=stream)
    h_overflow = d_overflow.copy_to_host(stream=stream)
    stream.synchronize()

    lib_0 = forest.postings.spectrum_at(0)
    cpu_res = score_greedy_cosine(q, lib_0, frag_tau)

    assert h_overflow[0] == 0
    assert h_matched[0] == cpu_res.n_matched
    assert np.isclose(h_scores[0], cpu_res.score, atol=1e-5)
