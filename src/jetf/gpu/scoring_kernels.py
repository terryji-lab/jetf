"""GPU single spectrum upper bound U_ind evaluation kernels (K3a).

Implements single-spectrum relaxation upper bounds (U_ind) on GPU with pure
register execution (zero cuda.local.array allocation) and guaranteed zero false dismissals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
import numpy as np
from numpy.typing import NDArray
from numba import cuda, float32, int16, int64, uint64

from jetf.gpu import require_cuda

if TYPE_CHECKING:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from jetf.gpu.bounds_kernels import BatchQueryDevice
    from jetf.gpu.device import GpuForestIndex

# FP32 安全膨胀因子 (1000 ppm / 0.1%)，确保单精度累加由于舍入或重排序绝不低于 CPU 基准值
SAFETY_MARGIN_FP32 = np.float32(1.0 + 1e-3)


@cuda.jit(device=True)
def _eval_uind_bound_device(
    q_start: int,
    q_end: int,
    lib_start: int,
    lib_end: int,
    q_mass: DeviceNDArray,
    q_intensity: DeviceNDArray,
    lib_mass: DeviceNDArray,
    lib_intensity: DeviceNDArray,
    frag_tau: float,
) -> float:
    """Device-only single-spectrum upper bound evaluation in pure registers.

    Strictly register-only execution: zero dynamic allocations, zero cuda.local.array.

    Args:
        q_start: Start index of query peaks in batch q_mass/q_intensity.
        q_end: End index of query peaks in batch q_mass/q_intensity.
        lib_start: Start index of library spectrum peaks in postings.
        lib_end: End index of library spectrum peaks in postings.
        q_mass: Global batch query mass array (float64).
        q_intensity: Global batch query intensity array (float32).
        lib_mass: Postings library peak mass array (float64).
        lib_intensity: Postings library peak intensity array (float32).
        frag_tau: Fragment tolerance in Da (float64).

    Returns:
        float32 single-spectrum upper bound with safety margin inflation.
    """
    if q_start >= q_end or lib_start >= lib_end:
        return np.float32(0.0)

    # 边界短路：若库谱最高质量 < 查询谱最低质量 - frag_tau 或 库谱最低质量 > 查询谱最高质量 + frag_tau
    if (
        lib_mass[lib_end - 1] < q_mass[q_start] - frag_tau
        or lib_mass[lib_start] > q_mass[q_end - 1] + frag_tau
    ):
        return np.float32(0.0)

    s = np.float32(0.0)
    has_overlap = False
    k_left = lib_start
    for p in range(q_start, q_end):
        c_low = q_mass[p] - frag_tau
        c_high = q_mass[p] + frag_tau

        # 二分查找：定位 lib_mass[k_left:lib_end] 中 >= c_low 的最左侧位置
        low = k_left
        high = lib_end
        while low < high:
            mid = (low + high) >> 1
            if lib_mass[mid] < c_low:
                low = mid + 1
            else:
                high = mid
        k_left = low
        if k_left >= lib_end:
            break

        # 二分查找：定位 lib_mass[k_left:lib_end] 中 > c_high 的最左侧位置
        low = k_left
        high = lib_end
        while low < high:
            mid = (low + high) >> 1
            if lib_mass[mid] <= c_high:
                low = mid + 1
            else:
                high = mid
        k_right = low

        if k_right > k_left:
            has_overlap = True
            m_val = lib_intensity[k_left]
            for k in range(k_left + 1, k_right):
                val = lib_intensity[k]
                if val > m_val:
                    m_val = val
            s += q_intensity[p] * m_val

    # 下溢保护：若存在至少一对满足容差窗口的重叠峰，但强度极小相乘在 FP32 下溢为 0.0 时，
    # 赋予保底极小值 1e-7，防止上界被误归零而导致候选谱被错误剪枝。
    if has_overlap and s == np.float32(0.0):
        s = np.float32(1e-7)

    if s > np.float32(0.0):
        return s * SAFETY_MARGIN_FP32
    return np.float32(0.0)


@cuda.jit
def _k3a_uind_dense_kernel(
    candidate_iids: DeviceNDArray,
    lib_offsets: DeviceNDArray,
    lib_mass: DeviceNDArray,
    lib_intensity: DeviceNDArray,
    q_offsets: DeviceNDArray,
    q_mass: DeviceNDArray,
    q_intensity: DeviceNDArray,
    frag_tau: float,
    n_queries: int,
    n_candidates: int,
    n_spectra: int,
    out_bounds: DeviceNDArray,
) -> None:
    """K3a Single-Spectrum Upper Bound Dense CUDA Kernel.

    Flattened 1D grid over n_queries * n_candidate_spectra tasks.
    """
    tid = cuda.grid(1)
    n_tasks = n_queries * n_candidates
    if tid < n_tasks:
        q_idx = tid // n_candidates
        c_pos = tid % n_candidates
        iid = candidate_iids[c_pos]

        if iid < 0 or iid >= n_spectra:
            out_bounds[q_idx, c_pos] = np.float32(0.0)
            return

        q_start = q_offsets[q_idx]
        q_end = q_offsets[q_idx + 1]

        lib_start = lib_offsets[iid]
        lib_end = lib_offsets[iid + 1]

        out_bounds[q_idx, c_pos] = _eval_uind_bound_device(
            q_start,
            q_end,
            lib_start,
            lib_end,
            q_mass,
            q_intensity,
            lib_mass,
            lib_intensity,
            frag_tau,
        )


@cuda.jit
def _k3a_uind_pairs_kernel(
    query_indices: DeviceNDArray,
    candidate_iids: DeviceNDArray,
    lib_offsets: DeviceNDArray,
    lib_mass: DeviceNDArray,
    lib_intensity: DeviceNDArray,
    q_offsets: DeviceNDArray,
    q_mass: DeviceNDArray,
    q_intensity: DeviceNDArray,
    frag_tau: float,
    n_pairs: int,
    n_queries: int,
    n_spectra: int,
    out_bounds: DeviceNDArray,
) -> None:
    """K3a Single-Spectrum Upper Bound Pairs CUDA Kernel.

    Flattened 1D grid over n_pairs tasks.
    """
    tid = cuda.grid(1)
    if tid < n_pairs:
        q_idx = query_indices[tid]
        iid = candidate_iids[tid]

        if q_idx < 0 or q_idx >= n_queries or iid < 0 or iid >= n_spectra:
            out_bounds[tid] = np.float32(0.0)
            return

        q_start = q_offsets[q_idx]
        q_end = q_offsets[q_idx + 1]

        lib_start = lib_offsets[iid]
        lib_end = lib_offsets[iid + 1]

        out_bounds[tid] = _eval_uind_bound_device(
            q_start,
            q_end,
            lib_start,
            lib_end,
            q_mass,
            q_intensity,
            lib_mass,
            lib_intensity,
            frag_tau,
        )


def batch_uind_dense_gpu(
    batch_query: BatchQueryDevice,
    gpu_forest: GpuForestIndex,
    candidate_iids: DeviceNDArray | NDArray | Sequence[int] | None = None,
    stream: Any = None,
    frag_tau: float | None = None,
) -> DeviceNDArray:
    """Evaluate batch single-spectrum upper bounds U_ind on GPU in dense mode.

    Args:
        batch_query: Device-resident packed queries. Must include q_mass.
        gpu_forest: Device-resident forest index containing postings table.
        candidate_iids: Candidate spectrum internal IDs (iids).
            If None, evaluates all library spectra in gpu_forest.
        stream: Optional CUDA stream for asynchronous execution.
        frag_tau: Fragment tolerance in Da. If None, defaults to batch_query.frag_tau.

    Returns:
        DeviceNDArray of shape (n_queries, n_candidates) with float32 upper bounds.
    """
    require_cuda()
    if batch_query.q_mass is None:
        raise ValueError(
            "BatchQueryDevice must have q_mass populated for single-spectrum bound evaluation."
        )

    n_queries = batch_query.n_queries
    if candidate_iids is None:
        n_candidates = gpu_forest.n_spectra
        d_candidate_iids = getattr(gpu_forest, "_d_all_spectrum_iids", None)
        if d_candidate_iids is None or d_candidate_iids.shape[0] != n_candidates:
            d_candidate_iids = cuda.to_device(
                np.arange(n_candidates, dtype=np.int64), stream=stream
            )
            setattr(gpu_forest, "_d_all_spectrum_iids", d_candidate_iids)
    elif hasattr(candidate_iids, "__cuda_array_interface__"):
        d_candidate_iids = candidate_iids
        n_candidates = int(d_candidate_iids.shape[0])
    else:
        candidate_iids_arr = np.ascontiguousarray(candidate_iids, dtype=np.int64)
        n_candidates = int(candidate_iids_arr.shape[0])
        d_candidate_iids = cuda.to_device(candidate_iids_arr, stream=stream)

    out_bounds = cuda.device_array(
        (n_queries, n_candidates), dtype=np.float32, stream=stream
    )
    n_tasks = n_queries * n_candidates
    if n_tasks == 0:
        return out_bounds

    tau = float(frag_tau if frag_tau is not None else batch_query.frag_tau)
    n_spectra = gpu_forest.n_spectra

    threads_per_block = 256
    blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block

    _k3a_uind_dense_kernel[blocks_per_grid, threads_per_block, stream](
        d_candidate_iids,
        gpu_forest.spectrum_offsets,
        gpu_forest.mass,
        gpu_forest.intensity,
        batch_query.q_offsets,
        batch_query.q_mass,
        batch_query.q_intensity,
        tau,
        n_queries,
        n_candidates,
        n_spectra,
        out_bounds,
    )
    return out_bounds


def batch_uind_pairs_gpu(
    batch_query: BatchQueryDevice,
    gpu_forest: GpuForestIndex,
    query_indices: DeviceNDArray | NDArray | Sequence[int],
    candidate_iids: DeviceNDArray | NDArray | Sequence[int],
    stream: Any = None,
    frag_tau: float | None = None,
) -> DeviceNDArray:
    """Evaluate batch single-spectrum upper bounds U_ind on GPU for explicit query-candidate pairs.

    Args:
        batch_query: Device-resident packed queries. Must include q_mass.
        gpu_forest: Device-resident forest index containing postings table.
        query_indices: Query indices corresponding to each pair.
        candidate_iids: Candidate spectrum internal IDs (iids) corresponding to each pair.
        stream: Optional CUDA stream for asynchronous execution.
        frag_tau: Fragment tolerance in Da. If None, defaults to batch_query.frag_tau.

    Returns:
        DeviceNDArray of shape (n_pairs,) with float32 upper bounds.
    """
    require_cuda()
    if batch_query.q_mass is None:
        raise ValueError(
            "BatchQueryDevice must have q_mass populated for single-spectrum bound evaluation."
        )

    if hasattr(query_indices, "__cuda_array_interface__"):
        d_query_indices = query_indices
        n_pairs = int(d_query_indices.shape[0])
    else:
        query_indices_arr = np.ascontiguousarray(query_indices, dtype=np.int64)
        n_pairs = int(query_indices_arr.shape[0])
        d_query_indices = cuda.to_device(query_indices_arr, stream=stream)

    if hasattr(candidate_iids, "__cuda_array_interface__"):
        d_candidate_iids = candidate_iids
        if int(d_candidate_iids.shape[0]) != n_pairs:
            raise ValueError(
                f"query_indices length ({n_pairs}) does not match candidate_iids length ({d_candidate_iids.shape[0]})"
            )
    else:
        candidate_iids_arr = np.ascontiguousarray(candidate_iids, dtype=np.int64)
        if int(candidate_iids_arr.shape[0]) != n_pairs:
            raise ValueError(
                f"query_indices length ({n_pairs}) does not match candidate_iids length ({candidate_iids_arr.shape[0]})"
            )
        d_candidate_iids = cuda.to_device(candidate_iids_arr, stream=stream)

    out_bounds = cuda.device_array((n_pairs,), dtype=np.float32, stream=stream)
    if n_pairs == 0:
        return out_bounds

    tau = float(frag_tau if frag_tau is not None else batch_query.frag_tau)
    n_queries = batch_query.n_queries
    n_spectra = gpu_forest.n_spectra

    threads_per_block = 256
    blocks_per_grid = (n_pairs + threads_per_block - 1) // threads_per_block

    _k3a_uind_pairs_kernel[blocks_per_grid, threads_per_block, stream](
        d_query_indices,
        d_candidate_iids,
        gpu_forest.spectrum_offsets,
        gpu_forest.mass,
        gpu_forest.intensity,
        batch_query.q_offsets,
        batch_query.q_mass,
        batch_query.q_intensity,
        tau,
        n_pairs,
        n_queries,
        n_spectra,
        out_bounds,
    )
    return out_bounds


@cuda.jit(device=True)
def _is_better_edge(
    w_a: float,
    q_pid_a: int,
    l_pid_a: int,
    w_b: float,
    q_pid_b: int,
    l_pid_b: int,
) -> bool:
    """Compare two candidate edges according to deterministic priority (-w, q_pid, l_pid).

    Matches CPU tie-breaking: np.lexsort((lib_pid, query_pid, -weights)).
    """
    if w_a > w_b:
        return True
    if w_a < w_b:
        return False
    if q_pid_a < q_pid_b:
        return True
    if q_pid_a > q_pid_b:
        return False
    return l_pid_a < l_pid_b


@cuda.jit
def _k3_greedy_cosine_pairs_kernel(
    query_indices: DeviceNDArray,
    candidate_iids: DeviceNDArray,
    lib_offsets: DeviceNDArray,
    lib_mass: DeviceNDArray,
    lib_intensity: DeviceNDArray,
    lib_peak_id: DeviceNDArray,
    q_offsets: DeviceNDArray,
    q_mass: DeviceNDArray,
    q_intensity: DeviceNDArray,
    q_peak_id: DeviceNDArray,
    frag_tau: float,
    n_pairs: int,
    n_queries: int,
    n_spectra: int,
    out_scores: DeviceNDArray,
    out_matched: DeviceNDArray,
    out_overflow: DeviceNDArray,
) -> None:
    """K3 Greedy Cosine Exact Scoring Pairs CUDA Kernel.

    Flattened 1D grid over n_pairs tasks.
    Evaluates exact deterministic greedy cosine similarity with one-to-one
    matching and tie-breaking (-w, q_pid, l_pid).
    """
    tid = cuda.grid(1)
    if tid < n_pairs:
        out_scores[tid] = float32(0.0)
        out_matched[tid] = 0
        out_overflow[tid] = 0

        q_idx = query_indices[tid]
        iid = candidate_iids[tid]

        if q_idx < 0 or q_idx >= n_queries or iid < 0 or iid >= n_spectra:
            return

        q_start = q_offsets[q_idx]
        q_end = q_offsets[q_idx + 1]

        lib_start = lib_offsets[iid]
        lib_end = lib_offsets[iid + 1]

        n_q = q_end - q_start
        n_l = lib_end - lib_start

        if n_q <= 0 or n_l <= 0:
            return

        # Bitmask array has size 8 x uint64 = 512 bits per spectrum.
        # If spectrum peak count exceeds 512, flag overflow to fallback to CPU.
        if n_q > 512 or n_l > 512:
            out_overflow[tid] = 1
            return

        if (
            lib_mass[lib_end - 1] < q_mass[q_start] - frag_tau
            or lib_mass[lib_start] > q_mass[q_end - 1] + frag_tau
        ):
            return

        # 3. Local fixed-size arrays for candidate edges
        MAX_EDGES = 128
        edge_w = cuda.local.array(128, dtype=float32)
        edge_q = cuda.local.array(128, dtype=int16)
        edge_l = cuda.local.array(128, dtype=int16)
        edge_q_pid = cuda.local.array(128, dtype=int64)
        edge_l_pid = cuda.local.array(128, dtype=int64)

        # Bitmasks for greedy matching (8 x uint64 covers 512 peaks)
        used_q = cuda.local.array(8, dtype=uint64)
        used_l = cuda.local.array(8, dtype=uint64)
        for b in range(8):
            used_q[b] = uint64(0)
            used_l[b] = uint64(0)

        # 4. Binary search range lookup and edge collection
        n_edges = 0
        k_left = lib_start

        for p in range(q_start, q_end):
            qm = q_mass[p]
            qi = q_intensity[p]
            if qi <= float32(0.0):
                continue
            c_low = qm - frag_tau
            c_high = qm + frag_tau

            low = k_left
            high = lib_end
            while low < high:
                mid = (low + high) >> 1
                if lib_mass[mid] < c_low:
                    low = mid + 1
                else:
                    high = mid
            k_left = low
            if k_left >= lib_end:
                break

            low = k_left
            high = lib_end
            while low < high:
                mid = (low + high) >> 1
                if lib_mass[mid] <= c_high:
                    low = mid + 1
                else:
                    high = mid
            k_right = low

            rel_q = int16(p - q_start)
            qp = q_peak_id[p]

            for j in range(k_left, k_right):
                li = lib_intensity[j]
                w = qi * li
                if w > float32(0.0):
                    if n_edges >= MAX_EDGES:
                        out_overflow[tid] = 1
                        return
                    edge_w[n_edges] = w
                    edge_q[n_edges] = rel_q
                    edge_l[n_edges] = int16(j - lib_start)
                    edge_q_pid[n_edges] = qp
                    edge_l_pid[n_edges] = lib_peak_id[j]
                    n_edges += 1

        if n_edges == 0:
            return

        # 5. Classic in-place insertion sort (priority: -w, q_pid, l_pid)
        for i in range(1, n_edges):
            cur_w = edge_w[i]
            cur_q = edge_q[i]
            cur_l = edge_l[i]
            cur_qp = edge_q_pid[i]
            cur_lp = edge_l_pid[i]

            j = i - 1
            while j >= 0 and _is_better_edge(
                cur_w, cur_qp, cur_lp, edge_w[j], edge_q_pid[j], edge_l_pid[j]
            ):
                edge_w[j + 1] = edge_w[j]
                edge_q[j + 1] = edge_q[j]
                edge_l[j + 1] = edge_l[j]
                edge_q_pid[j + 1] = edge_q_pid[j]
                edge_l_pid[j + 1] = edge_l_pid[j]
                j -= 1

            edge_w[j + 1] = cur_w
            edge_q[j + 1] = cur_q
            edge_l[j + 1] = cur_l
            edge_q_pid[j + 1] = cur_qp
            edge_l_pid[j + 1] = cur_lp

        # 6. Greedy selection using bitmasks
        score = float32(0.0)
        n_matched = 0

        for e in range(n_edges):
            rq = int(edge_q[e])
            rl = int(edge_l[e])

            wq = rq >> 6
            bq = uint64(1) << uint64(rq & 63)

            wl = rl >> 6
            bl = uint64(1) << uint64(rl & 63)

            if (used_q[wq] & bq) == uint64(0) and (used_l[wl] & bl) == uint64(0):
                used_q[wq] |= bq
                used_l[wl] |= bl
                score += edge_w[e]
                n_matched += 1

        # 7. Normalization and truncation
        # 浮点舍入容差钳制：GPU 使用 FP32 单精度累加，机器精度约为 ~1.19e-7。
        # 累加后理论 1.0 分数可能在 [1.0 - 1e-6, 1.0 + 1e-6] 波动，统一钳制为 1.0 (与 CPU 端的 FP64 1e-12 截断带逻辑统一对应)
        if score > float32(1.0) or abs(score - float32(1.0)) <= float32(1e-6):
            score = float32(1.0)

        out_scores[tid] = score
        out_matched[tid] = n_matched
        out_overflow[tid] = 0


def batch_greedy_cosine_pairs_gpu(
    batch_query: BatchQueryDevice,
    gpu_forest: GpuForestIndex,
    query_indices: DeviceNDArray | NDArray | Sequence[int],
    candidate_iids: DeviceNDArray | NDArray | Sequence[int],
    stream: Any = None,
    frag_tau: float | None = None,
) -> tuple[DeviceNDArray, DeviceNDArray, DeviceNDArray]:
    """Evaluate batch Greedy Cosine exact scores on GPU for query-candidate pairs.

    Args:
        batch_query: Device-resident packed queries. Must include q_mass and q_peak_id.
        gpu_forest: Device-resident forest index containing postings table.
        query_indices: Query indices corresponding to each candidate pair.
        candidate_iids: Candidate spectrum internal IDs (iids) corresponding to each pair.
        stream: Optional CUDA stream for asynchronous execution.
        frag_tau: Fragment tolerance in Da. If None, defaults to batch_query.frag_tau.

    Returns:
        Tuple of (d_scores, d_matched, d_overflow):
            - d_scores: DeviceNDArray of shape (n_pairs,) with float32 scores.
            - d_matched: DeviceNDArray of shape (n_pairs,) with int32 matched peak counts.
            - d_overflow: DeviceNDArray of shape (n_pairs,) with int8 overflow flags (1 if >128 edges or >512 peaks).
    """
    require_cuda()
    if batch_query.q_mass is None:
        raise ValueError(
            "BatchQueryDevice must have q_mass populated for exact scoring evaluation."
        )
    if batch_query.q_peak_id is None:
        raise ValueError(
            "BatchQueryDevice must have q_peak_id populated for exact scoring evaluation."
        )

    if hasattr(query_indices, "__cuda_array_interface__"):
        d_query_indices = query_indices
        n_pairs = int(d_query_indices.shape[0])
    else:
        query_indices_arr = np.ascontiguousarray(query_indices, dtype=np.int64)
        n_pairs = int(query_indices_arr.shape[0])
        d_query_indices = cuda.to_device(query_indices_arr, stream=stream)

    if hasattr(candidate_iids, "__cuda_array_interface__"):
        d_candidate_iids = candidate_iids
        if int(d_candidate_iids.shape[0]) != n_pairs:
            raise ValueError(
                f"query_indices length ({n_pairs}) does not match candidate_iids length ({d_candidate_iids.shape[0]})"
            )
    else:
        candidate_iids_arr = np.ascontiguousarray(candidate_iids, dtype=np.int64)
        if int(candidate_iids_arr.shape[0]) != n_pairs:
            raise ValueError(
                f"query_indices length ({n_pairs}) does not match candidate_iids length ({candidate_iids_arr.shape[0]})"
            )
        d_candidate_iids = cuda.to_device(candidate_iids_arr, stream=stream)

    out_scores = cuda.device_array((n_pairs,), dtype=np.float32, stream=stream)
    out_matched = cuda.device_array((n_pairs,), dtype=np.int32, stream=stream)
    out_overflow = cuda.device_array((n_pairs,), dtype=np.int8, stream=stream)

    if n_pairs == 0:
        return out_scores, out_matched, out_overflow

    tau = float(frag_tau if frag_tau is not None else batch_query.frag_tau)
    n_queries = batch_query.n_queries
    n_spectra = gpu_forest.n_spectra

    threads_per_block = 128
    blocks_per_grid = (n_pairs + threads_per_block - 1) // threads_per_block

    _k3_greedy_cosine_pairs_kernel[blocks_per_grid, threads_per_block, stream](
        d_query_indices,
        d_candidate_iids,
        gpu_forest.spectrum_offsets,
        gpu_forest.mass,
        gpu_forest.intensity,
        gpu_forest.peak_id,
        batch_query.q_offsets,
        batch_query.q_mass,
        batch_query.q_intensity,
        batch_query.q_peak_id,
        tau,
        n_pairs,
        n_queries,
        n_spectra,
        out_scores,
        out_matched,
        out_overflow,
    )
    return out_scores, out_matched, out_overflow

