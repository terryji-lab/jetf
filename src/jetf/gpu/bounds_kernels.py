"""GPU resident batch query structures and envelope bounds kernels (K1 & K2).

Implements single-precision GPU upper bound evaluations for tree roots (K1)
and candidate/leaf nodes (K2) with guaranteed zero false dismissals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence
import numpy as np
from numpy.typing import NDArray
from numba import cuda

from jetf.bounds import window_cells
from jetf.gpu import require_cuda

if TYPE_CHECKING:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from jetf.gpu.device import GpuForestIndex
    from jetf.types import SpectrumPeaks

# FP32 安全膨胀因子 (1000 ppm / 0.1%)，确保单精度累加由于舍入或重排序导致的极微小差异绝不低于 CPU 基准值
SAFETY_MARGIN_FP32 = np.float32(1.0 + 1e-3)


@dataclass(frozen=True, eq=False)
class BatchQueryDevice:
    """Device-resident batch of spectrum peak queries.

    Stores flattened contiguous arrays for intensities and window cells
    along with offset pointers for each query.
    """

    q_offsets: DeviceNDArray
    q_intensity: DeviceNDArray
    q_cell_lower: DeviceNDArray
    q_cell_upper: DeviceNDArray
    q_mass: DeviceNDArray | None = None
    q_peak_id: DeviceNDArray | None = None
    frag_tau: float = 0.0
    grid_da: float = 0.0

    @property
    def n_queries(self) -> int:
        """Total number of queries in the batch."""
        return int(self.q_offsets.shape[0]) - 1

    @property
    def total_peaks(self) -> int:
        """Total number of peaks across all queries."""
        return int(self.q_intensity.shape[0])

    def memory_bytes(self) -> int:
        """Total GPU device memory consumed by the batch query in bytes."""
        mem = (
            int(self.q_offsets.nbytes)
            + int(self.q_intensity.nbytes)
            + int(self.q_cell_lower.nbytes)
            + int(self.q_cell_upper.nbytes)
        )
        if self.q_mass is not None:
            mem += int(self.q_mass.nbytes)
        if self.q_peak_id is not None:
            mem += int(self.q_peak_id.nbytes)
        return mem

    @classmethod
    def from_queries(
        cls,
        queries: Sequence[SpectrumPeaks],
        frag_tau: float,
        grid_da: float,
        stream: Any = None,
    ) -> BatchQueryDevice:
        """Pack a sequence of queries into device-resident contiguous memory.

        Pre-allocates flat host buffers, computes window cells via bounds.window_cells,
        and performs asynchronous/synchronous host-to-device transfers.
        """
        require_cuda()
        n_queries = len(queries)
        if n_queries == 0:
            d_offsets = cuda.to_device(np.zeros(1, dtype=np.int64), stream=stream)
            d_intensity = cuda.to_device(np.empty(0, dtype=np.float32), stream=stream)
            d_cell_lower = cuda.to_device(np.empty(0, dtype=np.int64), stream=stream)
            d_cell_upper = cuda.to_device(np.empty(0, dtype=np.int64), stream=stream)
            d_mass = cuda.to_device(np.empty(0, dtype=np.float64), stream=stream)
            d_peak_id = cuda.to_device(np.empty(0, dtype=np.int64), stream=stream)
            return cls(
                q_offsets=d_offsets,
                q_intensity=d_intensity,
                q_cell_lower=d_cell_lower,
                q_cell_upper=d_cell_upper,
                q_mass=d_mass,
                q_peak_id=d_peak_id,
                frag_tau=float(frag_tau),
                grid_da=float(grid_da),
            )

        peak_counts = [q.n_peaks for q in queries]
        total_peaks = sum(peak_counts)

        q_offsets_np = np.empty(n_queries + 1, dtype=np.int64)
        q_offsets_np[0] = 0
        np.cumsum(peak_counts, out=q_offsets_np[1:])

        q_mass_np = np.empty(total_peaks, dtype=np.float64)
        q_intensity_np = np.empty(total_peaks, dtype=np.float32)
        q_cell_lower_np = np.empty(total_peaks, dtype=np.int64)
        q_cell_upper_np = np.empty(total_peaks, dtype=np.int64)
        q_peak_id_np = np.empty(total_peaks, dtype=np.int64)

        for i, q in enumerate(queries):
            if q.n_peaks > 1 and not np.all(q.mass[:-1] <= q.mass[1:]):
                raise ValueError("Query peaks mass must be sorted in ascending order")
            start = q_offsets_np[i]
            end = q_offsets_np[i + 1]
            if start < end:
                q_mass_np[start:end] = q.mass.astype(np.float64)
                q_intensity_np[start:end] = q.intensity.astype(np.float32)
                q_peak_id_np[start:end] = q.peak_id.astype(np.int64)
                low, high = window_cells(q.mass, frag_tau, grid_da)
                q_cell_lower_np[start:end] = low
                q_cell_upper_np[start:end] = high

        d_offsets = cuda.to_device(q_offsets_np, stream=stream)
        d_mass = cuda.to_device(q_mass_np, stream=stream)
        d_intensity = cuda.to_device(q_intensity_np, stream=stream)
        d_cell_lower = cuda.to_device(q_cell_lower_np, stream=stream)
        d_cell_upper = cuda.to_device(q_cell_upper_np, stream=stream)
        d_peak_id = cuda.to_device(q_peak_id_np, stream=stream)

        return cls(
            q_offsets=d_offsets,
            q_intensity=d_intensity,
            q_cell_lower=d_cell_lower,
            q_cell_upper=d_cell_upper,
            q_mass=d_mass,
            q_peak_id=d_peak_id,
            frag_tau=float(frag_tau),
            grid_da=float(grid_da),
        )

    def __repr__(self) -> str:
        kb = self.memory_bytes() / 1024
        return (
            f"<BatchQueryDevice queries={self.n_queries}, total_peaks={self.total_peaks}, "
            f"gpu_mem={kb:.2f}KB>"
        )


@cuda.jit(device=True)
def _eval_node_bound_device(
    node_id: int,
    env_offsets: DeviceNDArray,
    cell_index: DeviceNDArray,
    max_peak_amplitude: DeviceNDArray,
    q_start: int,
    q_end: int,
    q_cell_lower: DeviceNDArray,
    q_cell_upper: DeviceNDArray,
    q_intensity: DeviceNDArray,
) -> float:
    """Evaluate envelope peak upper bound for a single node and query on device."""
    n_peaks = q_end - q_start
    if n_peaks <= 0:
        return np.float32(0.0)

    n_nodes = env_offsets.shape[0] - 1
    if node_id < 0 or node_id >= n_nodes:
        return np.float32(0.0)

    start = env_offsets[node_id]
    end = env_offsets[node_id + 1]
    if start >= end:
        return np.float32(0.0)

    # 1. 快速包络盒相交过滤：若节点最大 cell < 查询最小 cell 或节点最小 cell > 查询最大 cell，直接短路
    if cell_index[end - 1] < q_cell_lower[q_start] or cell_index[start] > q_cell_upper[q_end - 1]:
        return np.float32(0.0)

    s = np.float32(0.0)
    has_overlap = False
    k_left = start
    for p in range(q_start, q_end):
        c_low = q_cell_lower[p]
        c_high = q_cell_upper[p]

        # 单调二分：query 峰质量升序，故 >= c_low 的起点单调不减
        low = k_left
        high = end
        while low < high:
            mid = (low + high) >> 1
            if cell_index[mid] < c_low:
                low = mid + 1
            else:
                high = mid
        k_left = low
        if k_left >= end:
            break

        # 二分查找：定位 cell_index[k_left:end] 中 > c_high 的最左侧位置
        low = k_left
        high = end
        while low < high:
            mid = (low + high) >> 1
            if cell_index[mid] <= c_high:
                low = mid + 1
            else:
                high = mid
        k_right = low

        if k_right > k_left:
            has_overlap = True
            m_val = max_peak_amplitude[k_left]
            for k in range(k_left + 1, k_right):
                val = max_peak_amplitude[k]
                if val > m_val:
                    m_val = val
            s += q_intensity[p] * m_val

    # 下溢保护：若存在至少一对满足容差窗口的重叠峰，但强度极小相乘在 FP32 下溢为 0.0 时，
    # 赋予保底极小值 1e-7，防止上界被误归零而导致树节点或候选谱被错误剪枝。
    if has_overlap and s == np.float32(0.0):
        s = np.float32(1e-7)

    if s > np.float32(0.0):
        return s * SAFETY_MARGIN_FP32
    return np.float32(0.0)


@cuda.jit
def _k1_root_bounds_kernel(
    tree_ids: DeviceNDArray,
    root_node_ids: DeviceNDArray,
    env_offsets: DeviceNDArray,
    cell_index: DeviceNDArray,
    max_peak_amplitude: DeviceNDArray,
    q_offsets: DeviceNDArray,
    q_intensity: DeviceNDArray,
    q_cell_lower: DeviceNDArray,
    q_cell_upper: DeviceNDArray,
    n_queries: int,
    n_trees: int,
    out_bounds: DeviceNDArray,
) -> None:
    """K1 Root Upper Bound CUDA Kernel.

    Flattened 1D grid over n_queries * n_trees tasks.
    """
    tid = cuda.grid(1)
    n_tasks = n_queries * n_trees
    if tid < n_tasks:
        q_idx = tid // n_trees
        t_pos = tid % n_trees
        t_id = tree_ids[t_pos]

        if t_id < 0 or t_id >= root_node_ids.shape[0]:
            out_bounds[q_idx, t_pos] = np.float32(0.0)
            return

        root_id = root_node_ids[t_id]
        q_start = q_offsets[q_idx]
        q_end = q_offsets[q_idx + 1]

        out_bounds[q_idx, t_pos] = _eval_node_bound_device(
            root_id,
            env_offsets,
            cell_index,
            max_peak_amplitude,
            q_start,
            q_end,
            q_cell_lower,
            q_cell_upper,
            q_intensity,
        )


@cuda.jit
def _k2_leaf_bounds_kernel(
    candidate_node_ids: DeviceNDArray,
    env_offsets: DeviceNDArray,
    cell_index: DeviceNDArray,
    max_peak_amplitude: DeviceNDArray,
    q_offsets: DeviceNDArray,
    q_intensity: DeviceNDArray,
    q_cell_lower: DeviceNDArray,
    q_cell_upper: DeviceNDArray,
    n_queries: int,
    n_candidate_nodes: int,
    out_bounds: DeviceNDArray,
) -> None:
    """K2 Candidate/Leaf Upper Bound CUDA Kernel.

    Flattened 1D grid over n_queries * n_candidate_nodes tasks.
    """
    tid = cuda.grid(1)
    n_tasks = n_queries * n_candidate_nodes
    if tid < n_tasks:
        q_idx = tid // n_candidate_nodes
        node_pos = tid % n_candidate_nodes
        node_id = candidate_node_ids[node_pos]

        q_start = q_offsets[q_idx]
        q_end = q_offsets[q_idx + 1]

        out_bounds[q_idx, node_pos] = _eval_node_bound_device(
            node_id,
            env_offsets,
            cell_index,
            max_peak_amplitude,
            q_start,
            q_end,
            q_cell_lower,
            q_cell_upper,
            q_intensity,
        )


def batch_root_bounds_gpu(
    batch_query: BatchQueryDevice,
    gpu_forest: GpuForestIndex,
    tree_ids: DeviceNDArray | NDArray | Sequence[int] | None = None,
    stream: Any = None,
) -> DeviceNDArray:
    """Evaluate batch root bounds on GPU for given queries and trees.

    Args:
        batch_query: Device-resident packed queries.
        gpu_forest: Device-resident forest index.
        tree_ids: Optional tree IDs to evaluate. If None, evaluates all n_trees.
        stream: Optional CUDA stream for asynchronous execution.

    Returns:
        DeviceNDArray of shape (n_queries, n_trees) with float32 upper bounds.
    """
    require_cuda()

    n_queries = batch_query.n_queries
    if tree_ids is None:
        n_trees = gpu_forest.n_trees
        d_tree_ids = getattr(gpu_forest, "_d_all_tree_ids", None)
        if d_tree_ids is None or d_tree_ids.shape[0] != n_trees:
            d_tree_ids = cuda.to_device(np.arange(n_trees, dtype=np.int64), stream=stream)
            setattr(gpu_forest, "_d_all_tree_ids", d_tree_ids)
    elif hasattr(tree_ids, "__cuda_array_interface__"):
        d_tree_ids = tree_ids
        n_trees = int(d_tree_ids.shape[0])
    else:
        tree_ids_arr = np.ascontiguousarray(tree_ids, dtype=np.int64)
        n_trees = int(tree_ids_arr.shape[0])
        d_tree_ids = cuda.to_device(tree_ids_arr, stream=stream)

    out_bounds = cuda.device_array((n_queries, n_trees), dtype=np.float32, stream=stream)
    n_tasks = n_queries * n_trees
    if n_tasks == 0:
        return out_bounds

    threads_per_block = 256
    blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block

    _k1_root_bounds_kernel[blocks_per_grid, threads_per_block, stream](
        d_tree_ids,
        gpu_forest.root_node_id,
        gpu_forest.node_envelope_offsets,
        gpu_forest.cell_index,
        gpu_forest.max_peak_amplitude,
        batch_query.q_offsets,
        batch_query.q_intensity,
        batch_query.q_cell_lower,
        batch_query.q_cell_upper,
        n_queries,
        n_trees,
        out_bounds,
    )
    return out_bounds


def batch_node_bounds_gpu(
    batch_query: BatchQueryDevice,
    gpu_forest: GpuForestIndex,
    node_ids: DeviceNDArray | NDArray | Sequence[int],
    stream: Any = None,
) -> DeviceNDArray:
    """Evaluate batch candidate/leaf node bounds on GPU for given queries and nodes.

    Args:
        batch_query: Device-resident packed queries.
        gpu_forest: Device-resident forest index.
        node_ids: Node IDs to evaluate (e.g. leaf nodes).
        stream: Optional CUDA stream for asynchronous execution.

    Returns:
        DeviceNDArray of shape (n_queries, n_nodes) with float32 upper bounds.
    """
    require_cuda()

    n_queries = batch_query.n_queries
    if hasattr(node_ids, "__cuda_array_interface__"):
        d_node_ids = node_ids
        n_nodes = int(d_node_ids.shape[0])
    else:
        node_ids_arr = np.ascontiguousarray(node_ids, dtype=np.int64)
        n_nodes = int(node_ids_arr.shape[0])
        d_node_ids = cuda.to_device(node_ids_arr, stream=stream)

    out_bounds = cuda.device_array((n_queries, n_nodes), dtype=np.float32, stream=stream)
    n_tasks = n_queries * n_nodes
    if n_tasks == 0:
        return out_bounds

    threads_per_block = 256
    blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block

    _k2_leaf_bounds_kernel[blocks_per_grid, threads_per_block, stream](
        d_node_ids,
        gpu_forest.node_envelope_offsets,
        gpu_forest.cell_index,
        gpu_forest.max_peak_amplitude,
        batch_query.q_offsets,
        batch_query.q_intensity,
        batch_query.q_cell_lower,
        batch_query.q_cell_upper,
        n_queries,
        n_nodes,
        out_bounds,
    )
    return out_bounds
