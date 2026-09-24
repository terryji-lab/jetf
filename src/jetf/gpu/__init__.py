"""JET-Forest GPU acceleration package.

Provides runtime CUDA checks, device data structures, memory resident management,
and high-throughput envelope bounding kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jetf.gpu.device import (
        GpuForestEnvelopes,
        GpuForestIndex,
        GpuForestNodes,
        GpuForestPostings,
        GpuForestTrees,
    )
    from jetf.gpu.bounds_kernels import (
        BatchQueryDevice,
        batch_node_bounds_gpu,
        batch_root_bounds_gpu,
    )
    from jetf.gpu.scoring_kernels import (
        batch_greedy_cosine_pairs_gpu,
        batch_uind_dense_gpu,
        batch_uind_pairs_gpu,
    )
    from jetf.gpu.search import (
        DEFAULT_GPU_SCORE_MARGIN,
        search_forest_batch_gpu,
        search_threshold_batch_gpu,
        search_topk_batch_gpu,
        presort_by_precursor_mz,
    )

__all__ = [
    "DEFAULT_GPU_SCORE_MARGIN",
    "is_cuda_available",
    "require_cuda",
    "GpuForestIndex",
    "BatchQueryDevice",
    "batch_root_bounds_gpu",
    "batch_node_bounds_gpu",
    "batch_uind_dense_gpu",
    "batch_uind_pairs_gpu",
    "batch_greedy_cosine_pairs_gpu",
    "search_threshold_batch_gpu",
    "search_topk_batch_gpu",
    "search_forest_batch_gpu",
    "presort_by_precursor_mz",
]


def is_cuda_available() -> bool:
    """Safe runtime detection of CUDA availability.

    Returns False safely in any environment without CUDA or GPU support.
    """
    try:
        from numba import cuda

        return bool(cuda.is_available() and len(cuda.gpus) > 0)
    except Exception:
        return False


def require_cuda() -> None:
    """Raise RuntimeError with clear troubleshooting guidance if CUDA is unavailable."""
    if not is_cuda_available():
        raise RuntimeError(
            "CUDA is not available or no compatible NVIDIA GPU device was detected. "
            "Please ensure NVIDIA drivers and CUDA toolkit are installed, "
            "and numba.cuda.is_available() evaluates to True."
        )


def __getattr__(name: str) -> Any:
    if name in (
        "GpuForestIndex",
        "GpuForestEnvelopes",
        "GpuForestTrees",
        "GpuForestNodes",
        "GpuForestPostings",
    ):
        from jetf.gpu.device import (
            GpuForestEnvelopes,
            GpuForestIndex,
            GpuForestNodes,
            GpuForestPostings,
            GpuForestTrees,
        )

        mapping = {
            "GpuForestIndex": GpuForestIndex,
            "GpuForestEnvelopes": GpuForestEnvelopes,
            "GpuForestTrees": GpuForestTrees,
            "GpuForestNodes": GpuForestNodes,
            "GpuForestPostings": GpuForestPostings,
        }
        return mapping[name]

    if name in (
        "BatchQueryDevice",
        "batch_root_bounds_gpu",
        "batch_node_bounds_gpu",
    ):
        from jetf.gpu.bounds_kernels import (
            BatchQueryDevice,
            batch_node_bounds_gpu,
            batch_root_bounds_gpu,
        )

        mapping = {
            "BatchQueryDevice": BatchQueryDevice,
            "batch_root_bounds_gpu": batch_root_bounds_gpu,
            "batch_node_bounds_gpu": batch_node_bounds_gpu,
        }
        return mapping[name]

    if name in (
        "batch_greedy_cosine_pairs_gpu",
        "batch_uind_dense_gpu",
        "batch_uind_pairs_gpu",
    ):
        from jetf.gpu.scoring_kernels import (
            batch_greedy_cosine_pairs_gpu,
            batch_uind_dense_gpu,
            batch_uind_pairs_gpu,
        )

        mapping = {
            "batch_greedy_cosine_pairs_gpu": batch_greedy_cosine_pairs_gpu,
            "batch_uind_dense_gpu": batch_uind_dense_gpu,
            "batch_uind_pairs_gpu": batch_uind_pairs_gpu,
        }
        return mapping[name]

    if name in (
        "DEFAULT_GPU_SCORE_MARGIN",
        "search_threshold_batch_gpu",
        "search_topk_batch_gpu",
        "search_forest_batch_gpu",
        "presort_by_precursor_mz",
    ):
        from jetf.gpu.search import (
            DEFAULT_GPU_SCORE_MARGIN,
            presort_by_precursor_mz,
            search_forest_batch_gpu,
            search_threshold_batch_gpu,
            search_topk_batch_gpu,
        )

        mapping = {
            "DEFAULT_GPU_SCORE_MARGIN": DEFAULT_GPU_SCORE_MARGIN,
            "search_threshold_batch_gpu": search_threshold_batch_gpu,
            "search_topk_batch_gpu": search_topk_batch_gpu,
            "search_forest_batch_gpu": search_forest_batch_gpu,
            "presort_by_precursor_mz": presort_by_precursor_mz,
        }
        return mapping[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
