"""GPU resident data structures and device memory migration for JET-Forest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import numpy as np

from jetf.gpu import require_cuda

if TYPE_CHECKING:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from jetf.structure import ForestIndex


@dataclass(frozen=True, eq=False)
class GpuForestEnvelopes:
    """Device-resident node envelope summaries."""

    grid_da: float
    node_envelope_offsets: DeviceNDArray
    cell_index: DeviceNDArray
    max_peak_amplitude: DeviceNDArray

    @property
    def n_nodes(self) -> int:
        return int(self.node_envelope_offsets.shape[0]) - 1

    def memory_bytes(self) -> int:
        return (
            int(self.node_envelope_offsets.nbytes)
            + int(self.cell_index.nbytes)
            + int(self.max_peak_amplitude.nbytes)
        )


@dataclass(frozen=True, eq=False)
class GpuForestTrees:
    """Device-resident forest tree columnar index."""

    precursor_min: DeviceNDArray
    precursor_max: DeviceNDArray
    root_node_id: DeviceNDArray
    leaf_offsets: DeviceNDArray
    leaf_node_ids: DeviceNDArray

    @property
    def n_trees(self) -> int:
        return int(self.precursor_min.shape[0])

    def memory_bytes(self) -> int:
        return (
            int(self.precursor_min.nbytes)
            + int(self.precursor_max.nbytes)
            + int(self.root_node_id.nbytes)
            + int(self.leaf_offsets.nbytes)
            + int(self.leaf_node_ids.nbytes)
        )


@dataclass(frozen=True, eq=False)
class GpuForestNodes:
    """Device-resident forest node descriptors."""

    id_start: DeviceNDArray
    id_end: DeviceNDArray
    is_leaf: DeviceNDArray
    tree_id: DeviceNDArray

    @property
    def n_nodes(self) -> int:
        return int(self.id_start.shape[0])

    def memory_bytes(self) -> int:
        return (
            int(self.id_start.nbytes)
            + int(self.id_end.nbytes)
            + int(self.is_leaf.nbytes)
            + int(self.tree_id.nbytes)
        )


@dataclass(frozen=True, eq=False)
class GpuForestPostings:
    """Device-resident peak postings table.

    Note: The 'energy' column is intentionally NOT transferred to device memory
    to conserve GPU VRAM (~0.78 GB savings on full GNPS scale).
    """

    mass: DeviceNDArray
    intensity: DeviceNDArray
    peak_id: DeviceNDArray
    spectrum_offsets: DeviceNDArray
    norm: DeviceNDArray

    @property
    def n_spectra(self) -> int:
        return int(self.spectrum_offsets.shape[0]) - 1

    def memory_bytes(self) -> int:
        return (
            int(self.mass.nbytes)
            + int(self.intensity.nbytes)
            + int(self.peak_id.nbytes)
            + int(self.spectrum_offsets.nbytes)
            + int(self.norm.nbytes)
        )


class GpuForestIndex:
    """GPU-resident JET-Forest index holding device arrays in VRAM."""

    def __init__(
        self,
        forest: ForestIndex,
        envelopes: GpuForestEnvelopes,
        trees: GpuForestTrees,
        nodes: GpuForestNodes,
        postings: GpuForestPostings,
    ) -> None:
        self.forest = forest
        self.envelopes = envelopes
        self.trees = trees
        self.nodes = nodes
        self.postings = postings

    # Convenient shortcuts for envelopes
    @property
    def cell_index(self) -> DeviceNDArray:
        return self.envelopes.cell_index

    @property
    def max_peak_amplitude(self) -> DeviceNDArray:
        return self.envelopes.max_peak_amplitude

    @property
    def node_envelope_offsets(self) -> DeviceNDArray:
        return self.envelopes.node_envelope_offsets

    @property
    def grid_da(self) -> float:
        return self.envelopes.grid_da

    # Convenient shortcuts for trees
    @property
    def precursor_min(self) -> DeviceNDArray:
        return self.trees.precursor_min

    @property
    def precursor_max(self) -> DeviceNDArray:
        return self.trees.precursor_max

    @property
    def root_node_id(self) -> DeviceNDArray:
        return self.trees.root_node_id

    @property
    def leaf_offsets(self) -> DeviceNDArray:
        return self.trees.leaf_offsets

    @property
    def leaf_node_ids(self) -> DeviceNDArray:
        return self.trees.leaf_node_ids

    # Convenient shortcuts for nodes
    @property
    def id_start(self) -> DeviceNDArray:
        return self.nodes.id_start

    @property
    def id_end(self) -> DeviceNDArray:
        return self.nodes.id_end

    @property
    def is_leaf(self) -> DeviceNDArray:
        return self.nodes.is_leaf

    @property
    def tree_id(self) -> DeviceNDArray:
        return self.nodes.tree_id

    # Convenient shortcuts for postings
    @property
    def mass(self) -> DeviceNDArray:
        return self.postings.mass

    @property
    def intensity(self) -> DeviceNDArray:
        return self.postings.intensity

    @property
    def peak_id(self) -> DeviceNDArray:
        return self.postings.peak_id

    @property
    def spectrum_offsets(self) -> DeviceNDArray:
        return self.postings.spectrum_offsets

    @property
    def norm(self) -> DeviceNDArray:
        return self.postings.norm

    # Global index convenience properties
    @property
    def n_trees(self) -> int:
        return self.trees.n_trees

    @property
    def n_nodes(self) -> int:
        return self.nodes.n_nodes

    @property
    def n_spectra(self) -> int:
        return self.postings.n_spectra

    @property
    def spec(self) -> Any:
        return self.forest.spec

    @property
    def partitions(self) -> Any:
        return self.forest.partitions

    def device_memory_summary(self) -> dict[str, int]:
        """Return memory breakdown in bytes across sub-components."""
        return {
            "envelopes": self.envelopes.memory_bytes(),
            "trees": self.trees.memory_bytes(),
            "nodes": self.nodes.memory_bytes(),
            "postings": self.postings.memory_bytes(),
        }

    def device_memory_bytes(self) -> int:
        """Return total allocated device memory in bytes."""
        return sum(self.device_memory_summary().values())

    @classmethod
    def from_forest(cls, forest: ForestIndex, stream: Any = None) -> GpuForestIndex:
        """Upload a host ForestIndex into GPU device memory.

        Converts max_peak_amplitude and intensity to float32 for high-throughput
        evaluation, while completely omitting the postings.energy column to save VRAM.
        """
        require_cuda()
        from numba import cuda

        # 1. Envelopes
        d_cell_index = cuda.to_device(
            np.ascontiguousarray(forest.envelopes.cell_index, dtype=np.int64),
            stream=stream,
        )
        d_max_peak_amplitude = cuda.to_device(
            np.ascontiguousarray(forest.envelopes.max_peak_amplitude, dtype=np.float32),
            stream=stream,
        )
        d_node_envelope_offsets = cuda.to_device(
            np.ascontiguousarray(forest.envelopes.node_envelope_offsets, dtype=np.int64),
            stream=stream,
        )
        gpu_envelopes = GpuForestEnvelopes(
            grid_da=float(forest.envelopes.grid_da),
            node_envelope_offsets=d_node_envelope_offsets,
            cell_index=d_cell_index,
            max_peak_amplitude=d_max_peak_amplitude,
        )

        # 2. Trees
        d_precursor_min = cuda.to_device(
            np.ascontiguousarray(forest.trees.precursor_min, dtype=np.float64),
            stream=stream,
        )
        d_precursor_max = cuda.to_device(
            np.ascontiguousarray(forest.trees.precursor_max, dtype=np.float64),
            stream=stream,
        )
        d_root_node_id = cuda.to_device(
            np.ascontiguousarray(forest.trees.root_node_id, dtype=np.int64),
            stream=stream,
        )
        d_leaf_offsets = cuda.to_device(
            np.ascontiguousarray(forest.trees.leaf_offsets, dtype=np.int64),
            stream=stream,
        )
        d_leaf_node_ids = cuda.to_device(
            np.ascontiguousarray(forest.trees.leaf_node_ids, dtype=np.int64),
            stream=stream,
        )
        gpu_trees = GpuForestTrees(
            precursor_min=d_precursor_min,
            precursor_max=d_precursor_max,
            root_node_id=d_root_node_id,
            leaf_offsets=d_leaf_offsets,
            leaf_node_ids=d_leaf_node_ids,
        )

        # 3. Nodes
        d_id_start = cuda.to_device(
            np.ascontiguousarray(forest.nodes.id_start, dtype=np.int64),
            stream=stream,
        )
        d_id_end = cuda.to_device(
            np.ascontiguousarray(forest.nodes.id_end, dtype=np.int64),
            stream=stream,
        )
        d_is_leaf = cuda.to_device(
            np.ascontiguousarray(forest.nodes.is_leaf, dtype=np.bool_),
            stream=stream,
        )
        d_tree_id = cuda.to_device(
            np.ascontiguousarray(forest.nodes.tree_id, dtype=np.int64),
            stream=stream,
        )
        gpu_nodes = GpuForestNodes(
            id_start=d_id_start,
            id_end=d_id_end,
            is_leaf=d_is_leaf,
            tree_id=d_tree_id,
        )

        # 4. Postings (Note: energy is strictly NOT copied to device!)
        d_mass = cuda.to_device(
            np.ascontiguousarray(forest.postings.mass),
            stream=stream,
        )
        d_intensity = cuda.to_device(
            np.ascontiguousarray(forest.postings.intensity, dtype=np.float32),
            stream=stream,
        )
        d_peak_id = cuda.to_device(
            np.ascontiguousarray(forest.postings.peak_id),
            stream=stream,
        )
        d_spectrum_offsets = cuda.to_device(
            np.ascontiguousarray(forest.postings.spectrum_offsets, dtype=np.int64),
            stream=stream,
        )
        d_norm = cuda.to_device(
            np.ascontiguousarray(forest.postings.norm, dtype=np.float64),
            stream=stream,
        )
        gpu_postings = GpuForestPostings(
            mass=d_mass,
            intensity=d_intensity,
            peak_id=d_peak_id,
            spectrum_offsets=d_spectrum_offsets,
            norm=d_norm,
        )

        return cls(
            forest=forest,
            envelopes=gpu_envelopes,
            trees=gpu_trees,
            nodes=gpu_nodes,
            postings=gpu_postings,
        )

    def __repr__(self) -> str:
        mb = self.device_memory_bytes() / (1024 * 1024)
        return (
            f"<GpuForestIndex trees={self.n_trees}, nodes={self.n_nodes}, "
            f"spectra={self.n_spectra}, gpu_mem={mb:.2f}MB>"
        )
