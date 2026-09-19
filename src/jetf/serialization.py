"""ForestIndex 快照序列化与反序列化（基于原子 npz 格式）。"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from jetf.structure import (
    ION_MODE_CODES,
    ION_MODES_BY_CODE,
    ForestEnvelopes,
    ForestIndex,
    ForestNodes,
    ForestPartition,
    ForestPostings,
    ForestSpec,
    ForestTrees,
    ZeroEnergyMembers,
)
from jetf.types import INTERNAL_ID_DTYPE


def save_forest_snapshot(forest: ForestIndex, path: Path | str) -> None:
    """将 ForestIndex 序列化为压缩的 .npz 快照文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    part_codes = np.array([ION_MODE_CODES[p.ion_mode] for p in forest.partitions], dtype=np.int8)
    part_tree_start = np.array([p.tree_start for p in forest.partitions], dtype=INTERNAL_ID_DTYPE)
    part_tree_end = np.array([p.tree_end for p in forest.partitions], dtype=INTERNAL_ID_DTYPE)
    part_node_start = np.array([p.node_start for p in forest.partitions], dtype=INTERNAL_ID_DTYPE)
    part_node_end = np.array([p.node_end for p in forest.partitions], dtype=INTERNAL_ID_DTYPE)
    part_id_start = np.array([p.id_start for p in forest.partitions], dtype=INTERNAL_ID_DTYPE)
    part_id_end = np.array([p.id_end for p in forest.partitions], dtype=INTERNAL_ID_DTYPE)

    np.savez_compressed(
        target,
        spec_versioned_id=str(forest.spec.versioned_id),
        spec_tree_capacity=np.int64(forest.spec.tree_capacity),
        spec_leaf_capacity=np.int64(forest.spec.leaf_capacity),
        spec_max_candidate_axes=np.int64(forest.spec.max_candidate_axes),
        spec_summary_grid_da=np.float64(forest.spec.summary_grid_da),
        n_spectra=np.int64(forest.n_spectra),
        part_codes=part_codes,
        part_tree_start=part_tree_start,
        part_tree_end=part_tree_end,
        part_node_start=part_node_start,
        part_node_end=part_node_end,
        part_id_start=part_id_start,
        part_id_end=part_id_end,
        tree_precursor_min=forest.trees.precursor_min,
        tree_precursor_max=forest.trees.precursor_max,
        tree_root_node_id=forest.trees.root_node_id,
        tree_leaf_offsets=forest.trees.leaf_offsets,
        tree_leaf_node_ids=forest.trees.leaf_node_ids,
        node_is_leaf=forest.nodes.is_leaf,
        node_member_count=forest.nodes.member_count,
        node_id_start=forest.nodes.id_start,
        node_id_end=forest.nodes.id_end,
        node_tree_id=forest.nodes.tree_id,
        node_envelope_offsets=forest.nodes.envelope_offsets,
        env_grid_da=np.float64(forest.envelopes.grid_da),
        env_node_envelope_offsets=forest.envelopes.node_envelope_offsets,
        env_cell_index=forest.envelopes.cell_index,
        env_max_peak_amplitude=forest.envelopes.max_peak_amplitude,
        env_max_cell_energy=forest.envelopes.max_cell_energy,
        post_mass=forest.postings.mass,
        post_intensity=forest.postings.intensity,
        post_energy=forest.postings.energy,
        post_peak_id=forest.postings.peak_id,
        post_spectrum_offsets=forest.postings.spectrum_offsets,
        post_norm=forest.postings.norm,
        internal_to_row=forest.internal_to_row,
        row_to_internal=forest.row_to_internal,
        zero_member=forest.zero_energy_members.member,
        zero_ion_mode=forest.zero_energy_members.ion_mode,
        zero_precursor_mz=forest.zero_energy_members.precursor_mz,
    )


def load_forest_snapshot(path: Path | str) -> ForestIndex:
    """从 .npz 快照文件加载还原 ForestIndex。"""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"快照文件不存在: {target}")

    with np.load(target) as data:
        spec = ForestSpec(
            versioned_id=str(data["spec_versioned_id"]),
            tree_capacity=int(data["spec_tree_capacity"]),
            leaf_capacity=int(data["spec_leaf_capacity"]),
            max_candidate_axes=int(data["spec_max_candidate_axes"]),
            summary_grid_da=float(data["spec_summary_grid_da"]),
        )
        n_spectra = int(data["n_spectra"])

        part_codes = data["part_codes"]
        part_tree_start = data["part_tree_start"]
        part_tree_end = data["part_tree_end"]
        part_node_start = data["part_node_start"]
        part_node_end = data["part_node_end"]
        part_id_start = data["part_id_start"]
        part_id_end = data["part_id_end"]

        partitions = []
        for i in range(len(part_codes)):
            mode = ION_MODES_BY_CODE[int(part_codes[i])]
            partitions.append(
                ForestPartition(
                    ion_mode=mode,
                    tree_start=int(part_tree_start[i]),
                    tree_end=int(part_tree_end[i]),
                    node_start=int(part_node_start[i]),
                    node_end=int(part_node_end[i]),
                    id_start=int(part_id_start[i]),
                    id_end=int(part_id_end[i]),
                )
            )

        trees = ForestTrees(
            precursor_min=data["tree_precursor_min"],
            precursor_max=data["tree_precursor_max"],
            root_node_id=data["tree_root_node_id"],
            leaf_offsets=data["tree_leaf_offsets"],
            leaf_node_ids=data["tree_leaf_node_ids"],
        )

        nodes = ForestNodes(
            is_leaf=data["node_is_leaf"],
            member_count=data["node_member_count"],
            id_start=data["node_id_start"],
            id_end=data["node_id_end"],
            tree_id=data["node_tree_id"],
            envelope_offsets=data["node_envelope_offsets"],
        )

        envelopes = ForestEnvelopes(
            grid_da=float(data["env_grid_da"]),
            node_envelope_offsets=data["env_node_envelope_offsets"],
            cell_index=data["env_cell_index"],
            max_peak_amplitude=data["env_max_peak_amplitude"],
            max_cell_energy=data["env_max_cell_energy"],
        )

        postings = ForestPostings(
            mass=data["post_mass"],
            intensity=data["post_intensity"],
            energy=data["post_energy"],
            peak_id=data["post_peak_id"],
            spectrum_offsets=data["post_spectrum_offsets"],
            norm=data["post_norm"],
        )

        zero_members = ZeroEnergyMembers(
            member=data["zero_member"],
            ion_mode=data["zero_ion_mode"],
            precursor_mz=data["zero_precursor_mz"],
        )

        return ForestIndex(
            spec=spec,
            n_spectra=n_spectra,
            partitions=tuple(partitions),
            trees=trees,
            nodes=nodes,
            envelopes=envelopes,
            postings=postings,
            internal_to_row=data["internal_to_row"],
            row_to_internal=data["row_to_internal"],
            zero_energy_members=zero_members,
        )
