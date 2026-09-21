"""ForestIndex: 前体自适应包络森林列式数据结构与不变量校验。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from jetf.bounds import NodeEnvelope
from jetf.preprocessing import PreprocessedLibrary
from jetf.types import (
    ENERGY_DTYPE,
    INTENSITY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    PEAK_ID_DTYPE,
    IonMode,
    SpectrumMeta,
    SpectrumPeaks,
    check_column,
)


@dataclass(frozen=True)
class ForestSpec:
    """JET-Forest 构建规格参数。"""

    versioned_id: str = "forest/v1"
    tree_capacity: int = 64
    leaf_capacity: int = 16
    max_candidate_axes: int = 10
    summary_grid_da: float = 0.02

    def __post_init__(self) -> None:
        if not self.versioned_id:
            raise ValueError("versioned_id 不能为空")
        if self.tree_capacity < 4:
            raise ValueError(f"tree_capacity 必须 >= 4，得到 {self.tree_capacity}")
        if self.leaf_capacity < 2:
            raise ValueError(f"leaf_capacity 必须 >= 2，得到 {self.leaf_capacity}")
        if self.leaf_capacity > self.tree_capacity:
            raise ValueError(
                f"leaf_capacity ({self.leaf_capacity}) 不能大于 tree_capacity ({self.tree_capacity})"
            )
        if self.summary_grid_da <= 0.0:
            raise ValueError(f"summary_grid_da 必须为正数，得到 {self.summary_grid_da}")


DEFAULT_FOREST_SPEC = ForestSpec()

# 离子模式与内部整型代码映射
ION_MODE_CODES: dict[IonMode, int] = {
    IonMode.POSITIVE: 0,
    IonMode.NEGATIVE: 1,
    IonMode.UNKNOWN: 2,
}
ION_MODES_BY_CODE: tuple[IonMode, ...] = (
    IonMode.POSITIVE,
    IonMode.NEGATIVE,
    IonMode.UNKNOWN,
)


@dataclass(frozen=True, eq=False)
class ZeroEnergyMembers:
    """零能量谱隔离侧车（不进树索引）。"""

    member: NDArray[np.int64]
    ion_mode: NDArray[np.int8]
    precursor_mz: NDArray[np.float64]

    def __post_init__(self) -> None:
        check_column("member", self.member, INTERNAL_ID_DTYPE)
        check_column("ion_mode", self.ion_mode, np.dtype(np.int8))
        check_column("precursor_mz", self.precursor_mz, MASS_DTYPE)
        if not (self.member.shape == self.ion_mode.shape == self.precursor_mz.shape):
            raise ValueError("ZeroEnergyMembers 列长度必须一致")

    @property
    def n_members(self) -> int:
        return int(self.member.shape[0])

    @classmethod
    def from_library(cls, library: PreprocessedLibrary, zero_rows: NDArray[np.int64]) -> ZeroEnergyMembers:
        modes = np.array(
            [ION_MODE_CODES[library.spectra[int(r)].ion_mode] for r in zero_rows], dtype=np.int8
        )
        precursors = np.array(
            [
                library.spectra[int(r)].precursor_mz
                if library.spectra[int(r)].precursor_mz is not None
                else np.nan
                for r in zero_rows
            ],
            dtype=MASS_DTYPE,
        )
        return cls(member=zero_rows, ion_mode=modes, precursor_mz=precursors)


@dataclass(frozen=True, eq=False)
class ForestPartition:
    """单个离子模式硬分区在森林中的区间划分。"""

    ion_mode: IonMode
    tree_start: int
    tree_end: int
    node_start: int
    node_end: int
    id_start: int
    id_end: int

    @property
    def n_trees(self) -> int:
        return self.tree_end - self.tree_start

    @property
    def n_nodes(self) -> int:
        return self.node_end - self.node_start

    @property
    def n_spectra(self) -> int:
        return self.id_end - self.id_start


@dataclass(frozen=True, eq=False)
class ForestTrees:
    """森林小树的列式表示（长度 M 为树的总数）。"""

    precursor_min: NDArray[np.float64]
    precursor_max: NDArray[np.float64]
    root_node_id: NDArray[np.int64]
    leaf_offsets: NDArray[np.int64]  # 长度 M + 1，切分 leaf_node_ids
    leaf_node_ids: NDArray[np.int64]  # 全部叶节点的全局 node_id 平铺

    def __post_init__(self) -> None:
        n_trees = self.precursor_min.shape[0]
        check_column("precursor_min", self.precursor_min, MASS_DTYPE)
        check_column("precursor_max", self.precursor_max, MASS_DTYPE)
        check_column("root_node_id", self.root_node_id, INTERNAL_ID_DTYPE)
        check_column("leaf_offsets", self.leaf_offsets, INTERNAL_ID_DTYPE)
        check_column("leaf_node_ids", self.leaf_node_ids, INTERNAL_ID_DTYPE)

        if self.precursor_max.shape[0] != n_trees or self.root_node_id.shape[0] != n_trees:
            raise ValueError("ForestTrees 前体与根节点列长度必须与树数一致")
        if self.leaf_offsets.shape[0] != n_trees + 1:
            raise ValueError("ForestTrees leaf_offsets 长度必须为 n_trees + 1")

    @property
    def n_trees(self) -> int:
        return int(self.precursor_min.shape[0])

    def leaves_of_tree(self, tree_id: int) -> NDArray[np.int64]:
        start = int(self.leaf_offsets[tree_id])
        end = int(self.leaf_offsets[tree_id + 1])
        return self.leaf_node_ids[start:end]


@dataclass(frozen=True, eq=False)
class ForestNodes:
    """森林全部节点（包括根节点与叶节点）的列式表示。"""

    is_leaf: NDArray[np.bool_]
    member_count: NDArray[np.int64]
    id_start: NDArray[np.int64]
    id_end: NDArray[np.int64]
    tree_id: NDArray[np.int64]

    def __post_init__(self) -> None:
        n_nodes = self.is_leaf.shape[0]
        check_column("member_count", self.member_count, INTERNAL_ID_DTYPE)
        check_column("id_start", self.id_start, INTERNAL_ID_DTYPE)
        check_column("id_end", self.id_end, INTERNAL_ID_DTYPE)
        check_column("tree_id", self.tree_id, INTERNAL_ID_DTYPE)

        if self.member_count.shape[0] != n_nodes:
            raise ValueError("ForestNodes 列长度不匹配")

    @property
    def n_nodes(self) -> int:
        return int(self.is_leaf.shape[0])


@dataclass(frozen=True, eq=False)
class ForestEnvelopes:
    """全森林节点的平铺包络摘要。"""

    grid_da: float
    node_envelope_offsets: NDArray[np.int64]
    cell_index: NDArray[np.int64]
    max_peak_amplitude: NDArray[np.float64]

    def __post_init__(self) -> None:
        check_column("node_envelope_offsets", self.node_envelope_offsets, INTERNAL_ID_DTYPE)
        check_column("cell_index", self.cell_index, INTERNAL_ID_DTYPE)
        check_column("max_peak_amplitude", self.max_peak_amplitude, ENERGY_DTYPE)

    @property
    def n_nodes(self) -> int:
        return int(self.node_envelope_offsets.shape[0]) - 1

    def span(self, node_id: int) -> tuple[int, int]:
        if not 0 <= node_id < self.n_nodes:
            raise IndexError(f"节点序号越界: {node_id} (共 {self.n_nodes} 节点)")
        return int(self.node_envelope_offsets[node_id]), int(self.node_envelope_offsets[node_id + 1])

    def envelope_of(self, node_id: int) -> NodeEnvelope:
        start, stop = self.span(node_id)
        return NodeEnvelope._create_unchecked(
            grid_da=self.grid_da,
            cell_index=self.cell_index[start:stop],
            max_peak_amplitude=self.max_peak_amplitude[start:stop],
        )


@dataclass(frozen=True, eq=False)
class ForestPostings:
    """按微块（叶节点）对齐重排后的连续峰数据。"""

    mass: NDArray[np.float64]
    intensity: NDArray[np.float64]
    energy: NDArray[np.float64]
    peak_id: NDArray[np.int64]
    spectrum_offsets: NDArray[np.int64]
    norm: NDArray[np.float64]

    def __post_init__(self) -> None:
        check_column("mass", self.mass, MASS_DTYPE)
        check_column("intensity", self.intensity, INTENSITY_DTYPE)
        check_column("energy", self.energy, ENERGY_DTYPE)
        check_column("peak_id", self.peak_id, PEAK_ID_DTYPE)
        check_column("spectrum_offsets", self.spectrum_offsets, INTERNAL_ID_DTYPE)
        check_column("norm", self.norm, ENERGY_DTYPE)

    @property
    def n_spectra(self) -> int:
        return int(self.spectrum_offsets.shape[0]) - 1

    def spectrum_at(self, internal_id: int) -> SpectrumPeaks:
        if internal_id < 0 or internal_id >= self.n_spectra:
            raise IndexError(f"internal_id {internal_id} 超出合法区间 [0, {self.n_spectra})")
        start = int(self.spectrum_offsets[internal_id])
        stop = int(self.spectrum_offsets[internal_id + 1])
        return SpectrumPeaks._create_unchecked(
            mass=self.mass[start:stop],
            intensity=self.intensity[start:stop],
            energy=self.energy[start:stop],
            peak_id=self.peak_id[start:stop],
            norm=float(self.norm[internal_id]),
        )


@dataclass(frozen=True, eq=False)
class ForestIndex:
    """全库包络森林索引。"""

    spec: ForestSpec
    n_spectra: int
    partitions: tuple[ForestPartition, ...]
    trees: ForestTrees
    nodes: ForestNodes
    envelopes: ForestEnvelopes
    postings: ForestPostings
    internal_to_row: NDArray[np.int64]
    row_to_internal: NDArray[np.int64]
    zero_energy_members: ZeroEnergyMembers
    library_fingerprint: str = ""
    spectra: tuple[SpectrumMeta, ...] | None = None

    @property
    def n_trees(self) -> int:
        return self.trees.n_trees

    @property
    def n_nodes(self) -> int:
        return self.nodes.n_nodes

    def envelope_of(self, node_id: int) -> NodeEnvelope:
        return self.envelopes.envelope_of(node_id)

    def rows_of(self, node_id: int) -> NDArray[np.int64]:
        start = int(self.nodes.id_start[node_id])
        end = int(self.nodes.id_end[node_id])
        return self.internal_to_row[start:end]


def check_forest_index(forest: ForestIndex, library: PreprocessedLibrary) -> None:
    """校验 ForestIndex 的全局结构不变量。"""
    if forest.n_spectra != library.n_spectra:
        raise ValueError(
            f"森林总谱数不一致: 索引声明 {forest.n_spectra}, 库实际 {library.n_spectra}"
        )

    if abs(forest.spec.summary_grid_da - forest.envelopes.grid_da) > 1e-9:
        raise ValueError(
            f"森林索引网格不一致: spec.summary_grid_da={forest.spec.summary_grid_da} Da, "
            f"envelopes.grid_da={forest.envelopes.grid_da} Da"
        )


    # 1. 验证零分侧车与有能量谱总和
    n_zero = forest.zero_energy_members.n_members
    n_indexed = len(forest.internal_to_row)
    if n_zero + n_indexed != library.n_spectra:
        raise ValueError(
            f"零分侧车 ({n_zero}) + 森林成员 ({n_indexed}) 不等于总谱数 ({library.n_spectra})"
        )

    # 2. 验证 internal_to_row 和 row_to_internal 互反
    for iid, row in enumerate(forest.internal_to_row):
        if forest.row_to_internal[row] != iid:
            raise ValueError(f"双向映射不一致: internal_id {iid} -> row {row}")

    # 3. 验证每棵树的前体单调区间
    for t_id in range(forest.n_trees):
        root_id = int(forest.trees.root_node_id[t_id])
        rows = forest.rows_of(root_id)
        precursors = [
            library.spectra[r].precursor_mz
            for r in rows
            if library.spectra[r].precursor_mz is not None
        ]
        if precursors:
            p_min = min(precursors)
            p_max = max(precursors)
            if forest.trees.precursor_min[t_id] > p_min + 1e-9:
                raise ValueError(f"树 {t_id} precursor_min 低估真实前体")
            if forest.trees.precursor_max[t_id] < p_max - 1e-9:
                raise ValueError(f"树 {t_id} precursor_max 高估真实前体")

        # 验证该树下叶节点的连续并集覆盖
        leaf_ids = forest.trees.leaves_of_tree(t_id)
        leaf_id_starts = [int(forest.nodes.id_start[lid]) for lid in leaf_ids]
        leaf_id_ends = [int(forest.nodes.id_end[lid]) for lid in leaf_ids]
        root_start = int(forest.nodes.id_start[root_id])
        root_end = int(forest.nodes.id_end[root_id])

        if leaf_id_starts[0] != root_start:
            raise ValueError(f"树 {t_id} 首叶起点不等于根节点起点")
        if leaf_id_ends[-1] != root_end:
            raise ValueError(f"树 {t_id} 尾叶终点不等于根节点终点")
        for j in range(len(leaf_ids) - 1):
            if leaf_id_ends[j] != leaf_id_starts[j + 1]:
                raise ValueError(f"树 {t_id} 叶节点 internal_id 区间不连续")
