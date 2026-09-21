"""ForestIndex 构建器：从 PreprocessedLibrary 编译前体包络森林索引。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from jetf.bvh_sah import split_sah_bvh
from jetf.preprocessing import PreprocessedLibrary
from jetf.scoring import inflate_upper_bounds
from jetf.structure import (
    DEFAULT_FOREST_SPEC,
    ION_MODES_BY_CODE,
    ForestEnvelopes,
    ForestIndex,
    ForestNodes,
    ForestPartition,
    ForestPostings,
    ForestSpec,
    ForestTrees,
    ZeroEnergyMembers,
    check_forest_index,
)
from jetf.types import (
    ENERGY_DTYPE,
    INTENSITY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    PEAK_ID_DTYPE,
    IonMode,
)


def _row_positions(offsets: NDArray[np.int64], rows: Sequence[int]) -> NDArray[np.int64]:
    parts = [
        np.arange(int(offsets[row]), int(offsets[row + 1]), dtype=INTERNAL_ID_DTYPE)
        for row in rows
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=INTERNAL_ID_DTYPE)


def _member_peaks(
    library: PreprocessedLibrary, rows: Sequence[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    peaks = library.peaks
    positions = _row_positions(peaks.spectrum_offsets, rows)
    return peaks.mass[positions], peaks.intensity[positions], peaks.energy[positions]


def _peak_maxima(
    library: PreprocessedLibrary, rows: Sequence[int], grid_da: float
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    mass, amplitude, energy = _member_peaks(library, rows)
    keep = energy > 0.0
    if not np.any(keep):
        return np.empty(0, dtype=INTERNAL_ID_DTYPE), np.empty(0, dtype=ENERGY_DTYPE)
    cells = np.floor((mass[keep] + 1e-12) / grid_da).astype(INTERNAL_ID_DTYPE)
    support, inverse = np.unique(cells, return_inverse=True)
    maxima = np.zeros(support.shape[0], dtype=ENERGY_DTYPE)
    np.maximum.at(maxima, inverse, amplitude[keep])
    return support, maxima


def _extract_bucket_cells_and_amps(
    library: PreprocessedLibrary, bucket: Sequence[int]
) -> tuple[dict[int, set[int]], dict[int, dict[int, float]]]:
    """提取单棵小树内各谱 (<=64 条) 的 0.02 Da non-zero cell 集合与最大单峰幅度字典（按需局部提取，极度轻量）。"""
    res = library.resources
    fine = library.peaks
    grid_da = res.grid_da

    bucket_cells: dict[int, set[int]] = {}
    bucket_amps: dict[int, dict[int, float]] = {}

    for row in bucket:
        start = int(res.spectrum_offsets[row])
        end = int(res.spectrum_offsets[row + 1])
        bucket_cells[row] = set(res.cell_index[start:end])

        p_start = int(fine.spectrum_offsets[row])
        p_end = int(fine.spectrum_offsets[row + 1])
        amps: dict[int, float] = {}
        for m, a in zip(fine.mass[p_start:p_end], fine.intensity[p_start:p_end]):
            c = int(np.floor((m + 1e-12) / grid_da))
            if c not in amps or a > amps[c]:
                amps[c] = float(a)
        bucket_amps[row] = amps

    return bucket_cells, bucket_amps


def _build_single_leaf_envelope(
    library: PreprocessedLibrary, rows: Sequence[int], grid_da: float
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """从叶成员行构建叶包络（支持集、最大单峰幅度），上偏一次。"""
    support, amplitudes = _peak_maxima(library, rows, grid_da)
    return support, inflate_upper_bounds(amplitudes)


def _merge_root_envelope(
    leaf_envelopes: Sequence[tuple[NDArray[np.int64], NDArray[np.float64]]],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """将若干子叶包络逐坐标取 max 合并为根包络（不上偏，直接继承子包络上偏）。"""
    all_cells = np.unique(np.concatenate([e[0] for e in leaf_envelopes]))
    root_m = np.zeros(all_cells.shape[0], dtype=ENERGY_DTYPE)

    for supp, m in leaf_envelopes:
        idx = np.searchsorted(all_cells, supp)
        np.maximum.at(root_m, idx, m)

    return all_cells, root_m


def build_forest_index(
    library: PreprocessedLibrary,
    spec: ForestSpec = DEFAULT_FOREST_SPEC,
) -> ForestIndex:
    """从 PreprocessedLibrary 构建 JET-Forest 列式森林索引。"""
    if library.spec.grid_da != spec.summary_grid_da:
        raise ValueError(
            f"网格配置不一致: 预处理网格为 {library.spec.grid_da} Da, "
            f"森林索引网格为 {spec.summary_grid_da} Da"
        )

    grid_da = spec.summary_grid_da

    # 1. 分离零能量谱与有能量谱
    zero_rows: list[int] = []
    active_rows: list[int] = []

    res = library.resources
    for row in range(library.n_spectra):
        start = int(res.spectrum_offsets[row])
        end = int(res.spectrum_offsets[row + 1])
        if start == end:
            zero_rows.append(row)
        else:
            active_rows.append(row)

    zero_energy_members = ZeroEnergyMembers.from_library(
        library, np.array(zero_rows, dtype=INTERNAL_ID_DTYPE)
    )

    # 2. 按离子模式硬分区，模式内部按前体质量严格升序排序
    partitions: list[ForestPartition] = []

    tree_prec_min: list[float] = []
    tree_prec_max: list[float] = []
    tree_root_node_ids: list[int] = []
    tree_leaf_offsets: list[int] = [0]
    tree_leaf_node_ids: list[int] = []

    node_is_leaf: list[bool] = []
    node_member_count: list[int] = []
    node_id_start: list[int] = []
    node_id_end: list[int] = []
    node_tree_id: list[int] = []

    env_cells: list[NDArray[np.int64]] = []
    env_m: list[NDArray[np.float64]] = []

    internal_to_row: list[int] = []
    row_to_internal: NDArray[np.int64] = np.full(library.n_spectra, -1, dtype=INTERNAL_ID_DTYPE)

    current_internal_id = 0
    current_node_id = 0
    current_tree_id = 0

    for ion_mode in ION_MODES_BY_CODE:
        mode_rows = [r for r in active_rows if library.spectra[r].ion_mode == ion_mode]

        # 排序：有限前体升序；缺失前体置后；外部 ID 破并列
        def sort_key(row_idx: int) -> tuple[int, float, str, int]:
            p = library.spectra[row_idx].precursor_mz
            has_prec = 0 if (p is not None and np.isfinite(p)) else 1
            prec_val = p if (p is not None and np.isfinite(p)) else 0.0
            ext_id = library.spectra[row_idx].external_id
            return (has_prec, prec_val, ext_id, row_idx)

        mode_rows.sort(key=sort_key)

        part_tree_start = current_tree_id
        part_node_start = current_node_id
        part_id_start = current_internal_id

        # 切分为若干小树 (每树最多 tree_capacity 谱)
        for i in range(0, len(mode_rows), spec.tree_capacity):
            bucket = mode_rows[i : i + spec.tree_capacity]
            t_id = current_tree_id
            current_tree_id += 1

            prec_list = [
                library.spectra[r].precursor_mz
                for r in bucket
                if library.spectra[r].precursor_mz is not None
                and np.isfinite(library.spectra[r].precursor_mz)
            ]
            if prec_list:
                p_min = float(min(prec_list))
                p_max = float(max(prec_list))
            else:
                p_min = float(np.nan)
                p_max = float(np.nan)

            tree_prec_min.append(p_min)
            tree_prec_max.append(p_max)

            # 在 bucket 内按需局部提取 SAH 所需 cell 与 amplitude 结构 (单树 <= 64 谱，仅几 KB 内存)
            bucket_cells, bucket_amps = _extract_bucket_cells_and_amps(library, bucket)

            # 在 bucket 内调用 SAH 切分生成叶子
            leaves = split_sah_bvh(
                bucket,
                bucket_cells,
                bucket_amps,
                target_leaf_size=spec.leaf_capacity,
                max_candidate_axes=spec.max_candidate_axes,
            )

            leaf_start_id = current_internal_id
            leaf_info: list[tuple[int, int, int, tuple[NDArray[np.int64], NDArray[np.float64]]]] = []
            leaf_envs: list[tuple[NDArray[np.int64], NDArray[np.float64]]] = []

            for leaf_rows in leaves:
                l_start = current_internal_id
                l_count = len(leaf_rows)
                current_internal_id += l_count
                l_end = current_internal_id

                for row in leaf_rows:
                    internal_to_row.append(row)
                    row_to_internal[row] = len(internal_to_row) - 1

                supp, m_arr = _build_single_leaf_envelope(library, leaf_rows, grid_da)
                env_tuple = (supp, m_arr)
                leaf_envs.append(env_tuple)
                leaf_info.append((l_start, l_end, l_count, env_tuple))

            leaf_end_id = current_internal_id

            # 编译根包络
            root_supp, root_m = _merge_root_envelope(leaf_envs)

            # 追加根节点
            root_id = current_node_id
            current_node_id += 1
            tree_root_node_ids.append(root_id)

            node_is_leaf.append(False)
            node_member_count.append(len(bucket))
            node_id_start.append(leaf_start_id)
            node_id_end.append(leaf_end_id)
            node_tree_id.append(t_id)

            env_cells.append(root_supp)
            env_m.append(root_m)

            # 追加叶节点
            for l_start, l_end, l_count, (supp, m_arr) in leaf_info:
                lid = current_node_id
                current_node_id += 1
                tree_leaf_node_ids.append(lid)

                node_is_leaf.append(True)
                node_member_count.append(l_count)
                node_id_start.append(l_start)
                node_id_end.append(l_end)
                node_tree_id.append(t_id)

                env_cells.append(supp)
                env_m.append(m_arr)

            tree_leaf_offsets.append(len(tree_leaf_node_ids))

        part_tree_end = current_tree_id
        part_node_end = current_node_id
        part_id_end = current_internal_id

        partitions.append(
            ForestPartition(
                ion_mode=ion_mode,
                tree_start=part_tree_start,
                tree_end=part_tree_end,
                node_start=part_node_start,
                node_end=part_node_end,
                id_start=part_id_start,
                id_end=part_id_end,
            )
        )

    # 4. 连续平铺 envelope_offsets
    flat_env_offsets = [0]
    for c_arr in env_cells:
        flat_env_offsets.append(flat_env_offsets[-1] + len(c_arr))

    # 5. 组装连续峰缓冲 ForestPostings (预分配内存，避免数百万切片对象与拼接翻倍开销)
    fine = library.peaks
    n_internal = len(internal_to_row)
    if n_internal > 0:
        row_arr = np.array(internal_to_row, dtype=INTERNAL_ID_DTYPE)
        lens = fine.spectrum_offsets[row_arr + 1] - fine.spectrum_offsets[row_arr]
        post_offsets = np.empty(n_internal + 1, dtype=INTERNAL_ID_DTYPE)
        post_offsets[0] = 0
        np.cumsum(lens, out=post_offsets[1:])
        total_post_peaks = int(post_offsets[-1])

        post_mass = np.empty(total_post_peaks, dtype=MASS_DTYPE)
        post_intensity = np.empty(total_post_peaks, dtype=INTENSITY_DTYPE)
        post_energy = np.empty(total_post_peaks, dtype=ENERGY_DTYPE)
        post_peak_id = np.empty(total_post_peaks, dtype=PEAK_ID_DTYPE)

        for i, row in enumerate(internal_to_row):
            s_st = int(fine.spectrum_offsets[row])
            s_ed = int(fine.spectrum_offsets[row + 1])
            d_st = int(post_offsets[i])
            d_ed = int(post_offsets[i + 1])
            post_mass[d_st:d_ed] = fine.mass[s_st:s_ed]
            post_intensity[d_st:d_ed] = fine.intensity[s_st:s_ed]
            post_energy[d_st:d_ed] = fine.energy[s_st:s_ed]
            post_peak_id[d_st:d_ed] = fine.peak_id[s_st:s_ed]

        post_norm = np.asarray(fine.norm[row_arr], dtype=ENERGY_DTYPE)
    else:
        post_offsets = np.zeros(1, dtype=INTERNAL_ID_DTYPE)
        post_mass = np.empty(0, dtype=MASS_DTYPE)
        post_intensity = np.empty(0, dtype=INTENSITY_DTYPE)
        post_energy = np.empty(0, dtype=ENERGY_DTYPE)
        post_peak_id = np.empty(0, dtype=PEAK_ID_DTYPE)
        post_norm = np.empty(0, dtype=ENERGY_DTYPE)

    postings = ForestPostings(
        mass=post_mass,
        intensity=post_intensity,
        energy=post_energy,
        peak_id=post_peak_id,
        spectrum_offsets=post_offsets,
        norm=post_norm,
    )

    envelopes = ForestEnvelopes(
        grid_da=grid_da,
        node_envelope_offsets=np.array(flat_env_offsets, dtype=INTERNAL_ID_DTYPE),
        cell_index=np.concatenate(env_cells)
        if env_cells
        else np.empty(0, dtype=INTERNAL_ID_DTYPE),
        max_peak_amplitude=np.concatenate(env_m)
        if env_m
        else np.empty(0, dtype=ENERGY_DTYPE),
    )

    trees = ForestTrees(
        precursor_min=np.array(tree_prec_min, dtype=MASS_DTYPE),
        precursor_max=np.array(tree_prec_max, dtype=MASS_DTYPE),
        root_node_id=np.array(tree_root_node_ids, dtype=INTERNAL_ID_DTYPE),
        leaf_offsets=np.array(tree_leaf_offsets, dtype=INTERNAL_ID_DTYPE),
        leaf_node_ids=np.array(tree_leaf_node_ids, dtype=INTERNAL_ID_DTYPE),
    )

    nodes = ForestNodes(
        is_leaf=np.array(node_is_leaf, dtype=np.bool_),
        member_count=np.array(node_member_count, dtype=INTERNAL_ID_DTYPE),
        id_start=np.array(node_id_start, dtype=INTERNAL_ID_DTYPE),
        id_end=np.array(node_id_end, dtype=INTERNAL_ID_DTYPE),
        tree_id=np.array(node_tree_id, dtype=INTERNAL_ID_DTYPE),
    )

    forest_index = ForestIndex(
        spec=spec,
        n_spectra=library.n_spectra,
        partitions=tuple(partitions),
        trees=trees,
        nodes=nodes,
        envelopes=envelopes,
        postings=postings,
        internal_to_row=np.array(internal_to_row, dtype=INTERNAL_ID_DTYPE),
        row_to_internal=row_to_internal,
        zero_energy_members=zero_energy_members,
        library_fingerprint=library.fingerprint,
        spectra=library.spectra,
    )

    check_forest_index(forest_index, library)
    return forest_index
