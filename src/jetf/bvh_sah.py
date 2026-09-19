"""图形学表面积启发式（BVH-SAH）在 0.02 Da 质谱空间中的二分紧致切分算法。"""

from __future__ import annotations

from collections.abc import Sequence


def sah_cost(
    cells_list: Sequence[set[int]],
    amps_list: Sequence[dict[int, float]],
) -> float:
    """计算一组谱打包为单个节点时的 SAH 代价。

    Cost(B) = |Z(B)| * sum_{h in Z(B)} m_h(B)
    """
    if not cells_list:
        return 0.0
    union_cells = set().union(*cells_list)
    if not union_cells:
        return 0.0

    max_amps: dict[int, float] = {}
    for amps in amps_list:
        for c, a in amps.items():
            if c not in max_amps or a > max_amps[c]:
                max_amps[c] = a

    return float(len(union_cells)) * float(sum(max_amps.values()))


def split_sah_bvh(
    spectrum_indices: Sequence[int],
    spec_cells: Sequence[set[int]],
    spec_amps: Sequence[dict[int, float]],
    target_leaf_size: int = 16,
    max_candidate_axes: int = 10,
) -> list[list[int]]:
    """递归二分切分算法，输出紧致叶节点成员行列表。

    参数:
    - spectrum_indices: 当前桶内的成员行序号列表 (通常 N <= 64)
    - spec_cells: 全库每条谱的 0.02 Da non-zero cell 集合
    - spec_amps: 全库每条谱的 {cell: max_amplitude} 映射
    - target_leaf_size: 叶节点目标大小 (默认 16)
    - max_candidate_axes: 考察的最大候选切分轴数 (默认 10)
    """
    indices = list(spectrum_indices)
    n = len(indices)
    if n <= target_leaf_size:
        return [indices]

    # 1. 统计当前子集内各 0.02 Da cell 的出现频次
    cell_freq: dict[int, int] = {}
    for idx in indices:
        for c in spec_cells[idx]:
            cell_freq[c] = cell_freq.get(c, 0) + 1

    # 2. 挑选信息量最大的投影轴 (频次方差 f * (n - f) 最大，并列按 cell 编号升序破并列)
    candidate_axes = sorted(
        cell_freq.keys(),
        key=lambda c: (cell_freq[c] * (n - cell_freq[c]), -c),
        reverse=True,
    )[:max_candidate_axes]

    best_split: tuple[list[int], list[int]] | None = None
    min_total_cost = float("inf")
    half = n // 2
    min_leaf = max(4, target_leaf_size // 4)

    # 3. 在候选轴上寻找最小 SAH Cost 的平衡切分
    if candidate_axes:
        for axis_cell in candidate_axes:
            # 按在该 cell 的峰强度升序排序，并列按原谱序号升序破并列（保证确定性）
            sorted_indices = sorted(
                indices,
                key=lambda idx: (spec_amps[idx].get(axis_cell, 0.0), idx),
            )

            # 考察平衡切分位置
            candidate_positions = [half]
            for delta in [2, 4, 6]:
                if half - delta >= min_leaf:
                    candidate_positions.append(half - delta)
                if half + delta <= n - min_leaf:
                    candidate_positions.append(half + delta)

            for split_pos in candidate_positions:
                left_group = sorted_indices[:split_pos]
                right_group = sorted_indices[split_pos:]

                c_left = sah_cost(
                    [spec_cells[i] for i in left_group],
                    [spec_amps[i] for i in left_group],
                )
                c_right = sah_cost(
                    [spec_cells[i] for i in right_group],
                    [spec_amps[i] for i in right_group],
                )
                total_cost = c_left + c_right

                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_split = (left_group, right_group)

    # 4. 兜底处理：若无有效候选轴或无法产生更优划分，按原顺序直接均分
    if best_split is None:
        best_split = (indices[:half], indices[half:])

    # 5. 递归切分左右子集
    left_leaves = split_sah_bvh(
        best_split[0],
        spec_cells,
        spec_amps,
        target_leaf_size=target_leaf_size,
        max_candidate_axes=max_candidate_axes,
    )
    right_leaves = split_sah_bvh(
        best_split[1],
        spec_cells,
        spec_amps,
        target_leaf_size=target_leaf_size,
        max_candidate_axes=max_candidate_axes,
    )
    return left_leaves + right_leaves
