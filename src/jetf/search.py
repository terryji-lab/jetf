"""JET-Forest 检索执行器：双模态原生检索（身份检索与开放检索）。"""

from __future__ import annotations

import heapq
import time
import numpy as np

from jetf.bounds import build_query_context, peak_bound
from jetf.preprocessing import PreprocessedLibrary
from jetf.query import QueryConfig, ion_mode_passes
from jetf.results import (
    EvalTimers,
    ResultSet,
    SearchHit,
    SearchOutcome,
    SearchStats,
    ZeroScoreCandidates,
    elapsed_ms,
    needs_zero_supplement,
    search_versions,
    supplement_zero_score,
)
from jetf.scoring import score_greedy_cosine, single_spectrum_bound
from jetf.structure import ForestIndex
from jetf.types import SpectrumPeaks


def search_forest(
    query: SpectrumPeaks,
    forest: ForestIndex,
    library: PreprocessedLibrary,
    config: QueryConfig,
    uind: bool = True,
) -> SearchOutcome:
    """在包络森林索引上执行双模态检索。

    参数:
    - query: 查询谱的峰表示 (SpectrumPeaks)
    - forest: 全库包络森林索引 (ForestIndex)
    - library: 预处理全库 (PreprocessedLibrary)
    - config: 查询配置 (QueryConfig)
    - uind: 是否开启单谱 Uind 上界过滤 (默认 True)
    """
    timers = EvalTimers()
    results = ResultSet(config)
    scored_mask = np.zeros(library.n_spectra, dtype=np.bool_)
    n_scored = 0

    roots_pruned = 0
    leaves_pruned = 0
    uind_pruned = 0
    nodes_visited = 0

    # 确定离子模式过滤
    eligible_partitions = [
        p
        for p in forest.partitions
        if ion_mode_passes(config.ion_mode, config.ion_mode_policy, p.ion_mode)
    ]

    is_identity = config.precursor_window is not None
    frag_tau = config.fragment_tolerance_da

    if is_identity:
        # =====================================================================
        # 模式 A: 身份检索 (Identity Search，带前体窗口)
        # =====================================================================
        w_min = float(config.precursor_window.mz - config.precursor_window.tolerance_da)
        w_max = float(config.precursor_window.mz + config.precursor_window.tolerance_da)

        for partition in eligible_partitions:
            p_tstart = partition.tree_start
            p_tend = partition.tree_end
            if p_tstart >= p_tend:
                continue

            prec_mins = forest.trees.precursor_min[p_tstart:p_tend]
            prec_maxs = forest.trees.precursor_max[p_tstart:p_tend]

            # 二分查找前体区间重叠树范围
            start_offset = int(np.searchsorted(prec_maxs, w_min, side="left"))
            end_offset = int(np.searchsorted(prec_mins, w_max, side="right"))

            t_begin = p_tstart + start_offset
            t_finish = min(p_tend, p_tstart + end_offset)

            for t_id in range(t_begin, t_finish):
                t_pmin = forest.trees.precursor_min[t_id]
                t_pmax = forest.trees.precursor_max[t_id]

                if np.isnan(t_pmin) or t_pmin > w_max or t_pmax < w_min:
                    continue

                # 1. 根包络检查
                root_id = int(forest.trees.root_node_id[t_id])
                nodes_visited += 1

                t_bstart = time.perf_counter()
                root_env = forest.envelope_of(root_id)
                ctx = build_query_context(query, root_env, frag_tau)
                u_root = peak_bound(ctx, root_env)
                timers.bound_eval_ms += elapsed_ms(t_bstart)

                if u_root < results.theta():
                    roots_pruned += 1
                    continue  # 整树秒杀！

                # 2. 展开该树的叶节点
                leaf_ids = forest.trees.leaves_of_tree(t_id)
                for lid in leaf_ids:
                    nodes_visited += 1

                    t_bstart = time.perf_counter()
                    leaf_env = forest.envelope_of(lid)
                    ctx_leaf = build_query_context(query, leaf_env, frag_tau)
                    u_leaf = peak_bound(ctx_leaf, leaf_env)
                    timers.bound_eval_ms += elapsed_ms(t_bstart)

                    if u_leaf < results.theta():
                        leaves_pruned += 1
                        continue  # 整叶跳过！

                    # 3. 叶内微块成员精评
                    id_start = int(forest.nodes.id_start[lid])
                    id_end = int(forest.nodes.id_end[lid])

                    for iid in range(id_start, id_end):
                        row = int(forest.internal_to_row[iid])
                        meta = library.spectra[row]

                        if config.exclude_spectrum_id is not None and meta.external_id == config.exclude_spectrum_id:
                            continue

                        p_mz = meta.precursor_mz
                        if p_mz is None or not config.precursor_window.contains(p_mz):
                            continue

                        member_peaks = forest.postings.spectrum_at(iid)

                        # Uind 单谱精确上界过滤
                        if uind:
                            t_ustart = time.perf_counter()
                            u_ind_val = single_spectrum_bound(query, member_peaks, frag_tau)
                            timers.bound_eval_ms += elapsed_ms(t_ustart)

                            if u_ind_val < results.theta():
                                uind_pruned += 1
                                continue

                        # 精确确定性 Greedy Cosine 评分
                        t_estart = time.perf_counter()
                        score_res = score_greedy_cosine(query, member_peaks, frag_tau)
                        timers.exact_eval_ms += elapsed_ms(t_estart)

                        n_scored += 1
                        scored_mask[row] = True

                        if score_res.n_matched >= config.min_matched_peaks:
                            results.update(
                                SearchHit(
                                    score=score_res.score,
                                    external_id=meta.external_id,
                                    spectrum_index=row,
                                    n_matched=score_res.n_matched,
                                )
                            )

    else:
        # =====================================================================
        # 模式 B: 开放检索 (Open / Threshold Search，全局 Best-First 优先队列)
        # =====================================================================
        pqueue: list[tuple[float, int, int, bool, int]] = []
        entry_count = 0

        # 1. 评估所有合法分区的树根并入堆
        for partition in eligible_partitions:
            for t_id in range(partition.tree_start, partition.tree_end):
                root_id = int(forest.trees.root_node_id[t_id])
                nodes_visited += 1

                t_bstart = time.perf_counter()
                root_env = forest.envelope_of(root_id)
                ctx = build_query_context(query, root_env, frag_tau)
                u_root = peak_bound(ctx, root_env)
                timers.bound_eval_ms += elapsed_ms(t_bstart)

                if u_root < results.theta():
                    roots_pruned += 1
                else:
                    entry_count += 1
                    heapq.heappush(pqueue, (-u_root, entry_count, root_id, False, t_id))

        # 2. 全局 Best-First 逐层展开
        while pqueue:
            neg_u, _, node_id, is_leaf, t_id = heapq.heappop(pqueue)
            u_val = -neg_u

            # 堆顶是当前未访问候选中上界最大者；如果它 < theta，堆内剩余全部节点都可以剪除
            if u_val < results.theta():
                if is_leaf:
                    leaves_pruned += 1
                else:
                    roots_pruned += 1
                while pqueue:
                    _, _, _, rem_is_leaf, _ = heapq.heappop(pqueue)
                    if rem_is_leaf:
                        leaves_pruned += 1
                    else:
                        roots_pruned += 1
                break

            if not is_leaf:
                # 展开树根 -> 将其所有叶子压入堆
                leaf_ids = forest.trees.leaves_of_tree(t_id)
                for lid in leaf_ids:
                    nodes_visited += 1

                    t_bstart = time.perf_counter()
                    leaf_env = forest.envelope_of(lid)
                    ctx_leaf = build_query_context(query, leaf_env, frag_tau)
                    u_leaf = peak_bound(ctx_leaf, leaf_env)
                    timers.bound_eval_ms += elapsed_ms(t_bstart)

                    if u_leaf < results.theta():
                        leaves_pruned += 1
                    else:
                        entry_count += 1
                        heapq.heappush(pqueue, (-u_leaf, entry_count, lid, True, t_id))
            else:
                # 精评叶节点
                id_start = int(forest.nodes.id_start[node_id])
                id_end = int(forest.nodes.id_end[node_id])

                for iid in range(id_start, id_end):
                    row = int(forest.internal_to_row[iid])
                    meta = library.spectra[row]

                    if config.exclude_spectrum_id is not None and meta.external_id == config.exclude_spectrum_id:
                        continue

                    member_peaks = forest.postings.spectrum_at(iid)

                    if uind:
                        t_ustart = time.perf_counter()
                        u_ind_val = single_spectrum_bound(query, member_peaks, frag_tau)
                        timers.bound_eval_ms += elapsed_ms(t_ustart)

                        if u_ind_val < results.theta():
                            uind_pruned += 1
                            continue

                    t_estart = time.perf_counter()
                    score_res = score_greedy_cosine(query, member_peaks, frag_tau)
                    timers.exact_eval_ms += elapsed_ms(t_estart)

                    n_scored += 1
                    scored_mask[row] = True

                    if score_res.n_matched >= config.min_matched_peaks:
                        results.update(
                            SearchHit(
                                score=score_res.score,
                                external_id=meta.external_id,
                                spectrum_index=row,
                                n_matched=score_res.n_matched,
                            )
                        )

    # 4. 零分记录按元数据资格补足
    if needs_zero_supplement(results, config):
        candidates = ZeroScoreCandidates(
            member=forest.zero_energy_members.member,
            ion_mode=forest.zero_energy_members.ion_mode,
            precursor_mz=forest.zero_energy_members.precursor_mz,
        )
        supplement_zero_score(results, library, config, candidates, scored_mask)

    hits = results.finish()

    pruned_by_layer = {
        "roots_pruned": roots_pruned,
        "leaves_pruned": leaves_pruned,
        "uind_pruned": uind_pruned,
    }

    stats = SearchStats(
        nodes_visited=nodes_visited,
        pruned_by_layer=pruned_by_layer,
        n_scored=n_scored,
        bound_eval_time_ms=timers.bound_eval_ms,
        exact_eval_time_ms=timers.exact_eval_ms,
        seed_count=0,
        skipped_block_intervals=(),
    )

    versions = search_versions(library, config)

    return SearchOutcome(
        mode=config.mode,
        hits=hits,
        complete=True,
        stats=stats,
        versions=versions,
    )
