"""JET-Forest 检索执行器：全库开放式检索（Open / Threshold Search，Best-First 优先队列）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import heapq
import math
import time
import numpy as np

from jetf.bounds import (
    _HAVE_NUMBA,
    adaptive_numba_threads,
    batch_node_bounds,
    batch_root_bounds,
    build_query_context,
    peak_bound,
    window_cells,
)
from jetf.preprocessing import PreprocessedLibrary
from jetf.query import QueryConfig, SearchMode, ion_mode_passes, is_eligible
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
from jetf.types import INTERNAL_ID_DTYPE, SpectrumMeta, SpectrumPeaks, validate_query

from typing import Sequence

if _HAVE_NUMBA:
    from jetf.bounds import _batch_node_bounds_numba

_validate_query = validate_query


def _evaluate_leaf_spectra(
    node_id: int,
    query: SpectrumPeaks,
    forest: ForestIndex,
    spectra: Sequence[SpectrumMeta],
    config: QueryConfig,
    results: ResultSet,
    timers: EvalTimers,
    scored_rows: set[int],
    frag_tau: float,
    uind: bool,
) -> tuple[int, int]:
    """精评单个叶节点的所有合规谱图，更新结果集，返回 (n_scored_delta, uind_pruned_delta)。"""
    id_start = int(forest.nodes.id_start[node_id])
    id_end = int(forest.nodes.id_end[node_id])
    n_scored_delta = 0
    uind_pruned_delta = 0

    for iid in range(id_start, id_end):
        row = int(forest.internal_to_row[iid])
        if row in scored_rows:
            continue
        meta = spectra[row]

        if not is_eligible(config, meta):
            continue

        member_peaks = forest.postings.spectrum_at(iid)

        if uind:
            t_ustart = time.perf_counter()
            u_ind_val = single_spectrum_bound(query, member_peaks, frag_tau)
            timers.bound_eval_ms += elapsed_ms(t_ustart)

            if u_ind_val < results.theta():
                uind_pruned_delta += 1
                continue

        t_estart = time.perf_counter()
        score_res = score_greedy_cosine(query, member_peaks, frag_tau)
        timers.exact_eval_ms += elapsed_ms(t_estart)

        n_scored_delta += 1
        scored_rows.add(row)

        if score_res.n_matched >= config.min_matched_peaks:
            results.update(
                SearchHit(
                    score=score_res.score,
                    external_id=meta.external_id,
                    spectrum_index=row,
                    n_matched=score_res.n_matched,
                )
            )

    return n_scored_delta, uind_pruned_delta



def search_forest(
    query: SpectrumPeaks,
    forest: ForestIndex,
    library: PreprocessedLibrary | QueryConfig | None = None,
    config: QueryConfig | None = None,
    uind: bool = True,
) -> SearchOutcome:
    """在包络森林索引上执行全库开放式检索 (Open / Threshold Search)。

    采用层次化包络与全局 Best-First 优先队列，动态抬升门槛 theta 并进行分支与叶节点剪枝。

    参数:
    - query: 查询谱的峰表示 (SpectrumPeaks)
    - forest: 全库包络森林索引 (ForestIndex)
    - library: 预处理全库 (PreprocessedLibrary, 可选，若快照已含元数据可省略)
    - config: 开放检索配置 (QueryConfig)
    - uind: 是否开启单谱 Uind 上界过滤 (默认 True)
    """
    _validate_query(query)

    if isinstance(library, QueryConfig):
        config = library
        library = None
    if config is None:
        raise ValueError("必须提供 QueryConfig 配置")

    if library is None:
        if forest.spectra is None:
            raise ValueError("未提供 library 且 forest 快照未包含 spectra 元数据，无法执行检索")
        spectra = forest.spectra
        n_spectra = forest.n_spectra
    else:
        if forest.n_spectra != library.n_spectra:
            raise ValueError(
                f"森林总谱数不一致: 索引声明 {forest.n_spectra}, 库实际 {library.n_spectra}"
            )
        if forest.library_fingerprint and getattr(library, "fingerprint", None):
            if forest.library_fingerprint != library.fingerprint:
                raise ValueError(
                    f"森林索引指纹 ({forest.library_fingerprint}) 与传入全库指纹 ({library.fingerprint}) 不匹配"
                )
        spectra = library.spectra
        n_spectra = library.n_spectra

    timers = EvalTimers()
    results = ResultSet(config)
    scored_rows: set[int] = set()
    n_scored = 0

    # 0 峰空谱快速短路
    if query.mass.size == 0:
        if needs_zero_supplement(results, config):
            candidates = ZeroScoreCandidates(
                member=forest.zero_energy_members.member,
                ion_mode=forest.zero_energy_members.ion_mode,
                precursor_mz=forest.zero_energy_members.precursor_mz,
            )
            supplement_zero_score(results, spectra, config, candidates, scored_rows)
        hits = results.finish()
        pruned_by_layer = {
            "roots_pruned": 0,
            "leaves_pruned": 0,
            "uind_pruned": 0,
        }
        stats = SearchStats(
            nodes_visited=0,
            pruned_by_layer=pruned_by_layer,
            n_scored=0,
            bound_eval_time_ms=0.0,
            exact_eval_time_ms=0.0,
        )
        versions = search_versions(library, config, default_version=forest.spec.versioned_id)
        return SearchOutcome(
            mode=config.mode,
            hits=hits,
            complete=True,
            stats=stats,
            versions=versions,
        )

    roots_pruned = 0
    leaves_pruned = 0
    uind_pruned = 0
    nodes_visited = 0

    frag_tau = config.fragment_tolerance_da
    q_cell_lower, q_cell_upper = (
        window_cells(query.mass, frag_tau, forest.spec.summary_grid_da)
        if query.mass.size > 0
        else (np.empty(0, dtype=INTERNAL_ID_DTYPE), np.empty(0, dtype=INTERNAL_ID_DTYPE))
    )

    # 提前准备展平数组以供 JIT 零分配内核快速访问
    env_offsets = forest.envelopes.node_envelope_offsets
    cell_index = forest.envelopes.cell_index
    max_peak_amplitude = forest.envelopes.max_peak_amplitude
    q_intensity = np.ascontiguousarray(query.intensity, dtype=np.float64)
    q_cell_lower_arr = np.ascontiguousarray(q_cell_lower, dtype=INTERNAL_ID_DTYPE)
    q_cell_upper_arr = np.ascontiguousarray(q_cell_upper, dtype=INTERNAL_ID_DTYPE)

    # 确定离子模式过滤
    eligible_partitions = [
        p
        for p in forest.partitions
        if ion_mode_passes(config.ion_mode, config.ion_mode_policy, p.ion_mode)
    ]

    # =====================================================================
    # 全库开放式检索 (Open / Threshold Search，全局 Best-First 优先队列)
    # =====================================================================
    pqueue: list[tuple[float, int, int, bool, int]] = []
    entry_count = 0

    # 1. 批量评估所有合法分区的树根
    for partition in eligible_partitions:
        p_tstart = partition.tree_start
        p_tend = partition.tree_end
        if p_tstart >= p_tend:
            continue

        if config.precursor_window is not None:
            w_min = config.precursor_window.min_mz
            w_max = config.precursor_window.max_mz
            p_mins = forest.trees.precursor_min[p_tstart:p_tend]
            p_maxs = forest.trees.precursor_max[p_tstart:p_tend]

            start_offset = int(np.searchsorted(p_maxs, w_min, side="left"))
            end_offset = int(np.searchsorted(p_mins, w_max, side="right"))

            t_begin = p_tstart + start_offset
            t_finish = p_tstart + end_offset

            p_trees = [
                t
                for t in range(t_begin, t_finish)
                if not (
                    np.isnan(p_mins[t - p_tstart])
                    or np.isnan(p_maxs[t - p_tstart])
                    or p_mins[t - p_tstart] > w_max
                    or p_maxs[t - p_tstart] < w_min
                )
            ]
        else:
            p_trees = list(range(p_tstart, p_tend))

        if not p_trees:
            continue

        nodes_visited += len(p_trees)
        t_bstart = time.perf_counter()
        u_roots = batch_root_bounds(query, forest, p_trees, q_cell_lower, q_cell_upper)
        timers.bound_eval_ms += elapsed_ms(t_bstart)

        curr_theta = results.theta()
        for t_id, u_root in zip(p_trees, u_roots):
            u_float = float(u_root)
            # 原地剔除低于当前门槛或恒为 0 的无效小树 (200万库约 16.2% 无效小树在此被短路)
            if u_float < curr_theta or (config.min_matched_peaks > 0 and u_float <= 0.0):
                roots_pruned += 1
            else:
                root_id = int(forest.trees.root_node_id[t_id])
                entry_count += 1
                pqueue.append((-u_float, entry_count, root_id, False, t_id))

    # O(N) 批量建堆替代 31,303 次单个 heappush
    heapq.heapify(pqueue)

    # 2. 动态门槛 theta 快速预植入（Top-K 模式下对堆顶最优候选树贪心预精评）
    if config.mode == SearchMode.TOP_K and pqueue:
        probe_limit = min(3, len(pqueue))
        for _ in range(probe_limit):
            if not pqueue:
                break
            top_u = -pqueue[0][0]
            if top_u <= 0.0:
                break
            if results.theta() > -np.inf and top_u < results.theta():
                break

            # 弹出当前全局根上界最高的小树
            _, _, _, _, t_id = heapq.heappop(pqueue)
            leaf_ids = forest.trees.leaves_of_tree(t_id)
            nodes_visited += len(leaf_ids)

            t_bstart = time.perf_counter()
            if _HAVE_NUMBA:
                u_leaves = _batch_node_bounds_numba(
                    leaf_ids, env_offsets, cell_index, max_peak_amplitude, q_intensity, q_cell_lower_arr, q_cell_upper_arr
                )
            else:
                u_leaves = batch_node_bounds(query, forest, leaf_ids, q_cell_lower, q_cell_upper)
            timers.bound_eval_ms += elapsed_ms(t_bstart)

            # 按叶节点上界从高到低精评
            sorted_leaf_idx = np.argsort(-u_leaves)
            for l_idx in sorted_leaf_idx:
                lid = int(leaf_ids[l_idx])
                u_l = float(u_leaves[l_idx])
                curr_th = results.theta()
                if u_l < curr_th or (config.min_matched_peaks > 0 and u_l <= 0.0):
                    leaves_pruned += 1
                else:
                    sc, up = _evaluate_leaf_spectra(
                        lid, query, forest, spectra, config, results, timers, scored_rows, frag_tau, uind
                    )
                    n_scored += sc
                    uind_pruned += up

    # 3. 全局 Best-First 逐层展开
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
            # 展开树根 -> 直接调用 JIT 批量求叶界，零内存分配
            leaf_ids = forest.trees.leaves_of_tree(t_id)
            nodes_visited += len(leaf_ids)

            t_bstart = time.perf_counter()
            if _HAVE_NUMBA:
                u_leaves = _batch_node_bounds_numba(
                    leaf_ids, env_offsets, cell_index, max_peak_amplitude, q_intensity, q_cell_lower_arr, q_cell_upper_arr
                )
            else:
                u_leaves = batch_node_bounds(query, forest, leaf_ids, q_cell_lower, q_cell_upper)
            timers.bound_eval_ms += elapsed_ms(t_bstart)

            curr_theta = results.theta()
            for lid, u_leaf in zip(leaf_ids, u_leaves):
                u_l_val = float(u_leaf)
                if u_l_val < curr_theta or (config.min_matched_peaks > 0 and u_l_val <= 0.0):
                    leaves_pruned += 1
                else:
                    entry_count += 1
                    heapq.heappush(pqueue, (-u_l_val, entry_count, int(lid), True, t_id))
        else:
            # 精评叶节点
            sc, up = _evaluate_leaf_spectra(
                node_id, query, forest, spectra, config, results, timers, scored_rows, frag_tau, uind
            )
            n_scored += sc
            uind_pruned += up

    # 4. 零分记录按元数据资格补足
    if needs_zero_supplement(results, config):
        candidates = ZeroScoreCandidates(
            member=forest.zero_energy_members.member,
            ion_mode=forest.zero_energy_members.ion_mode,
            precursor_mz=forest.zero_energy_members.precursor_mz,
        )
        supplement_zero_score(results, spectra, config, candidates, scored_rows)

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
    )

    versions = search_versions(library, config, default_version=forest.spec.versioned_id)

    return SearchOutcome(
        mode=config.mode,
        hits=hits,
        complete=True,
        stats=stats,
        versions=versions,
    )


def search_forest_batch(
    queries: Sequence[SpectrumPeaks],
    forest: ForestIndex,
    library: PreprocessedLibrary | None = None,
    config: QueryConfig | Sequence[QueryConfig] | None = None,
    concurrency: int = 1,
) -> list[SearchOutcome]:
    """多查询批量检索，内置自适应多线程并发控制与线程安全隔离。

    参数:
        queries: 查询谱列表 (SpectrumPeaks)。
        forest: 森林索引 (ForestIndex)。
        library: 预处理参考库 (PreprocessedLibrary)，若快照已自包含元数据可为 None。
        config: 单个统一 QueryConfig，或与 queries 长度相同的 QueryConfig 序列。
        concurrency: 并发工作线程数 (默认 1 为串行)。

    返回:
        按输入 queries 顺序严格排列的 SearchOutcome 列表。
    """
    n_queries = len(queries)
    if n_queries == 0:
        return []

    if config is None:
        raise ValueError("必须提供 config 参数 (QueryConfig 或其序列)")

    if isinstance(config, QueryConfig):
        cfg_list = [config] * n_queries
    elif isinstance(config, Sequence):
        if len(config) != n_queries:
            raise ValueError(
                f"config 序列长度 ({len(config)}) 与 queries 长度 ({n_queries}) 不匹配"
            )
        cfg_list = list(config)
    else:
        raise TypeError(f"未知的 config 类型: {type(config).__name__}")

    effective_concurrency = max(1, min(concurrency, n_queries))

    with adaptive_numba_threads(effective_concurrency) as target_inner:
        if effective_concurrency <= 1:
            return [
                search_forest(queries[i], forest, library, cfg_list[i])
                for i in range(n_queries)
            ]
        else:
            def _init_worker(inner_threads: int | None) -> None:
                if inner_threads is not None:
                    try:
                        import numba
                        numba.set_num_threads(inner_threads)
                    except Exception:
                        pass

            def _worker(idx: int) -> SearchOutcome:
                return search_forest(queries[idx], forest, library, cfg_list[idx])

            with ThreadPoolExecutor(
                max_workers=effective_concurrency,
                initializer=_init_worker,
                initargs=(target_inner,),
            ) as executor:
                return list(executor.map(_worker, range(n_queries)))

