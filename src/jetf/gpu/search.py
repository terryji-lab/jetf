"""GPU-accelerated batch threshold search pipeline for JET-Forest.

Implements end-to-end multi-layer filtering (K1 root bounds, K2 leaf bounds,
and K3a single-spectrum U_ind upper bounds) with conservative FP32 inflation
guaranteeing zero false dismissals and bitwise numerical consistency with CPU.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence
import numpy as np

from jetf.bounds import (
    _HAVE_NUMBA,
    batch_node_bounds,
    window_cells,
)
if _HAVE_NUMBA:
    from jetf.bounds import _batch_node_bounds_numba_serial
from jetf.gpu import (
    BatchQueryDevice,
    GpuForestIndex,
    batch_greedy_cosine_pairs_gpu,
    batch_node_bounds_gpu,
    batch_root_bounds_gpu,
    batch_uind_pairs_gpu,
    require_cuda,
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
from jetf.types import SpectrumMeta, SpectrumPeaks, validate_query

# Default score margin for GPU results to account for FP32 accumulation variance (~5e-5)
DEFAULT_GPU_SCORE_MARGIN: float = 5e-5


def _find_eligible_trees(
    forest: ForestIndex,
    config: QueryConfig,
) -> list[int]:
    """Filter candidate tree IDs according to ion mode policy and precursor window."""
    eligible_partitions = [
        p
        for p in forest.partitions
        if ion_mode_passes(config.ion_mode, config.ion_mode_policy, p.ion_mode)
    ]
    p_trees: list[int] = []
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

            for t in range(t_begin, t_finish):
                if not (
                    np.isnan(p_mins[t - p_tstart])
                    or np.isnan(p_maxs[t - p_tstart])
                    or p_mins[t - p_tstart] > w_max
                    or p_maxs[t - p_tstart] < w_min
                ):
                    p_trees.append(t)
        else:
            p_trees.extend(range(p_tstart, p_tend))
    return p_trees


def presort_by_precursor_mz(
    queries: Sequence[SpectrumPeaks],
    configs: Sequence[QueryConfig],
) -> tuple[list[SpectrumPeaks], list[QueryConfig], np.ndarray | None]:
    """Presort queries and configs by precursor window center m/z.

    Queries without precursor windows are assigned inf center m/z and placed last.
    If no query has a precursor window, returns (list(queries), list(configs), None).

    Returns:
        (sorted_queries, sorted_configs, inv_order)
        where inv_order maps original query index to its sorted position:
        original_outcomes = [sorted_outcomes[inv_order[i]] for i in range(len(queries))]
    """
    has_precursor_window = any(c.precursor_window is not None for c in configs)
    if not has_precursor_window:
        return list(queries), list(configs), None

    center_mzs = np.array(
        [
            (c.precursor_window.min_mz + c.precursor_window.max_mz) / 2.0
            if c.precursor_window is not None
            else np.inf
            for c in configs
        ],
        dtype=np.float64,
    )
    sort_order = np.argsort(center_mzs, kind="stable")
    inv_order = np.empty_like(sort_order)
    inv_order[sort_order] = np.arange(len(sort_order), dtype=sort_order.dtype)

    sorted_queries = [queries[i] for i in sort_order]
    sorted_configs = [configs[i] for i in sort_order]
    return sorted_queries, sorted_configs, inv_order


class _PrefetchedBatchK1:
    """Encapsulates prefetched device data and K1 bound results for a query batch."""

    def __init__(
        self,
        b_idx: int,
        batch_queries: list[SpectrumPeaks],
        batch_cfgs: list[QueryConfig],
        has_peaks: list[bool],
        frag_tau: float,
        batch_query: BatchQueryDevice | None,
        eligible_trees_for_q: list[list[int]],
        union_trees: np.ndarray,
        d_root_bounds: Any,
        h_root_bounds: np.ndarray | None,
        k1_launch_ms: float,
        stream: Any,
    ):
        self.b_idx = b_idx
        self.batch_queries = batch_queries
        self.batch_cfgs = batch_cfgs
        self.has_peaks = has_peaks
        self.frag_tau = frag_tau
        self.batch_query = batch_query
        self.eligible_trees_for_q = eligible_trees_for_q
        self.union_trees = union_trees
        self.d_root_bounds = d_root_bounds
        self.h_root_bounds = h_root_bounds
        self.k1_launch_ms = k1_launch_ms
        self.stream = stream


def _launch_batch_k1(
    b_idx: int,
    batch_queries: Sequence[SpectrumPeaks],
    batch_cfgs: Sequence[QueryConfig],
    gpu_f: GpuForestIndex,
    forest: ForestIndex,
    tree_cache: dict[tuple[Any, ...], list[int]],
    stream: Any,
) -> _PrefetchedBatchK1:
    """Asynchronously pack query batch, evaluate K1 root bounds on GPU, and initiate D2H transfer."""
    B = len(batch_queries)
    has_peaks = [q.mass.size > 0 for q in batch_queries]

    if not any(has_peaks):
        return _PrefetchedBatchK1(
            b_idx=b_idx,
            batch_queries=list(batch_queries),
            batch_cfgs=list(batch_cfgs),
            has_peaks=has_peaks,
            frag_tau=batch_cfgs[0].fragment_tolerance_da,
            batch_query=None,
            eligible_trees_for_q=[[] for _ in range(B)],
            union_trees=np.empty(0, dtype=np.int64),
            d_root_bounds=None,
            h_root_bounds=None,
            k1_launch_ms=0.0,
            stream=stream,
        )

    frag_tau = batch_cfgs[0].fragment_tolerance_da
    if not all(c.fragment_tolerance_da == frag_tau for c in batch_cfgs):
        raise ValueError(
            "All queries within the same batch must share identical fragment_tolerance_da"
        )

    # 1. Asynchronously pack queries and H2D copy to device on stream
    batch_query = BatchQueryDevice.from_queries(
        batch_queries,
        frag_tau=frag_tau,
        grid_da=gpu_f.grid_da,
        stream=stream,
    )

    # 2. Candidate trees resolution on Host
    eligible_trees_for_q: list[list[int]] = []
    for i in range(B):
        if not has_peaks[i]:
            eligible_trees_for_q.append([])
            continue
        cfg = batch_cfgs[i]
        pw_key = (
            None
            if cfg.precursor_window is None
            else (cfg.precursor_window.min_mz, cfg.precursor_window.max_mz)
        )
        cache_key = (cfg.ion_mode, cfg.ion_mode_policy, pw_key)
        if cache_key not in tree_cache:
            tree_cache[cache_key] = _find_eligible_trees(forest, cfg)
        eligible_trees_for_q.append(tree_cache[cache_key])

    all_eligible_trees = [t for trees in eligible_trees_for_q for t in trees]

    # 3. K1 launch & asynchronous D2H copy
    if len(all_eligible_trees) > 0:
        union_trees = np.unique(np.array(all_eligible_trees, dtype=np.int64))
        t_k1_launch_start = time.perf_counter()
        d_root_bounds = batch_root_bounds_gpu(
            batch_query, gpu_f, tree_ids=union_trees, stream=stream
        )
        h_root_bounds = d_root_bounds.copy_to_host(stream=stream)
        k1_launch_ms = elapsed_ms(t_k1_launch_start)
    else:
        union_trees = np.empty(0, dtype=np.int64)
        d_root_bounds = None
        h_root_bounds = None
        k1_launch_ms = 0.0

    return _PrefetchedBatchK1(
        b_idx=b_idx,
        batch_queries=list(batch_queries),
        batch_cfgs=list(batch_cfgs),
        has_peaks=has_peaks,
        frag_tau=frag_tau,
        batch_query=batch_query,
        eligible_trees_for_q=eligible_trees_for_q,
        union_trees=union_trees,
        d_root_bounds=d_root_bounds,
        h_root_bounds=h_root_bounds,
        k1_launch_ms=k1_launch_ms,
        stream=stream,
    )


def _execute_gpu_batch_pipeline(
    queries: list[SpectrumPeaks],
    cfgs: list[QueryConfig],
    gpu_f: GpuForestIndex,
    forest: ForestIndex,
    spectra: Sequence[SpectrumMeta],
    library: PreprocessedLibrary | None,
    batch_size: int,
    probe_trees: int,
    uind: bool,
    stream: Any,
    score_margin: float,
    mode: SearchMode,
) -> list[SearchOutcome]:
    """Execute unified multi-layer GPU search pipeline with double-buffering overlap.

    Supports both SearchMode.THRESHOLD and SearchMode.TOP_K.
    """
    n_queries = len(queries)
    if n_queries == 0:
        return []

    # Precursor mz adaptive bucketing presort
    has_precursor_window = any(c.precursor_window is not None for c in cfgs)
    if has_precursor_window and n_queries > batch_size:
        queries_proc, cfgs_proc, inv_order = presort_by_precursor_mz(queries, cfgs)
    else:
        queries_proc = queries
        cfgs_proc = cfgs
        inv_order = None

    batch_slices = [
        (b_start, min(b_start + batch_size, n_queries))
        for b_start in range(0, n_queries, batch_size)
    ]
    n_batches = len(batch_slices)
    all_outcomes: list[SearchOutcome] = []

    if stream is None:
        from numba import cuda

        streams = [cuda.stream(), cuda.stream()]
    else:
        streams = [stream, stream]

    tree_cache: dict[tuple[Any, ...], list[int]] = {}

    def _prefetch_batch(b_i: int) -> _PrefetchedBatchK1 | None:
        if b_i < n_batches:
            b_s, b_e = batch_slices[b_i]
            b_stream = streams[b_i % 2]
            return _launch_batch_k1(
                b_idx=b_i,
                batch_queries=queries_proc[b_s:b_e],
                batch_cfgs=cfgs_proc[b_s:b_e],
                gpu_f=gpu_f,
                forest=forest,
                tree_cache=tree_cache,
                stream=b_stream,
            )
        return None

    # Pre-launch Batch 0 on streams[0]
    next_prefetched: _PrefetchedBatchK1 | None = _prefetch_batch(0)

    # Double-buffering batch processing loop
    for b_idx in range(n_batches):
        curr_batch = next_prefetched
        assert curr_batch is not None
        next_prefetched = None  # consumed

        curr_stream = curr_batch.stream
        batch_queries = curr_batch.batch_queries
        batch_cfgs = curr_batch.batch_cfgs
        has_peaks = curr_batch.has_peaks
        frag_tau = curr_batch.frag_tau
        batch_query = curr_batch.batch_query
        eligible_trees_for_q = curr_batch.eligible_trees_for_q
        union_trees = curr_batch.union_trees
        d_root_bounds = curr_batch.d_root_bounds
        h_root_bounds = curr_batch.h_root_bounds
        B = len(batch_queries)

        results = [ResultSet(cfg, score_margin=score_margin) for cfg in batch_cfgs]
        scored_rows: list[set[int]] = [set() for _ in range(B)]
        timers = [EvalTimers() for _ in range(B)]
        nodes_visited = [0] * B
        roots_pruned = [0] * B
        leaves_pruned = [0] * B
        uind_pruned = [0] * B
        n_scored = [0] * B
        probe_scored = [0] * B

        for i in range(B):
            nodes_visited[i] += len(eligible_trees_for_q[i])

        # Fast short-circuit if ALL queries in batch have 0 peaks
        if not any(has_peaks):
            next_prefetched = _prefetch_batch(b_idx + 1)
            for i in range(B):
                cfg = batch_cfgs[i]
                res = results[i]
                if needs_zero_supplement(res, cfg):
                    candidates = ZeroScoreCandidates(
                        member=forest.zero_energy_members.member,
                        ion_mode=forest.zero_energy_members.ion_mode,
                        precursor_mz=forest.zero_energy_members.precursor_mz,
                    )
                    supplement_zero_score(res, spectra, cfg, candidates, scored_rows[i])
                hits = res.finish()
                pruned_by_layer: dict[str, int] = {
                    "roots_pruned": 0,
                    "leaves_pruned": 0,
                    "uind_pruned": 0,
                }
                if mode == SearchMode.TOP_K:
                    pruned_by_layer["probed_roots"] = 0
                stats = SearchStats(
                    nodes_visited=0,
                    pruned_by_layer=pruned_by_layer,
                    n_scored=0,
                    bound_eval_time_ms=0.0,
                    exact_eval_time_ms=0.0,
                    probe_scored=0,
                )
                versions = search_versions(library, cfg, default_version=forest.spec.versioned_id)
                all_outcomes.append(
                    SearchOutcome(
                        mode=cfg.mode,
                        hits=hits,
                        complete=True,
                        stats=stats,
                        versions=versions,
                    )
                )
            continue

        probed_trees_for_q: list[set[int]] | None = (
            [set() for _ in range(B)] if mode == SearchMode.TOP_K else None
        )

        if len(union_trees) > 0 and h_root_bounds is not None and batch_query is not None:
            # Step 1: Synchronize K1 Root bounds evaluation on GPU
            t_sync_start = time.perf_counter()
            if curr_stream is not None:
                curr_stream.synchronize()
            k1_elapsed = curr_batch.k1_launch_ms + elapsed_ms(t_sync_start)
            for i in range(B):
                if has_peaks[i]:
                    timers[i].bound_eval_ms += k1_elapsed / B

            # Step 2: Speculative Probe Phase (CPU) - TOP_K mode only
            effective_probe_trees = max(0, int(probe_trees)) if mode == SearchMode.TOP_K else 0
            if effective_probe_trees > 0 and probed_trees_for_q is not None:
                for i in range(B):
                    q_trees = eligible_trees_for_q[i]
                    if not q_trees or not has_peaks[i]:
                        continue
                    tree_positions = np.searchsorted(union_trees, q_trees)
                    u_roots_q = h_root_bounds[i, tree_positions]
                    cfg = batch_cfgs[i]
                    res = results[i]
                    min_matched = cfg.min_matched_peaks
                    q_scored = scored_rows[i]
                    query = batch_queries[i]

                    q_cell_lower, q_cell_upper = window_cells(
                        query.mass, frag_tau, forest.spec.summary_grid_da
                    )

                    sort_order = np.argsort(-u_roots_q)
                    probed_count = 0
                    for s_idx in sort_order:
                        if probed_count >= effective_probe_trees:
                            break
                        t_id = int(q_trees[s_idx])
                        u_root_val = float(u_roots_q[s_idx])

                        curr_th = res.theta()
                        if (min_matched > 0 and u_root_val <= 0.0) or (
                            curr_th > -np.inf and u_root_val < curr_th
                        ):
                            break

                        probed_trees_for_q[i].add(t_id)
                        probed_count += 1
                        leaf_ids = forest.trees.leaves_of_tree(t_id)
                        nodes_visited[i] += len(leaf_ids)

                        if len(leaf_ids) == 0:
                            continue

                        # Evaluate leaf bounds on CPU with K2 bounds
                        t_k2_cpu_start = time.perf_counter()
                        if _HAVE_NUMBA:
                            u_leaves = _batch_node_bounds_numba_serial(
                                leaf_ids,
                                forest.envelopes.node_envelope_offsets,
                                forest.envelopes.cell_index,
                                forest.envelopes.max_peak_amplitude,
                                query.intensity,
                                q_cell_lower,
                                q_cell_upper,
                            )
                        else:
                            u_leaves = batch_node_bounds(
                                query, forest, leaf_ids, q_cell_lower, q_cell_upper
                            )
                        timers[i].bound_eval_ms += elapsed_ms(t_k2_cpu_start)

                        # Sort leaves by bound descending
                        sorted_leaf_idx = np.argsort(-u_leaves)
                        for l_idx in sorted_leaf_idx:
                            lid = int(leaf_ids[l_idx])
                            u_leaf = float(u_leaves[l_idx])
                            curr_th = res.theta()
                            if (curr_th > -np.inf and u_leaf < curr_th) or (
                                min_matched > 0 and u_leaf <= 0.0
                            ):
                                leaves_pruned[i] += 1
                                continue

                            id_start = int(forest.nodes.id_start[lid])
                            id_end = int(forest.nodes.id_end[lid])
                            for iid in range(id_start, id_end):
                                row = int(forest.internal_to_row[iid])
                                if row in q_scored:
                                    continue
                                meta = spectra[row]
                                if not is_eligible(cfg, meta):
                                    continue

                                member_peaks = forest.postings.spectrum_at(iid)

                                if uind:
                                    t_uind_start = time.perf_counter()
                                    u_ind_val = single_spectrum_bound(query, member_peaks, frag_tau)
                                    timers[i].bound_eval_ms += elapsed_ms(t_uind_start)

                                    curr_th = res.theta()
                                    if (curr_th > -np.inf and u_ind_val < curr_th) or (
                                        min_matched > 0 and u_ind_val <= 0.0
                                    ):
                                        uind_pruned[i] += 1
                                        continue

                                t_exact_start = time.perf_counter()
                                score_res = score_greedy_cosine(query, member_peaks, frag_tau)
                                timers[i].exact_eval_ms += elapsed_ms(t_exact_start)
                                n_scored[i] += 1
                                probe_scored[i] += 1
                                q_scored.add(row)

                                if score_res.n_matched >= min_matched:
                                    res.update(
                                        SearchHit(
                                            score=score_res.score,
                                            external_id=meta.external_id,
                                            spectrum_index=row,
                                            n_matched=score_res.n_matched,
                                        )
                                    )

            # Step 3: Bulk Pruning Phase (K1 prune)
            surviving_trees_for_q: list[list[int]] = []
            for i in range(B):
                q_trees = eligible_trees_for_q[i]
                if not q_trees or not has_peaks[i]:
                    surviving_trees_for_q.append([])
                    continue
                t_root_filter_start = time.perf_counter()
                tree_positions = np.searchsorted(union_trees, q_trees)
                u_roots = h_root_bounds[i, tree_positions]
                eff_theta = results[i].theta()
                min_matched = batch_cfgs[i].min_matched_peaks
                probed_set = probed_trees_for_q[i] if probed_trees_for_q is not None else None

                surviving = []
                for t_idx, t_id in enumerate(q_trees):
                    if probed_set is not None and t_id in probed_set:
                        continue
                    u_r = float(u_roots[t_idx])
                    if (eff_theta > -np.inf and u_r < eff_theta) or (
                        min_matched > 0 and u_r <= 0.0
                    ):
                        roots_pruned[i] += 1
                    else:
                        surviving.append(t_id)

                surviving_trees_for_q.append(surviving)
                timers[i].bound_eval_ms += elapsed_ms(t_root_filter_start)

            # Step 4: K2 Candidate leaf bounds evaluation on GPU
            all_surviving_trees = [t for trees in surviving_trees_for_q for t in trees]
            candidate_pairs: list[tuple[int, int, int, SpectrumMeta, float]] = []

            if len(all_surviving_trees) > 0:
                union_surviving_trees = np.unique(np.array(all_surviving_trees, dtype=np.int64))

                tree_to_leaf_range: dict[int, tuple[int, int]] = {}
                leaves_list: list[np.ndarray] = []
                curr_offset = 0
                for t_id in union_surviving_trees:
                    t_leaves = forest.trees.leaves_of_tree(int(t_id))
                    leaves_list.append(t_leaves)
                    n_l = len(t_leaves)
                    tree_to_leaf_range[int(t_id)] = (curr_offset, curr_offset + n_l)
                    curr_offset += n_l

                union_leaves = (
                    np.concatenate(leaves_list)
                    if leaves_list
                    else np.empty(0, dtype=np.int64)
                )

                if len(union_leaves) > 0:
                    t_k2_start = time.perf_counter()
                    d_leaf_bounds = batch_node_bounds_gpu(
                        batch_query, gpu_f, node_ids=union_leaves, stream=curr_stream
                    )
                    h_leaf_bounds = d_leaf_bounds.copy_to_host(stream=curr_stream)
                    if curr_stream is not None:
                        curr_stream.synchronize()
                    k2_elapsed = elapsed_ms(t_k2_start)
                    for i in range(B):
                        if has_peaks[i]:
                            timers[i].bound_eval_ms += k2_elapsed / B

                    # Step 5: Leaf filtering & host candidate spectrum pair extraction
                    for i in range(B):
                        surviving_trees = surviving_trees_for_q[i]
                        if not surviving_trees:
                            continue
                        t_host_start = time.perf_counter()
                        eff_theta = results[i].theta()
                        min_matched = batch_cfgs[i].min_matched_peaks

                        for t_id in surviving_trees:
                            l_start, l_end = tree_to_leaf_range[t_id]
                            nodes_visited[i] += (l_end - l_start)
                            for l_idx in range(l_start, l_end):
                                u_leaf = float(h_leaf_bounds[i, l_idx])
                                if (eff_theta > -np.inf and u_leaf < eff_theta) or (
                                    min_matched > 0 and u_leaf <= 0.0
                                ):
                                    leaves_pruned[i] += 1
                                else:
                                    lid = int(union_leaves[l_idx])
                                    id_start = int(forest.nodes.id_start[lid])
                                    id_end = int(forest.nodes.id_end[lid])
                                    for iid in range(id_start, id_end):
                                        row = int(forest.internal_to_row[iid])
                                        if row in scored_rows[i]:
                                            continue
                                        meta = spectra[row]
                                        if not is_eligible(batch_cfgs[i], meta):
                                            continue
                                        candidate_pairs.append((i, iid, row, meta, u_leaf))
                        timers[i].bound_eval_ms += elapsed_ms(t_host_start)

            # Step 6: K3a: Single-spectrum U_ind filtering on GPU
            exact_pairs: list[tuple[int, int, int, SpectrumMeta, float]] = []
            if len(candidate_pairs) > 0:
                if uind:
                    pair_q_indices = np.array([p[0] for p in candidate_pairs], dtype=np.int64)
                    pair_iids = np.array([p[1] for p in candidate_pairs], dtype=np.int64)

                    t_k3a_start = time.perf_counter()
                    d_uind = batch_uind_pairs_gpu(
                        batch_query=batch_query,
                        gpu_forest=gpu_f,
                        query_indices=pair_q_indices,
                        candidate_iids=pair_iids,
                        stream=curr_stream,
                        frag_tau=frag_tau,
                    )
                    h_uind = d_uind.copy_to_host(stream=curr_stream)
                    if curr_stream is not None:
                        curr_stream.synchronize()
                    k3a_elapsed = elapsed_ms(t_k3a_start)
                    for i in range(B):
                        if has_peaks[i]:
                            timers[i].bound_eval_ms += k3a_elapsed / B

                    for k, item in enumerate(candidate_pairs):
                        q_idx = item[0]
                        u_val = float(h_uind[k])
                        eff_theta = results[q_idx].theta()
                        if (eff_theta > -np.inf and u_val < eff_theta) or (
                            batch_cfgs[q_idx].min_matched_peaks > 0 and u_val <= 0.0
                        ):
                            uind_pruned[q_idx] += 1
                        else:
                            exact_pairs.append((item[0], item[1], item[2], item[3], u_val))
                else:
                    exact_pairs = [(p[0], p[1], p[2], p[3], p[4]) for p in candidate_pairs]

            # ASYNCHRONOUS OVERLAP: Launch next batch b+1 K1 while GPU performs exact scoring
            if next_prefetched is None:
                next_prefetched = _prefetch_batch(b_idx + 1)

            # Step 7: Exact scoring on GPU with dynamic theta updates and CPU overflow fallback
            if len(exact_pairs) > 0:
                pair_q_indices = np.array([p[0] for p in exact_pairs], dtype=np.int64)
                pair_iids = np.array([p[1] for p in exact_pairs], dtype=np.int64)

                t_exact_start = time.perf_counter()
                d_scores, d_matched, d_overflow = batch_greedy_cosine_pairs_gpu(
                    batch_query=batch_query,
                    gpu_forest=gpu_f,
                    query_indices=pair_q_indices,
                    candidate_iids=pair_iids,
                    stream=curr_stream,
                    frag_tau=frag_tau,
                )
                h_scores = d_scores.copy_to_host(stream=curr_stream)
                h_matched = d_matched.copy_to_host(stream=curr_stream)
                h_overflow = d_overflow.copy_to_host(stream=curr_stream)
                if curr_stream is not None:
                    curr_stream.synchronize()
                exact_elapsed = elapsed_ms(t_exact_start)
                for i in range(B):
                    if has_peaks[i]:
                        timers[i].exact_eval_ms += exact_elapsed / B

                q_to_exact_indices: dict[int, list[int]] = {}
                for k, item in enumerate(exact_pairs):
                    q_idx = item[0]
                    if q_idx not in q_to_exact_indices:
                        q_to_exact_indices[q_idx] = []
                    q_to_exact_indices[q_idx].append(k)

                for q_idx, k_list in q_to_exact_indices.items():
                    if mode == SearchMode.TOP_K:
                        k_list.sort(key=lambda k: -exact_pairs[k][4])
                    query = batch_queries[q_idx]
                    cfg = batch_cfgs[q_idx]
                    res = results[q_idx]
                    min_matched = cfg.min_matched_peaks
                    q_scored = scored_rows[q_idx]

                    for k in k_list:
                        _, iid, row, meta, u_val = exact_pairs[k]
                        if row in q_scored:
                            continue
                        eff_theta = res.theta()
                        if (eff_theta > -np.inf and u_val < eff_theta) or (
                            min_matched > 0 and u_val <= 0.0
                        ):
                            uind_pruned[q_idx] += 1
                            continue

                        n_scored[q_idx] += 1
                        q_scored.add(row)

                        if h_overflow[k] != 0:
                            member_peaks = forest.postings.spectrum_at(iid)
                            t_cpu_start = time.perf_counter()
                            score_res = score_greedy_cosine(query, member_peaks, frag_tau)
                            timers[q_idx].exact_eval_ms += elapsed_ms(t_cpu_start)
                            pair_score = score_res.score
                            pair_matched = score_res.n_matched
                        else:
                            pair_score = float(h_scores[k])
                            pair_matched = int(h_matched[k])

                        if pair_matched >= min_matched:
                            res.update(
                                SearchHit(
                                    score=pair_score,
                                    external_id=meta.external_id,
                                    spectrum_index=row,
                                    n_matched=pair_matched,
                                )
                            )

        # In case union_trees was empty or next_prefetched not yet launched
        if next_prefetched is None:
            next_prefetched = _prefetch_batch(b_idx + 1)

        # Step 8: Zero-score supplement & outcome assembly
        for i in range(B):
            cfg = batch_cfgs[i]
            res = results[i]
            if needs_zero_supplement(res, cfg):
                candidates = ZeroScoreCandidates(
                    member=forest.zero_energy_members.member,
                    ion_mode=forest.zero_energy_members.ion_mode,
                    precursor_mz=forest.zero_energy_members.precursor_mz,
                )
                supplement_zero_score(res, spectra, cfg, candidates, scored_rows[i])

            hits = res.finish()
            pruned_by_layer = {
                "roots_pruned": roots_pruned[i],
                "leaves_pruned": leaves_pruned[i],
                "uind_pruned": uind_pruned[i],
            }
            if probed_trees_for_q is not None:
                pruned_by_layer["probed_roots"] = len(probed_trees_for_q[i])
            stats = SearchStats(
                nodes_visited=nodes_visited[i],
                pruned_by_layer=pruned_by_layer,
                n_scored=n_scored[i],
                bound_eval_time_ms=timers[i].bound_eval_ms,
                exact_eval_time_ms=timers[i].exact_eval_ms,
                probe_scored=probe_scored[i],
            )
            versions = search_versions(library, cfg, default_version=forest.spec.versioned_id)
            all_outcomes.append(
                SearchOutcome(
                    mode=cfg.mode,
                    hits=hits,
                    complete=True,
                    stats=stats,
                    versions=versions,
                )
            )

    # Restore original query ordering if presorted
    if inv_order is not None:
        all_outcomes = [all_outcomes[inv_order[i]] for i in range(n_queries)]

    # Synchronize all streams to ensure device-side execution finishes cleanly
    for s in streams:
        if s is not None:
            s.synchronize()

    return all_outcomes


def search_threshold_batch_gpu(
    queries: Sequence[SpectrumPeaks],
    gpu_forest: GpuForestIndex | ForestIndex,
    library: PreprocessedLibrary | QueryConfig | Sequence[QueryConfig] | None = None,
    config: QueryConfig | Sequence[QueryConfig] | None = None,
    batch_size: int = 512,
    uind: bool = True,
    stream: Any = None,
    score_margin: float = DEFAULT_GPU_SCORE_MARGIN,
) -> list[SearchOutcome]:
    """Execute batch threshold search on GPU using multi-layer envelope bounds and U_ind filtering.

    Pipeline:
        1. Parameter validation & normalization.
        2. Precursor m/z adaptive presort bucketing (if precursor windows present and n_queries > batch_size).
        3. Double-buffered asynchronous overlap execution across batches.
        4. K1: Evaluate root envelope upper bounds on GPU, prune non-qualifying trees.
        5. K2: Expand surviving trees to leaf nodes, evaluate leaf envelope bounds on GPU.
        6. Expand surviving leaves into (q_idx, iid) pairs and filter with is_eligible().
        7. K3a: Evaluate single-spectrum U_ind upper bounds on GPU, filter non-qualifying pairs.
        8. Asynchronous launch of next batch K1 concurrently with Host-side CPU exact scoring.
        9. Exact score remaining pairs via scalar JIT score_greedy_cosine on CPU.
        10. Supplement zero scores if accepted, assemble SearchOutcome, and restore original query order.

    Args:
        queries: Sequence of query spectrum peaks.
        gpu_forest: Device-resident GpuForestIndex or host ForestIndex (automatically uploaded).
        library: Preprocessed library reference, or QueryConfig if config omitted.
        config: Single QueryConfig or sequence of QueryConfigs matching queries length.
        batch_size: Number of queries processed per batch on GPU (default 512).
        uind: Whether to enable single-spectrum U_ind upper bound pruning on GPU.
        stream: Optional CUDA stream for GPU execution (if None, uses double-buffering overlap streams).
        score_margin: Floating-point margin for threshold acceptance (default 5e-5).

    Returns:
        List of SearchOutcome instances matching queries in original input order.
    """
    require_cuda()

    # 1. Parameter normalization & defensive checks
    if isinstance(library, QueryConfig):
        config = library
        library = None
    elif isinstance(library, Sequence) and not isinstance(library, (str, bytes)):
        if len(library) == 0 or isinstance(library[0], QueryConfig):
            config = library
            library = None

    if config is None:
        raise ValueError("Must provide config parameter (QueryConfig or sequence of QueryConfig)")

    n_queries = len(queries)
    if n_queries == 0:
        return []

    for q in queries:
        validate_query(q)

    if isinstance(config, QueryConfig):
        cfg_list = [config] * n_queries
    elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        if len(config) != n_queries:
            raise ValueError(
                f"config sequence length ({len(config)}) does not match queries length ({n_queries})"
            )
        cfg_list = list(config)
    else:
        raise TypeError(f"Unknown config type: {type(config).__name__}")

    for cfg in cfg_list:
        if cfg.mode != SearchMode.THRESHOLD:
            if cfg.mode == SearchMode.TOP_K:
                raise NotImplementedError(
                    "GPU Top-K search is currently under development (Phase 4). "
                    "Please use SearchMode.THRESHOLD for search_threshold_batch_gpu or CPU search_forest_batch."
                )
            raise ValueError(f"Unsupported search mode: {cfg.mode}")

    # Forest & library resolution
    if isinstance(gpu_forest, GpuForestIndex):
        gpu_f = gpu_forest
        forest = gpu_forest.forest
    elif isinstance(gpu_forest, ForestIndex):
        gpu_f = GpuForestIndex.from_forest(gpu_forest, stream=stream)
        forest = gpu_forest
    else:
        raise TypeError(
            f"Expected GpuForestIndex or ForestIndex, got {type(gpu_forest).__name__}"
        )

    if library is None:
        if forest.spectra is None:
            raise ValueError("No library provided and forest snapshot lacks spectra metadata")
        spectra = forest.spectra
        n_spectra = forest.n_spectra
    else:
        if forest.n_spectra != library.n_spectra:
            raise ValueError(
                f"Forest spectrum count mismatch: index declares {forest.n_spectra}, library has {library.n_spectra}"
            )
        if forest.library_fingerprint and getattr(library, "fingerprint", None):
            if forest.library_fingerprint != library.fingerprint:
                raise ValueError(
                    f"Forest index fingerprint ({forest.library_fingerprint}) does not match library ({library.fingerprint})"
                )
        spectra = library.spectra
        n_spectra = library.n_spectra

    batch_size = max(1, int(batch_size))

    # Auto-bucket queries with different fragment_tolerance_da
    tolerances = {c.fragment_tolerance_da for c in cfg_list}
    if len(tolerances) > 1:
        tol_groups: dict[float, list[int]] = {}
        for idx, cfg in enumerate(cfg_list):
            tol = cfg.fragment_tolerance_da
            if tol not in tol_groups:
                tol_groups[tol] = []
            tol_groups[tol].append(idx)

        merged_outcomes: list[SearchOutcome | None] = [None] * n_queries
        for tol, indices in tol_groups.items():
            sub_queries = [queries[i] for i in indices]
            sub_cfgs = [cfg_list[i] for i in indices]
            sub_outcomes = search_threshold_batch_gpu(
                queries=sub_queries,
                gpu_forest=gpu_f,
                library=library,
                config=sub_cfgs,
                batch_size=batch_size,
                uind=uind,
                stream=stream,
                score_margin=score_margin,
            )
            if len(sub_outcomes) != len(indices):
                raise RuntimeError(
                    f"Tolerance sub-batch returned {len(sub_outcomes)} outcomes, expected {len(indices)}"
                )
            for orig_idx, outcome in zip(indices, sub_outcomes):
                merged_outcomes[orig_idx] = outcome

        assert all(o is not None for o in merged_outcomes), "Some query outcomes were not restored"
        return [o for o in merged_outcomes if o is not None]

    return _execute_gpu_batch_pipeline(
        queries=list(queries),
        cfgs=cfg_list,
        gpu_f=gpu_f,
        forest=forest,
        spectra=spectra,
        library=library,
        batch_size=batch_size,
        probe_trees=0,
        uind=uind,
        stream=stream,
        score_margin=score_margin,
        mode=SearchMode.THRESHOLD,
    )


def search_topk_batch_gpu(
    queries: Sequence[SpectrumPeaks],
    gpu_forest: GpuForestIndex | ForestIndex,
    library: PreprocessedLibrary | QueryConfig | Sequence[QueryConfig] | None = None,
    config: QueryConfig | Sequence[QueryConfig] | None = None,
    batch_size: int = 512,
    probe_trees: int = 3,
    uind: bool = True,
    stream: Any = None,
    score_margin: float = DEFAULT_GPU_SCORE_MARGIN,
) -> list[SearchOutcome]:
    """Execute batch Top-K search on GPU with speculative probe preheating, bulk pruning, and overlap pipeline.

    Algorithm:
        1. Parameter validation & normalization (verifies SearchMode.TOP_K).
        2. Precursor m/z adaptive presort bucketing (if precursor windows present and n_queries > batch_size).
        3. Double-buffered asynchronous overlap execution across batches.
        4. K1: Evaluate root envelope upper bounds on GPU (batch_root_bounds_gpu).
        5. Speculative Probe Phase (CPU):
           - Sort eligible trees per query descending by root upper bound.
           - Expand and exact-score candidate spectra from the top 1-3 trees to immediately
             elevate dynamic threshold theta from -inf to a high baseline (e.g. 0.75-0.95+).
        6. Bulk Pruning Phase:
           - Filter out trees where U_root < current theta (pruning >95% of remaining trees).
           - K2: Expand surviving trees to leaves and evaluate leaf envelope bounds on GPU.
           - Filter leaves where U_leaf < current theta.
           - K3a: Evaluate single-spectrum U_ind upper bounds on GPU for surviving candidate pairs.
           - Asynchronous launch of next batch K1 concurrently with Host-side CPU exact scoring.
           - Exact score surviving candidate pairs on CPU, sorted descending by upper bound
             with dynamic theta early-stopping.
        7. Zero-score supplement (if min_matched_peaks == 0 and heap not full), assembly, and order restoration.

    Args:
        queries: Sequence of query spectrum peaks.
        gpu_forest: Device-resident GpuForestIndex or host ForestIndex (automatically uploaded).
        library: Preprocessed library reference, or QueryConfig if config omitted.
        config: Single QueryConfig or sequence of QueryConfigs matching queries length.
        batch_size: Number of queries processed per batch on GPU (default 512).
        probe_trees: Number of highest root-bound trees to greedily probe per query (default 3).
        uind: Whether to enable single-spectrum U_ind upper bound pruning on GPU.
        stream: Optional CUDA stream for GPU execution (if None, uses double-buffering overlap streams).
        score_margin: Floating-point margin for threshold acceptance (default 5e-5).

    Returns:
        List of SearchOutcome instances matching queries in original input order.
    """
    require_cuda()

    # 1. Parameter normalization & defensive checks
    if isinstance(library, QueryConfig):
        config = library
        library = None
    elif isinstance(library, Sequence) and not isinstance(library, (str, bytes)):
        if len(library) == 0 or isinstance(library[0], QueryConfig):
            config = library
            library = None

    if config is None:
        raise ValueError("Must provide config parameter (QueryConfig or sequence of QueryConfig)")

    n_queries = len(queries)
    if n_queries == 0:
        return []

    for q in queries:
        validate_query(q)

    if isinstance(config, QueryConfig):
        cfg_list = [config] * n_queries
    elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        if len(config) != n_queries:
            raise ValueError(
                f"config sequence length ({len(config)}) does not match queries length ({n_queries})"
            )
        cfg_list = list(config)
    else:
        raise TypeError(f"Unknown config type: {type(config).__name__}")

    for cfg in cfg_list:
        if cfg.mode != SearchMode.TOP_K:
            if cfg.mode == SearchMode.THRESHOLD:
                raise ValueError(
                    "SearchMode.THRESHOLD is not supported by search_topk_batch_gpu. "
                    "Please use search_threshold_batch_gpu or search_forest_batch_gpu."
                )
            raise ValueError(f"Unsupported search mode: {cfg.mode}")
        if cfg.k is None:
            raise ValueError("QueryConfig.k must not be None in TOP_K mode")

    # Forest & library resolution
    if isinstance(gpu_forest, GpuForestIndex):
        gpu_f = gpu_forest
        forest = gpu_forest.forest
    elif isinstance(gpu_forest, ForestIndex):
        gpu_f = GpuForestIndex.from_forest(gpu_forest, stream=stream)
        forest = gpu_forest
    else:
        raise TypeError(
            f"Expected GpuForestIndex or ForestIndex, got {type(gpu_forest).__name__}"
        )

    if library is None:
        if forest.spectra is None:
            raise ValueError("No library provided and forest snapshot lacks spectra metadata")
        spectra = forest.spectra
        n_spectra = forest.n_spectra
    else:
        if forest.n_spectra != library.n_spectra:
            raise ValueError(
                f"Forest spectrum count mismatch: index declares {forest.n_spectra}, library has {library.n_spectra}"
            )
        if forest.library_fingerprint and getattr(library, "fingerprint", None):
            if forest.library_fingerprint != library.fingerprint:
                raise ValueError(
                    f"Forest index fingerprint ({forest.library_fingerprint}) does not match library ({library.fingerprint})"
                )
        spectra = library.spectra
        n_spectra = library.n_spectra

    batch_size = max(1, int(batch_size))

    # Auto-bucket queries with different fragment_tolerance_da
    tolerances = {c.fragment_tolerance_da for c in cfg_list}
    if len(tolerances) > 1:
        tol_groups: dict[float, list[int]] = {}
        for idx, cfg in enumerate(cfg_list):
            tol = cfg.fragment_tolerance_da
            if tol not in tol_groups:
                tol_groups[tol] = []
            tol_groups[tol].append(idx)

        merged_outcomes: list[SearchOutcome | None] = [None] * n_queries
        for tol, indices in tol_groups.items():
            sub_queries = [queries[i] for i in indices]
            sub_cfgs = [cfg_list[i] for i in indices]
            sub_outcomes = search_topk_batch_gpu(
                queries=sub_queries,
                gpu_forest=gpu_f,
                library=library,
                config=sub_cfgs,
                batch_size=batch_size,
                probe_trees=probe_trees,
                uind=uind,
                stream=stream,
                score_margin=score_margin,
            )
            if len(sub_outcomes) != len(indices):
                raise RuntimeError(
                    f"Tolerance sub-batch returned {len(sub_outcomes)} outcomes, expected {len(indices)}"
                )
            for orig_idx, outcome in zip(indices, sub_outcomes):
                merged_outcomes[orig_idx] = outcome

        assert all(o is not None for o in merged_outcomes), "Some query outcomes were not restored"
        return [o for o in merged_outcomes if o is not None]

    return _execute_gpu_batch_pipeline(
        queries=list(queries),
        cfgs=cfg_list,
        gpu_f=gpu_f,
        forest=forest,
        spectra=spectra,
        library=library,
        batch_size=batch_size,
        probe_trees=probe_trees,
        uind=uind,
        stream=stream,
        score_margin=score_margin,
        mode=SearchMode.TOP_K,
    )


def search_forest_batch_gpu(
    queries: Sequence[SpectrumPeaks],
    gpu_forest: GpuForestIndex | ForestIndex,
    library: PreprocessedLibrary | QueryConfig | Sequence[QueryConfig] | None = None,
    config: QueryConfig | Sequence[QueryConfig] | None = None,
    batch_size: int = 512,
    probe_trees: int = 3,
    uind: bool = True,
    stream: Any = None,
    score_margin: float = DEFAULT_GPU_SCORE_MARGIN,
) -> list[SearchOutcome]:
    """Unified GPU-accelerated batch search dispatcher.

    Routes THRESHOLD search requests to search_threshold_batch_gpu,
    and TOP_K search requests to search_topk_batch_gpu.
    Supports single QueryConfig or per-query QueryConfig sequences.
    Automatically handles mixed SearchModes and mixed tolerances across queries.
    """
    if isinstance(library, QueryConfig):
        config = library
        library = None
    elif isinstance(library, Sequence) and not isinstance(library, (str, bytes)):
        if len(library) == 0 or isinstance(library[0], QueryConfig):
            config = library
            library = None

    if config is None:
        raise ValueError("Must provide config parameter (QueryConfig or sequence of QueryConfig)")

    n_queries = len(queries)
    if n_queries == 0:
        return []

    if isinstance(config, QueryConfig):
        cfg_list = [config] * n_queries
    elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        if len(config) != n_queries:
            raise ValueError(
                f"config sequence length ({len(config)}) does not match queries length ({n_queries})"
            )
        cfg_list = list(config)
    else:
        raise TypeError(f"Unknown config type: {type(config).__name__}")

    # Check for mixed modes across queries
    modes = {c.mode for c in cfg_list}
    if len(modes) > 1:
        mode_groups: dict[SearchMode, list[int]] = {}
        for idx, cfg in enumerate(cfg_list):
            if cfg.mode not in mode_groups:
                mode_groups[cfg.mode] = []
            mode_groups[cfg.mode].append(idx)

        merged_outcomes: list[SearchOutcome | None] = [None] * n_queries
        for m, indices in mode_groups.items():
            sub_queries = [queries[i] for i in indices]
            sub_cfgs = [cfg_list[i] for i in indices]
            sub_outcomes = search_forest_batch_gpu(
                queries=sub_queries,
                gpu_forest=gpu_forest,
                library=library,
                config=sub_cfgs,
                batch_size=batch_size,
                probe_trees=probe_trees,
                uind=uind,
                stream=stream,
                score_margin=score_margin,
            )
            if len(sub_outcomes) != len(indices):
                raise RuntimeError(
                    f"Mode sub-batch returned {len(sub_outcomes)} outcomes, expected {len(indices)}"
                )
            for orig_idx, outcome in zip(indices, sub_outcomes):
                merged_outcomes[orig_idx] = outcome

        assert all(o is not None for o in merged_outcomes), "Some query outcomes were not restored"
        return [o for o in merged_outcomes if o is not None]

    mode = next(iter(modes))
    if mode == SearchMode.THRESHOLD:
        return search_threshold_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=library,
            config=cfg_list,
            batch_size=batch_size,
            uind=uind,
            stream=stream,
            score_margin=score_margin,
        )
    elif mode == SearchMode.TOP_K:
        return search_topk_batch_gpu(
            queries=queries,
            gpu_forest=gpu_forest,
            library=library,
            config=cfg_list,
            batch_size=batch_size,
            probe_trees=probe_trees,
            uind=uind,
            stream=stream,
            score_margin=score_margin,
        )
    else:
        raise ValueError(f"Unsupported search mode: {mode}")
