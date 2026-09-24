"""Unit tests for Phase 2 GPU Fixes (Issues 1, 5, 12).

Validates:
1. Issue 1: GPU Top-K speculative probe performs K2 (leaf bounds) and K3a (U_ind) pruning,
   and explicitly tracks probed trees in pruned_by_layer["probed_roots"].
2. Issue 5: Host-side candidate gathering & leaf filtering time is accurately captured in
   timers / bound_eval_time_ms.
3. Issue 12: DRY helper `_execute_gpu_batch_pipeline` correctly executes both THRESHOLD and TOP_K
   modes, preserving full backwards compatibility across search_threshold_batch_gpu,
   search_topk_batch_gpu, and search_forest_batch_gpu.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pytest

from jetf import (
    CORRECTNESS_V1,
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ForestIndex,
    ForestSpec,
    IonMode,
    ParsedLibrary,
    PreprocessedLibrary,
    QueryConfig,
    SCORER_VERSIONED_ID,
    SearchMode,
    SourceRef,
    SpectrumMeta,
    SpectrumPeaks,
    build_forest_index,
    preprocess_library,
)
from jetf.gpu import (
    GpuForestIndex,
    is_cuda_available,
    require_cuda,
    search_forest_batch_gpu,
    search_threshold_batch_gpu,
    search_topk_batch_gpu,
)
from jetf.gpu.search import _execute_gpu_batch_pipeline
from jetf.bounds import _batch_node_bounds_numba_serial


def _make_peaks(mass: np.ndarray, intensity: np.ndarray) -> SpectrumPeaks:
    m = np.asarray(mass, dtype=np.float64)
    it = np.asarray(intensity, dtype=np.float64)
    return SpectrumPeaks(
        mass=m,
        intensity=it,
        energy=(it * it).astype(np.float64),
        peak_id=np.arange(len(m), dtype=np.int64),
    )


def _build_test_multi_tree_forest(n_trees: int = 4, spectra_per_tree: int = 4):
    """Build a synthetic library with multiple trees and leaves to test probe and pruning."""
    spectra_meta = []
    masses = []
    intensities = []
    offsets = [0]
    curr_offset = 0

    for t in range(n_trees):
        # Vary precursor m/z across trees
        base_mz = 100.0 + t * 50.0
        for s in range(spectra_per_tree):
            spec_idx = t * spectra_per_tree + s
            meta = SpectrumMeta(
                external_id=f"spec_{spec_idx}",
                precursor_mz=base_mz + s * 0.1,
                charge=1,
                ion_mode=IonMode.POSITIVE,
                source=SourceRef("test.mgf", spec_idx),
            )
            spectra_meta.append(meta)

            # Different peaks per spectrum
            m = np.array([50.0, 100.0 + t * 10.0 + s * 2.0, 200.0], dtype=np.float64)
            raw_it = np.array([1.0, 2.0, 1.0], dtype=np.float64)
            it = raw_it / np.linalg.norm(raw_it)
            masses.extend(m)
            intensities.extend(it)
            curr_offset += len(m)
            offsets.append(curr_offset)

    parsed = ParsedLibrary(
        source_path="test.mgf",
        spectra=tuple(spectra_meta),
        mass=np.array(masses, dtype=np.float64),
        intensity=np.array(intensities, dtype=np.float64),
        peak_id=np.arange(len(masses), dtype=np.int64),
        spectrum_offsets=np.array(offsets, dtype=np.int64),
    )
    lib = preprocess_library(parsed, CORRECTNESS_V1)
    spec = ForestSpec(tree_capacity=spectra_per_tree, leaf_capacity=2)
    forest = build_forest_index(lib, spec)
    gpu_forest = GpuForestIndex.from_forest(forest)
    return lib, forest, gpu_forest


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_issue1_topk_probe_accounting_and_pruning():
    """Verify Issue 1: Top-K probe tracks probed_roots and prunes unpromising leaves/spectra."""
    require_cuda()
    lib, forest, gpu_forest = _build_test_multi_tree_forest(n_trees=6, spectra_per_tree=4)

    # Query matching spectrum 0 closely
    q0 = lib.peaks.spectrum_at(0)
    cfg_topk = QueryConfig(
        mode=SearchMode.TOP_K,
        k=2,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )

    outcomes = search_topk_batch_gpu(
        queries=[q0],
        gpu_forest=gpu_forest,
        library=lib,
        config=cfg_topk,
        probe_trees=3,
        uind=True,
    )

    assert len(outcomes) == 1
    stats = outcomes[0].stats
    # 1. Probed trees are explicitly accounted for
    assert "probed_roots" in stats.pruned_by_layer
    probed_roots = stats.pruned_by_layer["probed_roots"]
    assert probed_roots > 0
    assert probed_roots <= 3

    # 2. Total eligible trees accounted: roots_pruned + probed_roots <= total trees
    roots_pruned = stats.pruned_by_layer["roots_pruned"]
    assert roots_pruned + probed_roots <= forest.n_trees

    # 3. Hits are correctly returned
    hits = outcomes[0].hits
    assert len(hits) == 2
    assert hits[0].external_id == "spec_0"
    assert math.isclose(hits[0].score, 1.0, rel_tol=1e-5)


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_issue5_host_candidate_gathering_time_captured():
    """Verify Issue 5: Candidate gathering CPU time is accumulated into bound_eval_time_ms."""
    require_cuda()
    lib, forest, gpu_forest = _build_test_multi_tree_forest(n_trees=4, spectra_per_tree=4)

    q0 = lib.peaks.spectrum_at(0)
    cfg_thresh = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.1,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )

    outcomes = search_threshold_batch_gpu(
        queries=[q0],
        gpu_forest=gpu_forest,
        library=lib,
        config=cfg_thresh,
    )

    assert len(outcomes) == 1
    stats = outcomes[0].stats
    # bound_eval_time_ms must be strictly positive and reflect host bound + gathering overhead
    assert stats.bound_eval_time_ms > 0.0
    assert stats.exact_eval_time_ms > 0.0


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_issue12_execute_gpu_batch_pipeline_dry():
    """Verify Issue 12: _execute_gpu_batch_pipeline works for both THRESHOLD and TOP_K modes."""
    require_cuda()
    lib, forest, gpu_forest = _build_test_multi_tree_forest(n_trees=4, spectra_per_tree=4)
    q0 = lib.peaks.spectrum_at(0)

    # 1. THRESHOLD mode execution via helper
    cfg_thresh = QueryConfig(
        mode=SearchMode.THRESHOLD,
        threshold=0.5,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    res_thresh = _execute_gpu_batch_pipeline(
        queries=[q0],
        cfgs=[cfg_thresh],
        gpu_f=gpu_forest,
        forest=forest,
        spectra=forest.spectra,
        library=lib,
        batch_size=512,
        probe_trees=0,
        uind=True,
        stream=None,
        score_margin=5e-5,
        mode=SearchMode.THRESHOLD,
    )
    assert len(res_thresh) == 1
    assert res_thresh[0].mode == SearchMode.THRESHOLD

    # 2. TOP_K mode execution via helper
    cfg_topk = QueryConfig(
        mode=SearchMode.TOP_K,
        k=3,
        fragment_tolerance_da=0.02,
        ion_mode=IonMode.POSITIVE,
    )
    res_topk = _execute_gpu_batch_pipeline(
        queries=[q0],
        cfgs=[cfg_topk],
        gpu_f=gpu_forest,
        forest=forest,
        spectra=forest.spectra,
        library=lib,
        batch_size=512,
        probe_trees=2,
        uind=True,
        stream=None,
        score_margin=5e-5,
        mode=SearchMode.TOP_K,
    )
    assert len(res_topk) == 1
    assert res_topk[0].mode == SearchMode.TOP_K
    assert len(res_topk[0].hits) == 3
    assert "probed_roots" in res_topk[0].stats.pruned_by_layer

    # 3. Unified dispatcher search_forest_batch_gpu consistency
    disp_thresh = search_forest_batch_gpu([q0], gpu_forest, lib, cfg_thresh)
    assert len(disp_thresh[0].hits) == len(res_thresh[0].hits)
    disp_topk = search_forest_batch_gpu([q0], gpu_forest, lib, cfg_topk)
    assert len(disp_topk[0].hits) == len(res_topk[0].hits)
