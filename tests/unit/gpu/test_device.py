"""Unit tests for GPU resident structures and device memory migration."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from numba import cuda

from jetf import (
    DEFAULT_FOREST_SPEC,
    IonMode,
    ParsedLibrary,
    SourceRef,
    SpectrumMeta,
    build_forest_index,
    preprocess_library,
)
from jetf.gpu import GpuForestIndex, is_cuda_available, require_cuda

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _subset import stratified_subset_indices, subset_parsed_library
from jetf import CORRECTNESS_V1, parse_mgf

GNPS_PATH = Path(__file__).resolve().parents[3] / "GNPS-LIBRARY.mgf"


def make_synthetic_library(n_spectra: int = 24) -> ParsedLibrary:
    """Create a deterministic synthetic ParsedLibrary for fast GPU unit testing."""
    metas: list[SpectrumMeta] = []
    masses: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    peak_ids: list[np.ndarray] = []
    offsets: list[int] = [0]

    for i in range(n_spectra):
        mode = IonMode.POSITIVE if i % 2 == 0 else IonMode.NEGATIVE
        metas.append(
            SpectrumMeta(
                external_id=f"syn_spec_{i}",
                precursor_mz=150.0 + i * 15.0,
                charge=1,
                ion_mode=mode,
                source=SourceRef("syn_test.mgf", i),
            )
        )
        n_peaks = 3 + (i % 5)
        m = np.linspace(50.0, 50.0 + 20.0 * n_peaks, n_peaks, dtype=np.float64)
        it = np.linspace(10.0, 100.0, n_peaks, dtype=np.float64)
        pid = np.arange(n_peaks, dtype=np.int64)
        masses.append(m)
        intensities.append(it)
        peak_ids.append(pid)
        offsets.append(offsets[-1] + n_peaks)

    return ParsedLibrary(
        source_path="syn_test.mgf",
        spectra=tuple(metas),
        mass=np.concatenate(masses),
        intensity=np.concatenate(intensities),
        peak_id=np.concatenate(peak_ids),
        spectrum_offsets=np.array(offsets, dtype=np.int64),
    )


@pytest.fixture(scope="module")
def synthetic_forest():
    parsed = make_synthetic_library(n_spectra=24)
    prep = preprocess_library(parsed)
    return build_forest_index(prep)


def test_is_cuda_available_safety():
    """Verify is_cuda_available returns boolean and handles failure modes gracefully."""
    avail = is_cuda_available()
    assert isinstance(avail, bool)


def test_require_cuda_exception(monkeypatch):
    """Verify require_cuda raises RuntimeError with clear message when CUDA is unavailable."""
    import jetf.gpu as gpu_module

    monkeypatch.setattr(gpu_module, "is_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        gpu_module.require_cuda()


def test_gpu_module_lazy_import():
    """Verify lazy/conditional export of GpuForestIndex and __all__ exports."""
    import jetf.gpu as jgpu

    assert "is_cuda_available" in jgpu.__all__
    assert "require_cuda" in jgpu.__all__
    assert "GpuForestIndex" in jgpu.__all__
    assert jgpu.GpuForestIndex is GpuForestIndex

    with pytest.raises(AttributeError):
        _ = jgpu.NonExistentClass


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_forest_from_synthetic_forest(synthetic_forest):
    """Test GpuForestIndex creation, shapes, dtypes, values, and memory statistics."""
    forest = synthetic_forest
    gpu_forest = GpuForestIndex.from_forest(forest)

    # 1. Forest reference and scalar properties
    assert gpu_forest.forest is forest
    assert gpu_forest.n_trees == forest.n_trees
    assert gpu_forest.n_nodes == forest.n_nodes
    assert gpu_forest.n_spectra == forest.n_spectra
    assert gpu_forest.grid_da == forest.envelopes.grid_da
    assert gpu_forest.spec == forest.spec
    assert gpu_forest.partitions == forest.partitions

    # 2. Verify Envelopes device arrays
    assert gpu_forest.cell_index.shape == forest.envelopes.cell_index.shape
    assert gpu_forest.cell_index.dtype == np.int64
    np.testing.assert_array_equal(gpu_forest.cell_index.copy_to_host(), forest.envelopes.cell_index)

    assert gpu_forest.max_peak_amplitude.shape == forest.envelopes.max_peak_amplitude.shape
    assert gpu_forest.max_peak_amplitude.dtype == np.float32
    np.testing.assert_allclose(
        gpu_forest.max_peak_amplitude.copy_to_host(),
        forest.envelopes.max_peak_amplitude.astype(np.float32),
        rtol=1e-6,
        atol=1e-7,
    )

    assert gpu_forest.node_envelope_offsets.shape == forest.envelopes.node_envelope_offsets.shape
    assert gpu_forest.node_envelope_offsets.dtype == np.int64
    np.testing.assert_array_equal(
        gpu_forest.node_envelope_offsets.copy_to_host(),
        forest.envelopes.node_envelope_offsets,
    )

    # 3. Verify Trees device arrays
    assert gpu_forest.precursor_min.shape == forest.trees.precursor_min.shape
    assert gpu_forest.precursor_min.dtype == np.float64
    np.testing.assert_allclose(gpu_forest.precursor_min.copy_to_host(), forest.trees.precursor_min)

    assert gpu_forest.precursor_max.shape == forest.trees.precursor_max.shape
    assert gpu_forest.precursor_max.dtype == np.float64
    np.testing.assert_allclose(gpu_forest.precursor_max.copy_to_host(), forest.trees.precursor_max)

    assert gpu_forest.root_node_id.shape == forest.trees.root_node_id.shape
    assert gpu_forest.root_node_id.dtype == np.int64
    np.testing.assert_array_equal(gpu_forest.root_node_id.copy_to_host(), forest.trees.root_node_id)

    assert gpu_forest.leaf_offsets.shape == forest.trees.leaf_offsets.shape
    assert gpu_forest.leaf_offsets.dtype == np.int64
    np.testing.assert_array_equal(gpu_forest.leaf_offsets.copy_to_host(), forest.trees.leaf_offsets)

    assert gpu_forest.leaf_node_ids.shape == forest.trees.leaf_node_ids.shape
    assert gpu_forest.leaf_node_ids.dtype == np.int64
    np.testing.assert_array_equal(gpu_forest.leaf_node_ids.copy_to_host(), forest.trees.leaf_node_ids)

    # 4. Verify Nodes device arrays
    assert gpu_forest.id_start.shape == forest.nodes.id_start.shape
    assert gpu_forest.id_start.dtype == np.int64
    np.testing.assert_array_equal(gpu_forest.id_start.copy_to_host(), forest.nodes.id_start)

    assert gpu_forest.id_end.shape == forest.nodes.id_end.shape
    assert gpu_forest.id_end.dtype == np.int64
    np.testing.assert_array_equal(gpu_forest.id_end.copy_to_host(), forest.nodes.id_end)

    assert gpu_forest.is_leaf.shape == forest.nodes.is_leaf.shape
    assert gpu_forest.is_leaf.dtype == bool
    np.testing.assert_array_equal(gpu_forest.is_leaf.copy_to_host(), forest.nodes.is_leaf)

    assert gpu_forest.tree_id.shape == forest.nodes.tree_id.shape
    assert gpu_forest.tree_id.dtype == np.int64
    np.testing.assert_array_equal(gpu_forest.tree_id.copy_to_host(), forest.nodes.tree_id)

    # 5. Verify Postings device arrays (strictly without energy)
    assert gpu_forest.mass.shape == forest.postings.mass.shape
    assert gpu_forest.mass.dtype == forest.postings.mass.dtype
    np.testing.assert_allclose(gpu_forest.mass.copy_to_host(), forest.postings.mass)

    assert gpu_forest.intensity.shape == forest.postings.intensity.shape
    assert gpu_forest.intensity.dtype == np.float32
    np.testing.assert_allclose(
        gpu_forest.intensity.copy_to_host(),
        forest.postings.intensity.astype(np.float32),
        rtol=1e-6,
        atol=1e-7,
    )

    assert gpu_forest.peak_id.shape == forest.postings.peak_id.shape
    assert gpu_forest.peak_id.dtype == forest.postings.peak_id.dtype
    np.testing.assert_array_equal(gpu_forest.peak_id.copy_to_host(), forest.postings.peak_id)

    assert gpu_forest.spectrum_offsets.shape == forest.postings.spectrum_offsets.shape
    assert gpu_forest.spectrum_offsets.dtype == np.int64
    np.testing.assert_array_equal(
        gpu_forest.spectrum_offsets.copy_to_host(),
        forest.postings.spectrum_offsets,
    )

    assert gpu_forest.norm.shape == forest.postings.norm.shape
    assert gpu_forest.norm.dtype == np.float64
    np.testing.assert_allclose(gpu_forest.norm.copy_to_host(), forest.postings.norm)

    # 6. Verify postings.energy is NOT uploaded to device
    assert not hasattr(gpu_forest, "energy")
    assert not hasattr(gpu_forest.postings, "energy")
    with pytest.raises(AttributeError):
        _ = gpu_forest.energy
    with pytest.raises(AttributeError):
        _ = gpu_forest.postings.energy

    # 7. Device memory statistics
    summary = gpu_forest.device_memory_summary()
    assert set(summary.keys()) == {"envelopes", "trees", "nodes", "postings"}
    assert summary["envelopes"] > 0
    assert summary["trees"] > 0
    assert summary["nodes"] > 0
    assert summary["postings"] > 0

    total_bytes = gpu_forest.device_memory_bytes()
    assert total_bytes > 0
    assert total_bytes == sum(summary.values())


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_forest_stream_transfer(synthetic_forest):
    """Test GpuForestIndex transfer with custom CUDA stream."""
    stream = cuda.stream()
    gpu_forest = GpuForestIndex.from_forest(synthetic_forest, stream=stream)
    stream.synchronize()

    assert gpu_forest.n_trees == synthetic_forest.n_trees
    assert gpu_forest.device_memory_bytes() > 0
    np.testing.assert_array_equal(
        gpu_forest.cell_index.copy_to_host(stream=stream),
        synthetic_forest.envelopes.cell_index,
    )
    stream.synchronize()


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_gpu_forest_subset_library():
    """Verify GPU migration on real GNPS-derived subset library."""
    if not GNPS_PATH.exists():
        pytest.skip(f"GNPS library not found at {GNPS_PATH}")

    parsed = parse_mgf(GNPS_PATH)
    indices = stratified_subset_indices(parsed, n_per_stratum=20)
    sub_parsed = subset_parsed_library(parsed, indices)
    lib = preprocess_library(sub_parsed, CORRECTNESS_V1)
    forest = build_forest_index(lib, DEFAULT_FOREST_SPEC)

    gpu_forest = GpuForestIndex.from_forest(forest)
    assert gpu_forest.n_spectra == forest.n_spectra
    assert gpu_forest.n_trees == forest.n_trees
    assert gpu_forest.n_nodes == forest.n_nodes
    assert gpu_forest.device_memory_bytes() > 0

    summary = gpu_forest.device_memory_summary()
    assert summary["envelopes"] > 0
    assert summary["postings"] > 0
    assert not hasattr(gpu_forest, "energy")
    assert not hasattr(gpu_forest.postings, "energy")
