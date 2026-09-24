"""N-to-M Batch Benchmark for BLINK vs JET-Forest."""

import time
import json
import numpy as np
from pathlib import Path
from jetf.benchmarks.dataset import load_benchmark_dataset, slice_parsed_library
from jetf.benchmarks.blink_adapter import BlinkBenchmarkEngine, check_blink_available
from jetf.preprocessing import preprocess_library
from jetf.builder import build_forest_index
from jetf.structure import DEFAULT_FOREST_SPEC
from jetf.query import QueryConfig, SearchMode, IonModePolicy
from jetf.search import search_forest_batch

def run_ntom_benchmark():
    check_blink_available()
    print("Loading dataset GNPS-LIBRARY.mgf...")
    dataset = load_benchmark_dataset(Path("GNPS-LIBRARY.mgf"), library_size=None)
    total_spectra = dataset.n_spectra
    print(f"Total available spectra: {total_spectra}")

    # Scale combinations: (M_queries, N_references)
    mn_scales = [
        (50, 500),
        (100, 1000),
        (200, 2000),
        (500, 5000),
        (1000, 10000),
    ]

    results = []

    q_cfg = QueryConfig(
        mode=SearchMode.TOP_K,
        k=10,
        fragment_tolerance_da=0.01,
        min_matched_peaks=1,
        ion_mode_policy=IonModePolicy.ANY,
    )

    for m_queries, n_refs in mn_scales:
        if m_queries + n_refs > total_spectra:
            continue

        print(f"\n--- Running N-to-M Benchmark: M={m_queries} queries vs N={n_refs} references ({m_queries * n_refs:,} comparisons) ---")
        
        # Prepare reference library
        ref_indices = np.arange(n_refs, dtype=np.int64)
        sub_parsed_ref = slice_parsed_library(dataset.parsed, ref_indices)
        sub_lib_ref = preprocess_library(sub_parsed_ref, dataset.library.spec)
        sub_forest = build_forest_index(sub_lib_ref, DEFAULT_FOREST_SPEC)

        # Prepare queries (disjoint from references to avoid identity triviality)
        query_indices = np.arange(n_refs, n_refs + m_queries, dtype=np.int64)
        sub_parsed_q = slice_parsed_library(dataset.parsed, query_indices)
        sub_lib_q = preprocess_library(sub_parsed_q, dataset.library.spec)
        query_peaks = [sub_lib_q.peaks.spectrum_at(i) for i in range(m_queries)]

        # Build BLINK Engine
        t_b_build_start = time.perf_counter()
        blink_engine = BlinkBenchmarkEngine(
            sub_lib_ref, tolerance=0.01, bin_width=0.001, intensity_power=0.5
        )
        t_b_build = time.perf_counter() - t_b_build_start

        # Warmup
        blink_engine.score_batch(query_peaks[:min(10, m_queries)], top_k=10)
        search_forest_batch(query_peaks[:min(10, m_queries)], sub_forest, sub_lib_ref, q_cfg, concurrency=1)

        # 1. BLINK Batch Score
        t0 = time.perf_counter()
        blink_res = blink_engine.score_batch(query_peaks, top_k=10)
        t_blink_total = time.perf_counter() - t0
        blink_qps = m_queries / t_blink_total

        # 2. JET-Forest Batch Score (Concurrency = 1, Single-threaded)
        t0 = time.perf_counter()
        search_forest_batch(query_peaks, sub_forest, sub_lib_ref, q_cfg, concurrency=1)
        t_jetf_c1 = time.perf_counter() - t0
        jetf_c1_qps = m_queries / t_jetf_c1
        jetf_c1_speedup = t_blink_total / t_jetf_c1

        # 3. JET-Forest Batch Score (Concurrency = 4, Multi-threaded)
        t0 = time.perf_counter()
        search_forest_batch(query_peaks, sub_forest, sub_lib_ref, q_cfg, concurrency=4)
        t_jetf_c4 = time.perf_counter() - t0
        jetf_c4_qps = m_queries / t_jetf_c4
        jetf_c4_speedup = t_blink_total / t_jetf_c4

        # 4. JET-Forest Batch Score (Concurrency = 8, Multi-threaded)
        t0 = time.perf_counter()
        search_forest_batch(query_peaks, sub_forest, sub_lib_ref, q_cfg, concurrency=8)
        t_jetf_c8 = time.perf_counter() - t0
        jetf_c8_qps = m_queries / t_jetf_c8
        jetf_c8_speedup = t_blink_total / t_jetf_c8

        print(f"  BLINK Total Time: {t_blink_total*1000:.2f} ms (Disc: {blink_res.discretize_time_s*1000:.2f} ms, Score: {blink_res.score_time_s*1000:.2f} ms) | QPS: {blink_qps:.1f}")
        print(f"  JETF (c=1) Time:  {t_jetf_c1*1000:.2f} ms | QPS: {jetf_c1_qps:.1f} | Speedup vs BLINK: {jetf_c1_speedup:.2f}x")
        print(f"  JETF (c=4) Time:  {t_jetf_c4*1000:.2f} ms | QPS: {jetf_c4_qps:.1f} | Speedup vs BLINK: {jetf_c4_speedup:.2f}x")
        print(f"  JETF (c=8) Time:  {t_jetf_c8*1000:.2f} ms | QPS: {jetf_c8_qps:.1f} | Speedup vs BLINK: {jetf_c8_speedup:.2f}x")

        row = {
            "m_queries": m_queries,
            "n_refs": n_refs,
            "comparisons": m_queries * n_refs,
            "blink_total_ms": t_blink_total * 1000,
            "blink_disc_ms": blink_res.discretize_time_s * 1000,
            "blink_score_ms": blink_res.score_time_s * 1000,
            "blink_qps": blink_qps,
            "jetf_c1_ms": t_jetf_c1 * 1000,
            "jetf_c1_qps": jetf_c1_qps,
            "jetf_c1_speedup": jetf_c1_speedup,
            "jetf_c4_ms": t_jetf_c4 * 1000,
            "jetf_c4_qps": jetf_c4_qps,
            "jetf_c4_speedup": jetf_c4_speedup,
            "jetf_c8_ms": t_jetf_c8 * 1000,
            "jetf_c8_qps": jetf_c8_qps,
            "jetf_c8_speedup": jetf_c8_speedup,
        }
        results.append(row)

    with open("docs/benchmark-blink-ntom.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved docs/benchmark-blink-ntom.json successfully!")

if __name__ == "__main__":
    run_ntom_benchmark()
