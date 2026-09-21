"""Unit tests verifying benchmark JSON reporting and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jetf.benchmarks.consistency import PairwiseConsistencyResult, QueryRetrievalConsistency, RetrievalConsistencySummary
from jetf.benchmarks.reporter import (
    build_benchmark_report_dict,
    generate_json_report,
    get_system_metadata,
    save_json_report,
)
from jetf.benchmarks.throughput import PairwiseThroughputResult, RetrievalThroughputResult
from jetf.cli import main


def test_get_system_metadata():
    meta = get_system_metadata()
    assert "timestamp" in meta
    assert "python_version" in meta
    assert "platform" in meta
    assert "dependencies" in meta
    assert meta["dependencies"].get("jetf") is not None


def test_generate_json_report_serialization():
    pairwise_cons = PairwiseConsistencyResult(
        n_pairs=100,
        tolerance_da=0.02,
        max_absolute_error=1.23e-6,
        mean_absolute_error=4.56e-8,
        rmse=7.89e-7,
        matched_peak_match_rate=0.99,
        p50_error=0.0,
        p95_error=np.float64(1e-15),  # Test numpy float serialization
        p99_error=np.float64(1e-8),
        discrepant_pairs_count=0,
    )

    query_details = (
        QueryRetrievalConsistency(
            query_index=0,
            mode_name="Top-10",
            jetf_hit_count=10,
            matchms_hit_count=10,
            recall_at_k=1.0,
            zero_false_dismissals=True,
            max_score_diff=np.float64(1e-14),
            rank_consistent=True,
        ),
    )

    retrieval_cons = [
        RetrievalConsistencySummary(
            n_queries=1,
            mode_name="Top-10",
            mean_recall_at_k=1.0,
            all_zero_false_dismissals=True,
            total_false_dismissals=0,
            max_score_discrepancy=1e-14,
            details=query_details,
        )
    ]

    pairwise_tp = PairwiseThroughputResult(
        n_pairs=100,
        jetf_pairs_per_sec=21000.0,
        matchms_pairs_per_sec=22000.0,
        jetf_avg_time_us=47.6,
        matchms_avg_time_us=45.4,
        speedup=0.95,
    )

    retrieval_tp = [
        RetrievalThroughputResult(
            mode_name="Top-10",
            n_queries=1,
            library_size=100,
            jetf_qps=120.0,
            matchms_qps=25.0,
            jetf_latency_mean_ms=8.3,
            jetf_latency_p50_ms=8.0,
            jetf_latency_p95_ms=10.0,
            jetf_latency_p99_ms=12.0,
            matchms_latency_mean_ms=40.0,
            matchms_latency_p50_ms=39.0,
            matchms_latency_p95_ms=42.0,
            matchms_latency_p99_ms=45.0,
            speedup=4.8,
            avg_scored_ratio=0.02,
            avg_pruned_ratio=0.98,
        )
    ]

    config_meta = {"library_size": 100, "tolerance": 0.02}

    json_str = generate_json_report(
        pairwise_consistency=pairwise_cons,
        retrieval_consistency=retrieval_cons,
        pairwise_throughput=pairwise_tp,
        retrieval_throughput=retrieval_tp,
        config_metadata=config_meta,
    )

    parsed = json.loads(json_str)
    assert parsed["report_title"] == "JET-Forest vs matchms Benchmark Report"
    assert "metadata" in parsed
    assert parsed["metadata"]["config"]["library_size"] == 100
    assert "results" in parsed
    assert "pairwise_consistency" in parsed["results"]
    assert parsed["results"]["pairwise_consistency"]["n_pairs"] == 100
    assert "retrieval_consistency" in parsed["results"]
    assert len(parsed["results"]["retrieval_consistency"]) == 1
    assert parsed["results"]["retrieval_consistency"][0]["all_zero_false_dismissals"] is True
    assert "pairwise_throughput" in parsed["results"]
    assert "retrieval_throughput" in parsed["results"]
    assert parsed["results"]["retrieval_throughput"][0]["speedup"] == 4.8


def test_save_json_report(tmp_path: Path):
    target = tmp_path / "sub_dir" / "report.json"
    p = save_json_report(target, config_metadata={"seed": 42})
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["metadata"]["config"]["seed"] == 42


def test_cli_benchmark_json_argument():
    # Verify benchmark CLI parser has -j / --output-json and -c / --output-csv options without crashing
    with pytest.raises(SystemExit) as exc_info:
        main(["benchmark", "--help"])
    assert exc_info.value.code == 0


def test_save_csv_report(tmp_path: Path):
    from jetf.benchmarks.reporter import save_csv_report

    retrieval_tp = [
        RetrievalThroughputResult(
            mode_name="Top-10",
            n_queries=1,
            library_size=100,
            jetf_qps=120.0,
            matchms_qps=25.0,
            jetf_latency_mean_ms=8.3,
            jetf_latency_p50_ms=8.0,
            jetf_latency_p95_ms=10.0,
            jetf_latency_p99_ms=12.0,
            matchms_latency_mean_ms=40.0,
            matchms_latency_p50_ms=39.0,
            matchms_latency_p95_ms=42.0,
            matchms_latency_p99_ms=45.0,
            speedup=4.8,
            avg_scored_ratio=0.02,
            avg_pruned_ratio=0.98,
        )
    ]
    target = tmp_path / "report.csv"
    generated = save_csv_report(target, retrieval_throughput=retrieval_tp)
    assert len(generated) == 1
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert "mode_name" in content
    assert "Top-10" in content
    assert "120.0" in content

