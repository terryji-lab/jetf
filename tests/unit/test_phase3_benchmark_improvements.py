"""阶段 3 基准测试与评测健壮性增强回归测试套件。

覆盖验证点：
1. [Issue 2 & Phase 1 Flaw A] Robustify verify_correctness in gpu_throughput.py:
   - 对称检查额外 GPU 命中 (extra = g_key_set - c_key_set)
   - 容差边距门槛判定 (legitimate margin vs unexpected false positives)
   - CorrectnessResult 新增字段 fp32_tolerance_discrepant, extra_gpu_hits_count, unexpected_false_positives, all_zero_false_positives
   - FP32 vs FP64 对拍容差校准 (1e-6 ~ score_margin 计入正常容差波动，超出 score_margin 计入异常 query)
   - format_correctness_table 包含多余命中、零误检核验及 FP32 容差波动列
2. [Issue 6 & Issue 9] Metric Alignment & Latency Reporting Transparency:
   - format_benchmark_table 明确 "有效均值时延 (摊薄)", "任务时延 P50", "任务时延 P95"
   - 表格注脚说明 CPU MT 与 GPU 批处理耗时语义
   - format_retrieval_throughput_table 包含对应注脚
   - throughput.py 中 RetrievalThroughputResult 语义口径说明对齐
3. [Issue 7] Multi-Threaded matchms Support in unified_runner.py:
   - concurrency > 1 时使用 ThreadPoolExecutor 并发执行 matchms
   - concurrency <= 1 时保持单线程串行循环
   - 标签为 "matchms (NT)" 且保持 "matchms" 键向后兼容
4. [Issue 8] Sample Size and Statistical Stability in gpu_throughput.py:
   - 默认 --n-queries 改为 256
   - n_queries < 100 时输出离群点敏感性提示
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from jetf.benchmarks.dataset import BenchmarkDataset
from jetf.benchmarks.gpu_throughput import (
    BenchmarkRunResult,
    CorrectnessResult,
    format_benchmark_table,
    format_correctness_table,
    verify_correctness,
)
from jetf.benchmarks.reporter import format_retrieval_throughput_table
from jetf.benchmarks.throughput import RetrievalThroughputResult
from jetf.benchmarks.unified_runner import MultiEngineBenchmarkRunner
from jetf.builder import build_forest_index
from jetf.mgf import ParsedLibrary
from jetf.preprocessing import preprocess_library
from jetf.query import SearchMode
from jetf.results import SearchHit, SearchOutcome, SearchStats
from jetf.structure import DEFAULT_FOREST_SPEC
from jetf.types import IonMode, SourceRef, SpectrumMeta


def _make_dummy_outcome(hits: list[tuple[str, int, float]], mode: SearchMode = SearchMode.THRESHOLD) -> SearchOutcome:
    """构造用于测试对拍核验的 SearchOutcome。"""
    search_hits = tuple(
        SearchHit(external_id=h[0], spectrum_index=h[1], score=h[2], n_matched=5)
        for h in hits
    )
    stats = SearchStats(nodes_visited=10, n_scored=len(hits), pruned_by_layer={})
    return SearchOutcome(mode=mode, hits=search_hits, complete=True, stats=stats, versions={})


@pytest.fixture
def mini_bench_dataset() -> BenchmarkDataset:
    """构造微型合成质谱库数据集 (15 条谱)。"""
    rng = np.random.default_rng(2026)
    n_spectra = 15
    spectra = []
    masses = []
    intensities = []
    offsets = [0]

    for i in range(n_spectra):
        n_p = int(rng.integers(10, 25))
        m = np.sort(rng.uniform(60.0, 500.0, size=n_p))
        raw_it = rng.uniform(5.0, 100.0, size=n_p)
        masses.extend(m)
        intensities.extend(raw_it)
        offsets.append(offsets[-1] + n_p)
        pmz = float(np.max(m) + rng.uniform(15.0, 35.0))
        meta = SpectrumMeta(
            external_id=f"TEST_SPEC_{i:04d}",
            precursor_mz=pmz,
            charge=1,
            ion_mode=IonMode.POSITIVE,
            source=SourceRef("test.mgf", i),
        )
        spectra.append(meta)

    mass_arr = np.asarray(masses, dtype=np.float64)
    int_arr = np.asarray(intensities, dtype=np.float64)
    pid_arr = np.arange(len(mass_arr), dtype=np.int64)
    off_arr = np.asarray(offsets, dtype=np.int64)

    parsed = ParsedLibrary(
        source_path="test.mgf",
        spectra=tuple(spectra),
        mass=mass_arr,
        intensity=int_arr,
        peak_id=pid_arr,
        spectrum_offsets=off_arr,
    )
    lib = preprocess_library(parsed)
    forest = build_forest_index(lib, DEFAULT_FOREST_SPEC)
    return BenchmarkDataset(parsed=parsed, library=lib, forest=forest)


# =========================================================================
# 1. [Issue 2] Robustify verify_correctness
# =========================================================================

def test_correctness_result_dataclass_fields():
    """验证 CorrectnessResult 包含新增字段且具备安全默认值。"""
    c = CorrectnessResult(
        mode="threshold",
        n_queries_evaluated=10,
        all_zero_false_dismissals=True,
        total_false_dismissals=0,
        mean_recall_at_k=1.0,
        max_score_absolute_error=0.0,
        mean_score_absolute_error=0.0,
        hit_count_match_rate=1.0,
        discrepant_queries_count=0,
    )
    assert c.fp32_tolerance_discrepant == 0
    assert c.extra_gpu_hits_count == 0
    assert c.unexpected_false_positives == 0
    assert c.all_zero_false_positives is True


def test_verify_correctness_perfect_match():
    """验证 CPU 与 GPU 检索结果完全一致时的核验状态。"""
    cpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85), ("SPEC_B", 1, 0.75)])]
    gpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85), ("SPEC_B", 1, 0.75)])]

    res = verify_correctness(cpu_outs, gpu_outs, mode_name="threshold", threshold=0.7)
    assert res.all_zero_false_dismissals is True
    assert res.total_false_dismissals == 0
    assert res.extra_gpu_hits_count == 0
    assert res.unexpected_false_positives == 0
    assert res.all_zero_false_positives is True
    assert res.mean_recall_at_k == 1.0
    assert res.max_score_absolute_error == 0.0
    assert res.fp32_tolerance_discrepant == 0
    assert res.discrepant_queries_count == 0


def test_verify_correctness_legitimate_margin_extra_hit():
    """验证 GPU 额外命中位于浮点容差边距内 (threshold - score_margin) 时属于合法接收，不算假阳性。"""
    # 门槛 0.7，容差 1e-4，故 >= 0.69990 的命中属于合法边界接纳
    cpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85)])]
    gpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85), ("SPEC_B", 1, 0.69995)])]

    res = verify_correctness(
        cpu_outs,
        gpu_outs,
        mode_name="threshold",
        threshold=0.7,
        score_margin=1e-4,
    )
    assert res.all_zero_false_dismissals is True
    assert res.total_false_dismissals == 0
    assert res.extra_gpu_hits_count == 1
    assert res.unexpected_false_positives == 0  # 分数 0.69995 >= 0.69990，不作为假阳性
    assert res.all_zero_false_positives is True
    assert res.fp32_tolerance_discrepant == 0
    assert res.discrepant_queries_count == 0


def test_verify_correctness_unexpected_false_positive():
    """验证 GPU 额外命中分数显著低于门槛 (score < threshold - margin) 时被正确判定为假阳性。"""
    # 门槛 0.7，容差 1e-4，出现一个 0.50 的严重偏离命中
    cpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85)])]
    gpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85), ("SPEC_B", 1, 0.50)])]

    res = verify_correctness(
        cpu_outs,
        gpu_outs,
        mode_name="threshold",
        threshold=0.7,
        score_margin=1e-4,
    )
    assert res.all_zero_false_dismissals is True
    assert res.total_false_dismissals == 0
    assert res.extra_gpu_hits_count == 1
    assert res.unexpected_false_positives == 1
    assert res.all_zero_false_positives is False
    assert res.discrepant_queries_count == 1
    assert res.fp32_tolerance_discrepant == 0


def test_verify_correctness_false_dismissal():
    """验证 GPU 漏检 CPU 命中时被正确判定为假阴性漏检。"""
    cpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85), ("SPEC_B", 1, 0.75)])]
    gpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85)])]

    res = verify_correctness(cpu_outs, gpu_outs, mode_name="threshold", threshold=0.7)
    assert res.all_zero_false_dismissals is False
    assert res.total_false_dismissals == 1
    assert res.mean_recall_at_k == 0.5
    assert res.all_zero_false_positives is True
    assert res.discrepant_queries_count == 1
    assert res.fp32_tolerance_discrepant == 0


def test_verify_correctness_fp32_tolerance_discrepant():
    """验证分差在 1e-6 ~ score_margin 之间时被正确计入 fp32_tolerance_discrepant 且不标记异常。"""
    # 正常 FP32 累加误差 2e-5 (在 1e-6 到 1e-4 容差之间)
    cpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85000)])]
    gpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.85002)])]

    res = verify_correctness(
        cpu_outs,
        gpu_outs,
        mode_name="threshold",
        threshold=0.7,
        score_margin=1e-4,
    )
    assert res.all_zero_false_dismissals is True
    assert res.total_false_dismissals == 0
    assert res.extra_gpu_hits_count == 0
    assert res.unexpected_false_positives == 0
    assert res.all_zero_false_positives is True
    assert res.fp32_tolerance_discrepant == 1
    assert res.discrepant_queries_count == 0
    assert pytest.approx(res.max_score_absolute_error, rel=1e-4) == 2e-5


def test_verify_correctness_exceeding_score_margin_discrepant():
    """验证分差超出 score_margin 时被正确计入 discrepant_queries_count。"""
    # 超出 score_margin 容差的分差 3e-4 (> 1e-4)
    cpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.8500)])]
    gpu_outs = [_make_dummy_outcome([("SPEC_A", 0, 0.8503)])]

    res = verify_correctness(
        cpu_outs,
        gpu_outs,
        mode_name="threshold",
        threshold=0.7,
        score_margin=1e-4,
    )
    assert res.all_zero_false_dismissals is True
    assert res.total_false_dismissals == 0
    assert res.extra_gpu_hits_count == 0
    assert res.unexpected_false_positives == 0
    assert res.all_zero_false_positives is True
    assert res.fp32_tolerance_discrepant == 0
    assert res.discrepant_queries_count == 1
    assert pytest.approx(res.max_score_absolute_error, rel=1e-4) == 3e-4


def test_verify_correctness_multi_query_tolerance_and_exceeding():
    """验证多 query 场景下完美匹配、FP32 正常容差波动与超标分差的综合统计。"""
    cpu_outs = [
        _make_dummy_outcome([("SPEC_A", 0, 0.850000)]),              # Q0: 完全一致
        _make_dummy_outcome([("SPEC_B", 1, 0.800000)]),              # Q1: 分差 <= 1e-6 (5e-7)
        _make_dummy_outcome([("SPEC_C", 2, 0.750000)]),              # Q2: 正常 FP32 波动 (2e-5)
        _make_dummy_outcome([("SPEC_D", 3, 0.700000)]),              # Q3: 超标分差 (2e-4)
    ]
    gpu_outs = [
        _make_dummy_outcome([("SPEC_A", 0, 0.850000)]),
        _make_dummy_outcome([("SPEC_B", 1, 0.8000005)]),
        _make_dummy_outcome([("SPEC_C", 2, 0.750020)]),
        _make_dummy_outcome([("SPEC_D", 3, 0.700200)]),
    ]

    res = verify_correctness(
        cpu_outs,
        gpu_outs,
        mode_name="threshold",
        threshold=0.7,
        score_margin=1e-4,
    )
    assert res.n_queries_evaluated == 4
    assert res.fp32_tolerance_discrepant == 1
    assert res.discrepant_queries_count == 1
    assert res.all_zero_false_dismissals is True
    assert res.all_zero_false_positives is True


def test_format_correctness_table():
    """验证 format_correctness_table 正确展示零误检、多余命中列以及 FP32 容差波动数。"""
    c_pass = CorrectnessResult(
        mode="threshold",
        n_queries_evaluated=50,
        all_zero_false_dismissals=True,
        total_false_dismissals=0,
        mean_recall_at_k=1.0,
        max_score_absolute_error=1.2e-7,
        mean_score_absolute_error=3.4e-8,
        hit_count_match_rate=1.0,
        discrepant_queries_count=0,
        fp32_tolerance_discrepant=3,
        extra_gpu_hits_count=0,
        unexpected_false_positives=0,
        all_zero_false_positives=True,
    )
    table_str = format_correctness_table([c_pass])
    assert "零漏检状态 (Zero Miss)" in table_str
    assert "零误检状态 (Zero FP)" in table_str
    assert "多余命中/误检数 (Extra/FP)" in table_str
    assert "FP32容差波动数 (Tol Diff)" in table_str
    assert "PASS (Zero FP)" in table_str
    assert "0 / 0" in table_str
    assert "3" in table_str
    assert "FP32容差波动数" in table_str


# =========================================================================
# 2. [Issue 6 & Issue 9] Metric Alignment & Latency Reporting Transparency
# =========================================================================

def test_format_benchmark_table_headers_and_footnote():
    """验证 format_benchmark_table 明确使用'有效均值时延 (摊薄)'及相关分位数与注脚。"""
    run_res = BenchmarkRunResult(
        backend="CPU (4 Threads)",
        mode="threshold",
        batch_size=4,
        n_queries=128,
        library_size=1000,
        wall_time_s=0.5,
        qps=256.0,
        latency_mean_ms=3.91,
        latency_p50_ms=12.5,
        latency_p95_ms=15.0,
        latency_p99_ms=16.0,
        latency_min_ms=10.0,
        latency_max_ms=18.0,
        latency_std_ms=2.1,
        speedup_vs_1t=3.5,
        speedup_vs_mt=1.0,
        avg_roots_pruned=100.0,
        avg_leaves_pruned=500.0,
        avg_uind_pruned=20.0,
        avg_scored_count=50.0,
        avg_pruned_ratio=0.95,
    )
    table_str = format_benchmark_table([run_res], title="Threshold Search")
    assert "有效均值时延 (摊薄)" in table_str
    assert "任务时延 P50" in table_str
    assert "任务时延 P95" in table_str
    assert "CPU MT 的 P50/P95 包含多线程争用下的任务单次执行耗时" in table_str
    assert "有效服务时延 (Wall / N)" in table_str
    # 样本量 128 >= 100，不出现离群点警告
    assert "样本量 N < 100 时 P95/P99 分位数易受单离群点波动影响" not in table_str


def test_format_benchmark_table_small_sample_warning():
    """验证 N < 100 时 format_benchmark_table 自动追加离群点波动提示。"""
    run_res = BenchmarkRunResult(
        backend="CPU (1 Thread)",
        mode="threshold",
        batch_size=1,
        n_queries=50,
        library_size=1000,
        wall_time_s=0.2,
        qps=250.0,
        latency_mean_ms=4.0,
        latency_p50_ms=4.0,
        latency_p95_ms=5.0,
        latency_p99_ms=6.0,
        latency_min_ms=3.0,
        latency_max_ms=7.0,
        latency_std_ms=0.5,
        speedup_vs_1t=1.0,
        speedup_vs_mt=1.0,
        avg_roots_pruned=0.0,
        avg_leaves_pruned=0.0,
        avg_uind_pruned=0.0,
        avg_scored_count=0.0,
        avg_pruned_ratio=0.0,
    )
    table_str = format_benchmark_table([run_res], title="Small Sample")
    assert "样本量 N < 100 时 P95/P99 分位数易受单离群点波动影响" in table_str


def test_format_retrieval_throughput_table_footnote():
    """验证 reporter.py 中 format_retrieval_throughput_table 亦包含透明化注脚。"""
    tp = RetrievalThroughputResult(
        mode_name="Top-10",
        n_queries=120,
        library_size=1000,
        jetf_qps=500.0,
        matchms_qps=50.0,
        jetf_latency_mean_ms=2.0,
        jetf_latency_p50_ms=5.0,
        jetf_latency_p95_ms=8.0,
        jetf_latency_p99_ms=10.0,
        matchms_latency_mean_ms=20.0,
        matchms_latency_p50_ms=20.0,
        matchms_latency_p95_ms=22.0,
        matchms_latency_p99_ms=25.0,
        speedup=10.0,
        avg_scored_ratio=0.05,
        avg_pruned_ratio=0.95,
    )
    table_str = format_retrieval_throughput_table([tp])
    assert "CPU MT 的 P50/P95/P99 包含多线程争用下的任务单次执行耗时" in table_str


# =========================================================================
# 3. [Issue 7] Multi-Threaded matchms Support in unified_runner.py
# =========================================================================

def test_unified_runner_matchms_multithreaded_concurrency(mini_bench_dataset: BenchmarkDataset):
    """验证 unified_runner.py 中 matchms 在 concurrency > 1 时并行化执行并保持向后兼容。"""
    runner = MultiEngineBenchmarkRunner(
        mini_bench_dataset,
        engines=["matchms"],
        clean_matchms=False,
    )
    queries = runner.sample_queries(n_queries=6, seed=2026)

    # 1. 验证多线程并发执行 (concurrency=2)
    res_mt = runner.run_benchmark(queries, mode="open", top_k=5, concurrency=2)
    assert "matchms" in res_mt
    assert "matchms (2T)" in res_mt
    stat_mt = res_mt["matchms"]
    assert stat_mt.n_queries == 6
    assert stat_mt.batch_size == 2
    assert stat_mt.engine == "matchms (2T)"
    assert len(stat_mt.raw_latencies_ms) == 6
    assert stat_mt.mean_ms > 0.0

    # 2. 验证单线程串行执行 (concurrency=1)
    res_1t = runner.run_benchmark(queries, mode="open", top_k=5, concurrency=1)
    assert "matchms" in res_1t
    assert "matchms (1T)" not in res_1t
    stat_1t = res_1t["matchms"]
    assert stat_1t.batch_size == 1
    assert stat_1t.engine == "matchms"


# =========================================================================
# 4. [Issue 8] Sample Size and Statistical Stability
# =========================================================================

def test_gpu_throughput_argparse_default_n_queries():
    """验证 gpu_throughput.py 的 --n-queries 默认值已升级为 256。"""
    from jetf.benchmarks import gpu_throughput

    with patch.object(gpu_throughput, "is_cuda_available", return_value=False):
        # 构造 parser 测试默认参数值
        parser = argparse.ArgumentParser()
        # 模拟 main 中设置的 argument
        parser.add_argument("--n-queries", type=int, default=256)
        args = parser.parse_args([])
        assert args.n_queries == 256

    # 验证 gpu_throughput.py 文件源码中默认参数确为 256
    import inspect
    src = inspect.getsource(gpu_throughput.main)
    assert 'default=256' in src
