"""基准评测报告格式化与输出模块（控制台表格、JSON 与 CSV）。"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import datetime
import json
from pathlib import Path
import platform
from typing import Any, Sequence

import numpy as np
from tabulate import tabulate

from jetf.benchmarks.consistency import PairwiseConsistencyResult, RetrievalConsistencySummary
from jetf.benchmarks.throughput import PairwiseThroughputResult, RetrievalThroughputResult


def format_pairwise_consistency_table(result: PairwiseConsistencyResult, fmt: str = "simple") -> str:
    """格式化逐对谱评分一致性指标表格。"""
    headers = ["评估指标", "数值", "说明"]
    rows = [
        ["测试谱对数量 (Pairs)", f"{result.n_pairs:,}", "抽样/测试谱对总数"],
        ["匹配容差 (Tolerance)", f"{result.tolerance_da} Da", "片断匹配容差窗口"],
        ["最大绝对误差 (Max AE)", f"{result.max_absolute_error:.2e}", "JETF 与 matchms 打分绝对差峰值"],
        ["平均绝对误差 (MAE)", f"{result.mean_absolute_error:.2e}", "全样本绝对误差均值"],
        ["均方根误差 (RMSE)", f"{result.rmse:.2e}", "标准方均根误差"],
        ["50% 分位数误差 (P50)", f"{result.p50_error:.2e}", "中位数误差"],
        ["95% 分位数误差 (P95)", f"{result.p95_error:.2e}", "95% 谱对误差上限"],
        ["99% 分位数误差 (P99)", f"{result.p99_error:.2e}", "99% 谱对误差上限"],
        ["匹配峰数完全吻合率", f"{result.matched_peak_match_rate * 100:.2f}%", "n_matched 与 matches 完全相等比例"],
        ["显著差异对数 (|diff| > 1e-6)", f"{result.discrepant_pairs_count}", "异常分差样本数"],
    ]
    return tabulate(rows, headers=headers, tablefmt=fmt)


def format_retrieval_consistency_table(
    summaries: Sequence[RetrievalConsistencySummary], fmt: str = "simple"
) -> str:
    """格式化全库检索一致性汇总表格。"""
    headers = ["检索模式", "查询数", "平均召回率 (Recall@K)", "零漏检检验", "漏检总数", "最大分差"]
    rows = []
    for s in summaries:
        rows.append(
            [
                s.mode_name,
                s.n_queries,
                f"{s.mean_recall_at_k * 100:.2f}%",
                "PASS (Zero Miss)" if s.all_zero_false_dismissals else "FAIL",
                s.total_false_dismissals,
                f"{s.max_score_discrepancy:.2e}",
            ]
        )
    return tabulate(rows, headers=headers, tablefmt=fmt)


def format_pairwise_throughput_table(result: PairwiseThroughputResult, fmt: str = "simple") -> str:
    """格式化逐对谱评分算子性能表格。"""
    headers = ["算子实现", "每秒计算谱对数 (Pairs/s)", "单次平均时延 (μs)", "加速比"]
    rows = [
        ["matchms (Numba JIT)", f"{result.matchms_pairs_per_sec:,.0f}", f"{result.matchms_avg_time_us:.2f} μs", "1.00x"],
        ["JETF (NumPy 向量化)", f"{result.jetf_pairs_per_sec:,.0f}", f"{result.jetf_avg_time_us:.2f} μs", f"{result.speedup:.2f}x"],
    ]
    return tabulate(rows, headers=headers, tablefmt=fmt)


def format_retrieval_throughput_table(
    results: Sequence[RetrievalThroughputResult], fmt: str = "simple"
) -> str:
    """格式化 1-to-N 库检索吞吐量与加速比表格。"""
    is_small_sample = any(r.n_queries < 50 for r in results)
    jetf_header = "JETF 时延 (有效均值±Std [Med] ms)" if is_small_sample else "JETF 任务时延 (P50/P99 ms)"
    mms_header = "matchms 时延 (有效均值±Std [Med] ms)" if is_small_sample else "matchms 任务时延 (P50/P99 ms)"
    headers = [
        "检索场景",
        "库容量 (N)",
        "JETF QPS",
        "matchms QPS",
        jetf_header,
        mms_header,
        "加速比 (Speedup)",
        "包络剪枝率",
    ]
    rows = []
    for r in results:
        std_j = getattr(r, "jetf_latency_std_ms", 0.0)
        std_m = getattr(r, "matchms_latency_std_ms", 0.0)
        if is_small_sample:
            jetf_lat = f"{r.jetf_latency_mean_ms:.2f}±{std_j:.2f} [{r.jetf_latency_p50_ms:.2f}]"
        else:
            jetf_lat = f"{r.jetf_latency_p50_ms:.2f} / {r.jetf_latency_p99_ms:.2f}"

        if r.matchms_qps > 0:
            if is_small_sample:
                mms_lat = f"{r.matchms_latency_mean_ms:.2f}±{std_m:.2f} [{r.matchms_latency_p50_ms:.2f}]"
            else:
                mms_lat = f"{r.matchms_latency_p50_ms:.2f} / {r.matchms_latency_p99_ms:.2f}"
            mms_qps_str = f"{r.matchms_qps:,.1f}"
            speedup_str = f"{r.speedup:,.1f}x"
        else:
            mms_lat = "N/A (skipped)"
            mms_qps_str = "N/A"
            speedup_str = "N/A"
        rows.append(
            [
                r.mode_name,
                f"{r.library_size:,}",
                f"{r.jetf_qps:,.1f}",
                mms_qps_str,
                jetf_lat,
                mms_lat,
                speedup_str,
                f"{r.avg_pruned_ratio * 100:.2f}%",
            ]
        )
    table = tabulate(rows, headers=headers, tablefmt=fmt)
    footnote = (
        "* 注: CPU MT 的 P50/P95/P99 包含多线程争用下的任务单次执行耗时，而均值为系统级有效服务时延 (Wall / N)；"
        "GPU 采用批处理执行，任务耗时按批大小均匀分摊。"
    )
    if any(r.n_queries < 100 for r in results):
        footnote += "\n* (注: 样本量 N < 100 时 P95/P99 分位数易受单离群点波动影响)"
    return f"{table}\n{footnote}"

def get_system_metadata() -> dict[str, Any]:
    """采集当前运行环境元数据（系统、Python与核心依赖版本）。"""
    deps: dict[str, str | None] = {}
    for pkg in ("jetf", "matchms", "numpy", "numba", "scipy", "pandas", "tabulate"):
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = None

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "dependencies": deps,
    }


def _to_json_compatible(obj: Any) -> Any:
    """递归将 dataclass、NumPy 数据类型转换为标准 Python JSON 可序列化类型。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_json_compatible(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_benchmark_report_dict(
    pairwise_consistency: PairwiseConsistencyResult | None = None,
    retrieval_consistency: Sequence[RetrievalConsistencySummary] | None = None,
    pairwise_throughput: PairwiseThroughputResult | None = None,
    retrieval_throughput: Sequence[RetrievalThroughputResult] | None = None,
    config_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """将基准评测对象及元数据组装为完整的结构化字典。"""
    metadata = get_system_metadata()
    if config_metadata:
        metadata["config"] = _to_json_compatible(config_metadata)

    results: dict[str, Any] = {}
    if pairwise_consistency is not None:
        results["pairwise_consistency"] = pairwise_consistency
    if retrieval_consistency is not None:
        results["retrieval_consistency"] = list(retrieval_consistency)
    if pairwise_throughput is not None:
        results["pairwise_throughput"] = pairwise_throughput
    if retrieval_throughput is not None:
        results["retrieval_throughput"] = list(retrieval_throughput)

    raw_report = {
        "report_title": "JET-Forest vs matchms Benchmark Report",
        "metadata": metadata,
        "results": results,
    }
    return _to_json_compatible(raw_report)


def generate_json_report(
    pairwise_consistency: PairwiseConsistencyResult | None = None,
    retrieval_consistency: Sequence[RetrievalConsistencySummary] | None = None,
    pairwise_throughput: PairwiseThroughputResult | None = None,
    retrieval_throughput: Sequence[RetrievalThroughputResult] | None = None,
    config_metadata: dict[str, Any] | None = None,
    indent: int = 2,
) -> str:
    """生成结构化 JSON 格式的基准评测报告字符串。"""
    data = build_benchmark_report_dict(
        pairwise_consistency=pairwise_consistency,
        retrieval_consistency=retrieval_consistency,
        pairwise_throughput=pairwise_throughput,
        retrieval_throughput=retrieval_throughput,
        config_metadata=config_metadata,
    )
    return json.dumps(data, indent=indent, ensure_ascii=False)


def save_json_report(
    output_path: str | Path,
    pairwise_consistency: PairwiseConsistencyResult | None = None,
    retrieval_consistency: Sequence[RetrievalConsistencySummary] | None = None,
    pairwise_throughput: PairwiseThroughputResult | None = None,
    retrieval_throughput: Sequence[RetrievalThroughputResult] | None = None,
    config_metadata: dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """将评测结果保存为 JSON 文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_json_report(
        pairwise_consistency=pairwise_consistency,
        retrieval_consistency=retrieval_consistency,
        pairwise_throughput=pairwise_throughput,
        retrieval_throughput=retrieval_throughput,
        config_metadata=config_metadata,
        indent=indent,
    )
    path.write_text(content, encoding="utf-8")
    return path


def save_csv_report(
    output_path: str | Path,
    retrieval_throughput: Sequence[RetrievalThroughputResult] | None = None,
    retrieval_consistency: Sequence[RetrievalConsistencySummary] | None = None,
    pairwise_consistency: PairwiseConsistencyResult | None = None,
    pairwise_throughput: PairwiseThroughputResult | None = None,
) -> list[Path]:
    """将基准评测结果导出为平铺、紧凑、机器与分析友好的标准 CSV 文件。"""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. 检索宏基准表 (Retrieval Macrobenchmark: 包含吞吐、时延分布与零漏检召回率)
    if retrieval_throughput:
        def _normalize_mode_key(name: str) -> str:
            # 1. 移除前缀编号 (如 "1. ", "2. ")
            parts = name.split(maxsplit=1)
            if len(parts) > 1 and (parts[0].endswith(".") or parts[0].isdigit()):
                name = parts[1]
            # 2. 移除括号场景说明 (如 "(排除自身)", "(全库无限制)")
            if "(" in name:
                name = name.split("(", 1)[0].strip()
            return name.strip()

        cons_map: dict[str, RetrievalConsistencySummary] = {}
        if retrieval_consistency:
            for c in retrieval_consistency:
                cons_map[c.mode_name] = c
                cons_map[_normalize_mode_key(c.mode_name)] = c

        fieldnames = [
            "mode_name",
            "library_size",
            "n_queries",
            "recall_at_k",
            "zero_false_dismissals",
            "total_false_dismissals",
            "max_score_discrepancy",
            "jetf_qps",
            "matchms_qps",
            "speedup",
            "throughput_speedup",
            "jetf_latency_p50_ms",
            "jetf_latency_mean_ms",
            "jetf_latency_std_ms",
            "jetf_latency_p95_ms",
            "jetf_latency_p99_ms",
            "matchms_latency_p50_ms",
            "matchms_latency_mean_ms",
            "matchms_latency_std_ms",
            "avg_pruned_ratio",
        ]

        rows: list[dict[str, Any]] = []
        for tp in retrieval_throughput:
            norm_key = _normalize_mode_key(tp.mode_name)
            c = cons_map.get(tp.mode_name) or cons_map.get(norm_key)

            row: dict[str, Any] = {
                "mode_name": tp.mode_name,
                "library_size": tp.library_size,
                "n_queries": tp.n_queries,
                "recall_at_k": f"{c.mean_recall_at_k:.4f}" if c else "",
                "zero_false_dismissals": str(c.all_zero_false_dismissals) if c else "",
                "total_false_dismissals": c.total_false_dismissals if c else "",
                "max_score_discrepancy": f"{c.max_score_discrepancy:.2e}" if c else "",
                "jetf_qps": round(tp.jetf_qps, 2),
                "matchms_qps": round(tp.matchms_qps, 2) if tp.matchms_qps > 0 else "N/A",
                "speedup": f"{tp.speedup:.2f}x" if not np.isnan(tp.speedup) else "N/A",
                "throughput_speedup": (
                    f"{tp.throughput_speedup:.2f}x"
                    if getattr(tp, "throughput_speedup", 0.0) > 0
                    and not np.isnan(getattr(tp, "throughput_speedup", 0.0))
                    else "N/A"
                ),
                "jetf_latency_p50_ms": round(tp.jetf_latency_p50_ms, 2),
                "jetf_latency_mean_ms": round(tp.jetf_latency_mean_ms, 2),
                "jetf_latency_std_ms": round(tp.jetf_latency_std_ms, 2),
                "jetf_latency_p95_ms": round(tp.jetf_latency_p95_ms, 2),
                "jetf_latency_p99_ms": round(tp.jetf_latency_p99_ms, 2),
                "matchms_latency_p50_ms": round(tp.matchms_latency_p50_ms, 2) if tp.matchms_qps > 0 else "N/A",
                "matchms_latency_mean_ms": round(tp.matchms_latency_mean_ms, 2) if tp.matchms_qps > 0 else "N/A",
                "matchms_latency_std_ms": round(tp.matchms_latency_std_ms, 2) if tp.matchms_qps > 0 else "N/A",
                "avg_pruned_ratio": f"{tp.avg_pruned_ratio * 100:.2f}%",
            }
            rows.append(row)

        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        generated.append(out_p)

    # 2. 逐对谱打分与算子微基准表 (Pairwise Microbenchmark, 若存在)
    if pairwise_consistency or pairwise_throughput:
        pw_path = out_p.parent / f"{out_p.stem}_pairwise.csv"
        pw_row: dict[str, Any] = {}
        if pairwise_consistency:
            pw_row.update(
                {
                    "n_pairs": pairwise_consistency.n_pairs,
                    "tolerance_da": pairwise_consistency.tolerance_da,
                    "max_absolute_error": f"{pairwise_consistency.max_absolute_error:.2e}",
                    "mean_absolute_error": f"{pairwise_consistency.mean_absolute_error:.2e}",
                    "rmse": f"{pairwise_consistency.rmse:.2e}",
                    "p50_error": f"{pairwise_consistency.p50_error:.2e}",
                    "p95_error": f"{pairwise_consistency.p95_error:.2e}",
                    "p99_error": f"{pairwise_consistency.p99_error:.2e}",
                    "matched_peak_match_rate": f"{pairwise_consistency.matched_peak_match_rate * 100:.2f}%",
                    "discrepant_pairs_count": pairwise_consistency.discrepant_pairs_count,
                }
            )
        if pairwise_throughput:
            pw_row.update(
                {
                    "matchms_pairs_per_sec": round(pairwise_throughput.matchms_pairs_per_sec, 1),
                    "matchms_avg_time_us": round(pairwise_throughput.matchms_avg_time_us, 2),
                    "jetf_pairs_per_sec": round(pairwise_throughput.jetf_pairs_per_sec, 1),
                    "jetf_avg_time_us": round(pairwise_throughput.jetf_avg_time_us, 2),
                    "pairwise_speedup": f"{pairwise_throughput.speedup:.2f}x",
                }
            )
        if pw_row:
            with open(pw_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(pw_row.keys()))
                writer.writeheader()
                writer.writerow(pw_row)
            generated.append(pw_path)

    return generated
