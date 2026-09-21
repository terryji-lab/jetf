"""基准评测报告格式化与输出模块（控制台表格与 Markdown）。"""

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
    jetf_header = "JETF 时延 (Mean±Std [Med] ms)" if is_small_sample else "JETF 时延 (P50/P99 ms)"
    mms_header = "matchms 时延 (Mean±Std [Med] ms)" if is_small_sample else "matchms 时延 (P50/P99 ms)"
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
    return tabulate(rows, headers=headers, tablefmt=fmt)


def generate_full_markdown_report(
    pairwise_consistency: PairwiseConsistencyResult | None = None,
    retrieval_consistency: Sequence[RetrievalConsistencySummary] | None = None,
    pairwise_throughput: PairwiseThroughputResult | None = None,
    retrieval_throughput: Sequence[RetrievalThroughputResult] | None = None,
) -> str:
    """生成完整精美的 Markdown 评测报告。"""
    sections: list[str] = [
        "# JET-Forest 与 matchms 性能与一致性全景对比报告",
        "",
        "> 本报告自动生成自 `jetf benchmark` 套件，在确定性环境与真实质谱数据集上系统评测了 JET-Forest 相对 matchms 的评分一致性与吞吐量性能。",
        "",
    ]

    # 一致性部分
    if pairwise_consistency or retrieval_consistency:
        sections.extend([
            "## 一、结果一致性对比 (Result Consistency)",
            "",
            "### 1.1 单对谱评分等价性 (Pairwise Scoring Equivalence)",
            "比较 `jetf.scoring.score_greedy_cosine` 与 `matchms.similarity.CosineGreedy` 在相同容差与配置下的微观打分差异：",
            "",
        ])
        if pairwise_consistency:
            sections.append(format_pairwise_consistency_table(pairwise_consistency, fmt="github"))
            sections.append("")

        if retrieval_consistency:
            sections.extend([
                "### 1.2 全库检索零漏检与召回率 (Retrieval Recall & Zero False Dismissals)",
                "在不同检索场景下，以 matchms 全量穷举打分为金标准（Ground Truth），检验 JET-Forest 索引剪枝检索的召回率与零漏检特性：",
                "",
                format_retrieval_consistency_table(retrieval_consistency, fmt="github"),
                "",
            ])

    # 吞吐量部分
    if pairwise_throughput or retrieval_throughput:
        sections.extend([
            "## 二、吞吐量与时延对比 (Throughput & Latency)",
            "",
            "### 2.1 逐对谱计算内核微基准 (Pairwise Kernel Microbenchmark)",
            "",
        ])
        if pairwise_throughput:
            sections.append(format_pairwise_throughput_table(pairwise_throughput, fmt="github"))
            sections.append("")

        if retrieval_throughput:
            sections.extend([
                "### 2.2 端到端 1-to-N 库检索性能 (1-to-N Library Retrieval Macrobenchmark)",
                "评测真实查询谱在参考库中的检索吞吐量（QPS）、时延分布及包络森林剪枝带来的端到端加速比：",
                "",
                format_retrieval_throughput_table(retrieval_throughput, fmt="github"),
                "",
            ])

    # 评测结论（根据实际数据动态生成，避免硬编码正面结论与数据脱节）
    conclusions: list[str] = ["## 三、评测结论与工程洞见", ""]

    # 结论 1: 数值等价性（根据 MAE 和显著差异对数动态生成）
    if pairwise_consistency is not None:
        mae = pairwise_consistency.mean_absolute_error
        max_ae = pairwise_consistency.max_absolute_error
        n_disc = pairwise_consistency.discrepant_pairs_count
        if n_disc == 0 and mae < 1e-10:
            conclusions.append(
                "1. **数值等价性**：JET-Forest 与 matchms 在全部抽样谱对中表现出严格的数值等价性"
                f"（MAE = {mae:.2e}，Max AE = {max_ae:.2e}，零显著差异对），"
                "误差处于双精度浮点噪声范围内。"
            )
        elif n_disc > 0:
            conclusions.append(
                f"1. **数值一致性**：JET-Forest 与 matchms 平均绝对误差 MAE = {mae:.2e}，"
                f"其中 {n_disc} 对谱（{n_disc/pairwise_consistency.n_pairs*100:.1f}%）出现显著差异"
                f"（Max AE = {max_ae:.2e}），可能源于并列峰贪心匹配的 tie-breaking 顺序差异。"
                f"其余谱对误差处于浮点噪声范围内（P99 = {pairwise_consistency.p99_error:.2e}）。"
            )
        else:
            conclusions.append(
                f"1. **数值一致性**：MAE = {mae:.2e}，Max AE = {max_ae:.2e}，"
                f"P99 = {pairwise_consistency.p99_error:.2e}，零显著差异对。"
            )

    # 结论 2: 零漏检（根据实际召回率数据生成）
    if retrieval_consistency:
        all_pass = all(s.all_zero_false_dismissals for s in retrieval_consistency)
        total_miss = sum(s.total_false_dismissals for s in retrieval_consistency)
        total_queries = sum(s.n_queries for s in retrieval_consistency)
        if all_pass and total_miss == 0:
            conclusions.append(
                f"2. **零漏检验证**：在 {total_queries} 组查询的全库开放检索中，"
                "Recall@K 均为 100%，未观察到漏检（Zero False Dismissals）。"
                "注意：此结论基于有限抽样，不等同于数学证明。"
            )
        else:
            conclusions.append(
                f"2. **漏检警告**：在 {total_queries} 组查询中观察到 {total_miss} 次漏检，"
                "需进一步排查包络上界安全性。"
            )

    # 结论 3: 加速比（根据实际 speedup 数据生成）
    if retrieval_throughput:
        speedups = [r.speedup for r in retrieval_throughput if r.speedup > 0 and r.matchms_qps > 0]
        if speedups:
            min_sp = min(speedups)
            max_sp = max(speedups)
            conclusions.append(
                f"3. **端到端加速比**：在 {len(speedups)} 个检索场景中，"
                f"JET-Forest 相对 matchms 的加速比范围为 {min_sp:.1f}x ~ {max_sp:.1f}x。"
                "加速来源为包络剪枝避免了绝大多数不相关谱的精评计算。"
                "注意：matchms 侧计时包含 is_eligible 元数据过滤与全量候选排序的 Python 开销，"
                "与 JETF 单次 search_forest 调用的口径不完全对称。"
            )
        else:
            conclusions.append("3. **加速比**：matchms 基线被跳过或数据不可用，无法计算加速比。")

    sections.extend(conclusions)

    return "\n".join(sections)


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
        cons_map: dict[str, RetrievalConsistencySummary] = {}
        if retrieval_consistency:
            for c in retrieval_consistency:
                # 支持精准匹配或按检索模式模式名关键词匹配
                cons_map[c.mode_name] = c
                tokens = c.mode_name.split()
                if len(tokens) > 1:
                    cons_map[tokens[1]] = c

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
            tokens = tp.mode_name.split()
            key = tokens[1] if len(tokens) > 1 else tp.mode_name
            c = cons_map.get(tp.mode_name) or cons_map.get(key)

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
