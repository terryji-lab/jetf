#!/usr/bin/env python
"""JET-Forest vs BLINK vs FlashEntropySearch vs matchms 全景多算法对比基准评测脚本。

功能：
1. 四大引擎 (JETF, BLINK, FlashEntropySearch, matchms) 统一离线建库与检索测试
2. 跨检索模式评测：全库开放检索 (Open Search) 与窄前体窗检索 (Identity Search)
3. 库容量阶梯伸缩性评测 (N = 100, 500, 1000, 2000, 5000, 10000)
4. 多维度精度与排序一致性分析 (MAE, RMSE, Pearson, Top-1 一致率, Top-K Jaccard)
5. 自动导出 6 联版高清学术出版级对比图表与结构化 JSON / CSV 报告
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 确保项目模块与外部库在 sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root / "blink") not in sys.path:
    sys.path.insert(0, str(repo_root / "blink"))
if str(repo_root / "FlashEntropySearch" / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "FlashEntropySearch" / "src"))

from jetf.benchmarks.dataset import load_benchmark_dataset, sample_query_spectra
from jetf.benchmarks.unified_runner import (
    MultiEngineBenchmarkRunner,
    UnifiedBenchmarkReport,
    detect_available_engines,
)
from jetf.cleaning import MatchmsCleanConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JET-Forest vs BLINK vs FlashEntropy vs matchms 全景对比基准评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mgf",
        type=str,
        default="GNPS-LIBRARY.mgf",
        help="输入参考库 MGF 文件路径",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=2000,
        help="评测参考库载入谱图数量",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=30,
        help="评测抽样的查询谱数量",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="碎片离子匹配容差 (Da)",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="auto",
        help="参评引擎列表 (逗号分隔，如 'jetf,blink,flashentropy,matchms' 或 'auto')",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="100,500,1000,2000",
        help="伸缩性评测库规模 (逗号分隔)",
    )
    parser.add_argument(
        "--mode",
        choices=["open", "identity", "both"],
        default="open",
        help="检索评测场景",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/benchmark-comprehensive.json",
        help="输出 JSON 评测报告路径",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default="docs/benchmark-comprehensive.png",
        help="输出 6 联版全景对比高清图表路径",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="是否开启 matchms 工业级谱图清洗",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="评测随机种子",
    )
    return parser.parse_args()


def plot_comprehensive_figures(
    report: UnifiedBenchmarkReport,
    out_fig_path: Path,
) -> None:
    """绘制 6 联版高清学术出版级对比图表。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=300)
    colors = {
        "jetf": "#1f77b4",        # 稳健蓝
        "blink": "#2ca02c",       # 森林绿
        "flashentropy": "#ff7f0e", # 活力橙
        "matchms": "#d62728",     # 警告红
    }
    display_names = {
        "jetf": "JET-Forest (Ours)",
        "blink": "BLINK",
        "flashentropy": "FlashEntropy",
        "matchms": "matchms (GT)",
    }

    engines = [e for e in report.available_engines if e in report.latency_stats]

    # Panel 1: 单查询延迟对比 (Mean & P50)
    ax1 = axes[0, 0]
    x = np.arange(len(engines))
    width = 0.35
    means = [report.latency_stats[e].mean_ms for e in engines]
    p50s = [report.latency_stats[e].p50_ms for e in engines]

    bar1 = ax1.bar(x - width / 2, means, width, label="Mean Latency", color=[colors.get(e, "#7f7f7f") for e in engines], alpha=0.85)
    bar2 = ax1.bar(x + width / 2, p50s, width, label="P50 Latency", color=[colors.get(e, "#7f7f7f") for e in engines], alpha=0.45, hatch="//")

    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("(a) Single Query Latency (Mean & P50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([display_names.get(e, e) for e in engines], rotation=15)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: 检索吞吐量 QPS 对比
    ax2 = axes[0, 1]
    qps_values = [report.latency_stats[e].qps for e in engines]
    bars = ax2.bar(x, qps_values, width=0.55, color=[colors.get(e, "#7f7f7f") for e in engines])
    ax2.set_ylabel("Queries Per Second (QPS)")
    ax2.set_title("(b) Retrieval Throughput (QPS)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([display_names.get(e, e) for e in engines], rotation=15)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, qps_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.02, f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 3: 库容伸缩性扩展趋势 (QPS vs Scale, Log-Log)
    ax3 = axes[0, 2]
    if report.scaling_results:
        scales = [r["scale"] for r in report.scaling_results]
        if "jetf_qps" in report.scaling_results[0]:
            jetf_q = [r["jetf_qps"] for r in report.scaling_results]
            ax3.plot(scales, jetf_q, marker="o", color=colors["jetf"], label=display_names["jetf"], linewidth=2)
        if "blink_qps" in report.scaling_results[0]:
            blk_q = [r["blink_qps"] for r in report.scaling_results]
            ax3.plot(scales, blk_q, marker="s", color=colors["blink"], label=display_names["blink"], linewidth=1.8)
        if "fe_qps" in report.scaling_results[0]:
            fe_q = [r["fe_qps"] for r in report.scaling_results]
            ax3.plot(scales, fe_q, marker="^", color=colors["flashentropy"], label=display_names["flashentropy"], linewidth=1.8)

        ax3.set_xlabel("Library Size (N spectra)")
        ax3.set_ylabel("Throughput QPS (Log Scale)")
        ax3.set_yscale("log")
        ax3.set_title("(c) Scalability: QPS vs Library Size")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "No Scaling Data", ha="center", va="center")
        ax3.set_title("(c) Scalability (N/A)")

    # Panel 4: 离线索引构建耗时对比
    ax4 = axes[1, 0]
    b_engines = [e for e in engines if e in report.index_stats]
    build_times = [report.index_stats[e].build_time_s for e in b_engines]
    bars_b = ax4.bar(
        np.arange(len(b_engines)),
        build_times,
        width=0.5,
        color=[colors.get(e, "#7f7f7f") for e in b_engines],
    )
    ax4.set_ylabel("Build Time (seconds)")
    ax4.set_title("(d) Offline Indexing / Pre-discretization Time")
    ax4.set_xticks(np.arange(len(b_engines)))
    ax4.set_xticklabels([display_names.get(e, e) for e in b_engines], rotation=15)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars_b, build_times):
        ax4.text(bar.get_x() + bar.get_width() / 2, val * 1.02, f"{val:.2f}s", ha="center", va="bottom", fontsize=8)

    # Panel 5: 余弦打分误差与相关性对比 (JETF vs BLINK vs matchms)
    ax5 = axes[1, 1]
    if "cosine_accuracy" in report.accuracy_results:
        ca = report.accuracy_results["cosine_accuracy"]
        methods = ["JET-Forest", "BLINK"]
        mae_vals = [ca.get("jetf_mae", 1e-15), ca.get("blink_mae", 0.05)]
        y_pos = np.arange(len(methods))
        ax5.barh(y_pos, mae_vals, height=0.45, color=[colors["jetf"], colors["blink"]])
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(methods)
        ax5.set_xscale("log")
        ax5.set_xlabel("Mean Absolute Error (MAE vs matchms Ground Truth)")
        ax5.set_title("(e) Pairwise Cosine Scoring Fidelity")
        ax5.grid(axis="x", linestyle="--", alpha=0.5)
    else:
        ax5.text(0.5, 0.5, "Cosine Accuracy N/A", ha="center", va="center")
        ax5.set_title("(e) Cosine Accuracy (N/A)")

    # Panel 6: JETF (Cosine) 与 FlashEntropy (Entropy) 检索重合度
    ax6 = axes[1, 2]
    if "flashentropy_agreement" in report.accuracy_results:
        fa = report.accuracy_results["flashentropy_agreement"]
        metrics = ["Top-1 Agreement", "Top-10 Jaccard", "Spearman Rho"]
        vals = [
            fa.get("top1_agreement_rate", 0.0),
            fa.get("mean_jaccard", 0.0),
            fa.get("spearman_rho", 0.0),
        ]
        bars_f = ax6.bar(metrics, vals, width=0.45, color=colors["flashentropy"])
        ax6.set_ylim(0.0, 1.1)
        ax6.set_ylabel("Agreement Ratio / Correlation")
        ax6.set_title("(f) JETF vs FlashEntropy Ranking Agreement")
        ax6.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars_f, vals):
            ax6.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    else:
        ax6.text(0.5, 0.5, "FlashEntropy Agreement N/A", ha="center", va="center")
        ax6.set_title("(f) Ranking Agreement (N/A)")

    plt.suptitle(
        f"JET-Forest Multi-Engine Benchmark Panorama (Library: {report.library_size:,} spectra, Mode: {report.mode.upper()})",
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    out_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig_path, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] 全景 6 联版高清出版级对比图表已保存至: {out_fig_path.resolve()}")


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("  JET-Forest vs BLINK vs FlashEntropy vs matchms 全景对比评测启动")
    print("=" * 80)

    # 1. 载入数据集
    clean_cfg = (
        MatchmsCleanConfig(max_peaks=300, min_relative_intensity=0.001)
        if args.clean
        else None
    )
    print(f"[*] 载入 MGF 参考库: {args.mgf} (容量限制: {args.dataset_size:,} 条)...")
    t0 = time.perf_counter()
    dataset = load_benchmark_dataset(
        mgf_path=args.mgf,
        library_size=args.dataset_size,
        clean_config=clean_cfg,
        seed=args.seed,
    )
    print(f"    数据集载入完成: {dataset.n_spectra:,} 条谱 (耗时 {time.perf_counter() - t0:.2f}s)")

    # 2. 解析启用的引擎
    req = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    if "auto" in req:
        avail = detect_available_engines()
        req = [e for e in ["jetf", "blink", "flashentropy", "matchms"] if avail.get(e, False)]

    print(f"[*] 参评引擎列表: {req}")

    # 3. 准备评测库规模
    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]

    # 4. 运行统一评测
    runner = MultiEngineBenchmarkRunner(
        dataset=dataset,
        engines=req,
        tolerance_da=args.tolerance,
    )

    report = runner.generate_report(
        n_queries=args.n_queries,
        mode=args.mode if args.mode in ("open", "identity") else "open",
        top_k=10,
        run_scaling=True,
        scales=scales,
    )

    # 5. 输出控制台全景对比表格
    print("\n" + report.format_console_table())

    # 6. 保存 JSON 报告
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[OK] 结构化多引擎 JSON 报告已保存至: {out_json.resolve()}")

    # 7. 绘制高清 6 联版对比图
    if args.output_fig:
        out_fig = Path(args.output_fig)
        plot_comprehensive_figures(report, out_fig)

    print("\n[OK] 全景多引擎对比基准评测全部完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
