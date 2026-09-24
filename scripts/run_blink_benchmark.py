#!/usr/bin/env python
"""JET-Forest vs BLINK 基准评测套件可执行脚本。

功能：
1. 逐对谱打分数学等价性与精度评估 (Pairwise Accuracy vs matchms)
2. 全库检索 Top-K Jaccard 相似度与召回率 (Retrieval Accuracy vs matchms)
3. 相似/不相似 2x2 混淆矩阵评测 (Confusion Matrix & Classification Metrics)
4. 库规模伸缩性与对数线性模型外推 (Scaling Benchmark & Extrapolation up to 2M)
5. 自动生成 4 幅出版级高清图表与结构化 JSON / CSV 评测报告
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
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 确保项目与 BLINK 源码在 sys.path 中
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root / "blink") not in sys.path:
    sys.path.insert(0, str(repo_root / "blink"))

from jetf.benchmarks.blink_adapter import check_blink_available
from jetf.benchmarks.blink_accuracy import (
    evaluate_blink_confusion_matrix,
    evaluate_blink_pairwise_accuracy,
    evaluate_blink_retrieval_accuracy,
)
from jetf.benchmarks.blink_throughput import (
    benchmark_blink_scaling,
    extrapolate_blink_throughput,
    measure_jetf_snapshot_throughput,
)
from jetf.benchmarks.dataset import load_benchmark_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JET-Forest vs BLINK 基准测试与对比套件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mgf",
        type=str,
        default="GNPS-LIBRARY.mgf",
        help="输入 MGF 库文件路径（用于精度评测与小规模伸缩性）",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="all_gnps_forest.npz",
        help="全量 200 万库 ForestSnapshot 路径（用于大规模实测对比）",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "accuracy", "throughput"],
        default="all",
        help="运行评测模式",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=1000,
        help="逐对谱精度评测采样的谱对数量",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=50,
        help="检索评测与吞吐量评测采样的查询谱数量",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=2000,
        help="精度评测载入的基准库容量",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="100,500,1000,2000,5000",
        help="伸缩性吞吐评测的库规模列表 (逗号分隔)",
    )
    parser.add_argument(
        "-j",
        "--json",
        type=str,
        default="docs/benchmark-blink.json",
        help="JSON 格式评测结果输出路径",
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default="docs/benchmark-blink.csv",
        help="CSV 格式评测结果输出路径",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="docs/figures",
        help="生成的图表保存目录",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 图表绘制函数
# -----------------------------------------------------------------------------

def plot_score_scatter(
    pairwise_res,
    output_path: Path,
) -> None:
    """绘制得分散点图：BLINK vs matchms 与 JETF vs matchms。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    mms = pairwise_res.mms_scores
    blink = pairwise_res.blink_scores
    jetf = pairwise_res.jetf_scores

    # 1. BLINK vs matchms
    ax1 = axes[0]
    ax1.scatter(mms, blink, alpha=0.35, s=16, color="#d95f02", edgecolors="none", label="Spectrum Pairs")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.75, label="Perfect Agreement (y = x)")
    ax1.set_title("BLINK vs matchms (Ground Truth)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("matchms CosineGreedy Score", fontsize=11)
    ax1.set_ylabel("BLINK Cosine Score", fontsize=11)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle=":", alpha=0.6)

    text_blink = (
        f"MAE: {pairwise_res.mae:.6f}\n"
        f"RMSE: {pairwise_res.rmse:.6f}\n"
        f"Pearson r: {pairwise_res.pearson_r:.4f}\n"
        f"Spearman ρ: {pairwise_res.spearman_rho:.4f}\n"
        f"Discrepancy (>1e-3): {pairwise_res.discrepancy_rate * 100:.2f}%\n"
        f"Match Count Match: {pairwise_res.match_agreement_rate * 100:.1f}%\n"
        f"Bias: {pairwise_res.bias:+.6f}"
    )
    ax1.text(
        0.05, 0.95, text_blink, transform=ax1.transAxes,
        fontsize=9.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#d95f02", alpha=0.9),
    )
    ax1.legend(loc="lower right", framealpha=0.9)

    # 2. JET-Forest vs matchms
    ax2 = axes[1]
    ax2.scatter(mms, jetf, alpha=0.35, s=16, color="#2b5c8f", edgecolors="none", label="Spectrum Pairs")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.75, label="Perfect Agreement (y = x)")
    ax2.set_title("JET-Forest vs matchms (Ground Truth)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("matchms CosineGreedy Score", fontsize=11)
    ax2.set_ylabel("JET-Forest Score", fontsize=11)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle=":", alpha=0.6)

    text_jetf = (
        f"MAE: {pairwise_res.jetf_mae:.6f}\n"
        f"RMSE: {pairwise_res.jetf_rmse:.6f}\n"
        f"Pearson r: {pairwise_res.jetf_pearson_r:.4f}\n"
        f"Spearman ρ: {pairwise_res.jetf_spearman_rho:.4f}\n"
        f"Discrepancy (>1e-3): {pairwise_res.jetf_discrepancy_rate * 100:.2f}%\n"
        f"Match Count Match: {pairwise_res.jetf_match_agreement_rate * 100:.1f}%\n"
        f"Bias: {pairwise_res.jetf_bias:+.6f}"
    )
    ax2.text(
        0.05, 0.95, text_jetf, transform=ax2.transAxes,
        fontsize=9.5, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#2b5c8f", alpha=0.9),
    )
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] 得分散点图已保存: {output_path}")


def plot_confusion_matrices(
    blink_cm,
    jetf_cm,
    output_path: Path,
) -> None:
    """绘制 2x2 混淆矩阵热力图对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    cms = [("BLINK", blink_cm, axes[0], plt.cm.Oranges), ("JET-Forest", jetf_cm, axes[1], plt.cm.Blues)]

    for name, cm, ax, cmap in cms:
        matrix = np.array([[cm.tp, cm.fn], [cm.fp, cm.tn]])
        total = matrix.sum()

        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Similar (Pred)", "Dissimilar (Pred)"], fontsize=10.5)
        ax.set_yticklabels(["Similar (GT)", "Dissimilar (GT)"], fontsize=10.5)

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                pct = (val / total * 100.0) if total > 0 else 0.0
                cell_text = f"{val}\n({pct:.1f}%)"
                color = "white" if val > matrix.max() * 0.55 else "black"
                ax.text(j, i, cell_text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        title = (
            f"{name} (vs matchms)\n"
            f"F1={cm.f1:.3f} | Acc={cm.accuracy*100:.1f}% | Prec={cm.precision*100:.1f}% | Rec={cm.recall*100:.1f}%"
        )
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] 混淆矩阵热力图已保存: {output_path}")


def plot_throughput_extrapolation(
    scaling_res,
    extrap_res,
    output_path: Path,
) -> None:
    """绘制对数伸缩曲线与外推图 (Log-Log Scale)。"""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    # 1. 测量的 BLINK 点
    ax.scatter(
        scaling_res.scales,
        scaling_res.blink_latency_mean_ms,
        color="#d95f02",
        s=60,
        zorder=4,
        label="BLINK Measured Latency",
    )

    # 2. BLINK 拟合与外推曲线
    all_x = np.array(scaling_res.scales + extrap_res.target_scales)
    sort_idx = np.argsort(all_x)
    all_x_sorted = all_x[sort_idx]
    all_blink_y = 10.0 ** (extrap_res.alpha + extrap_res.beta * np.log10(all_x_sorted))

    ax.plot(
        all_x_sorted,
        all_blink_y,
        color="#d95f02",
        linestyle="--",
        lw=2.0,
        label=f"BLINK Linear Fit: log10(T) = {extrap_res.alpha:.2f} + {extrap_res.beta:.2f}·log10(N) (R²={extrap_res.r_squared:.4f})",
    )

    # 3. 测量的 JET-Forest 点
    ax.plot(
        scaling_res.scales,
        scaling_res.jetf_latency_mean_ms,
        color="#2b5c8f",
        marker="o",
        lw=2.0,
        label="JET-Forest Measured Latency",
    )

    # 4. 200 万实测 JET-Forest 点
    if extrap_res.jetf_2m_measured_latency_ms is not None:
        ax.scatter(
            [2003310],
            [extrap_res.jetf_2m_measured_latency_ms],
            color="#008080",
            s=120,
            marker="*",
            zorder=5,
            label=f"JET-Forest 2.0M Actual ({extrap_res.jetf_2m_measured_latency_ms:.1f} ms, QPS={extrap_res.jetf_2m_measured_qps:.1f})",
        )
        # 标注 2M 加速比
        if extrap_res.jetf_speedup_at_2m is not None:
            blink_2m = 10.0 ** (extrap_res.alpha + extrap_res.beta * np.log10(2003310))
            ax.annotate(
                f"Extrapolated BLINK: {blink_2m/1000.0:.1f} s\nJET-Forest: {extrap_res.jetf_2m_measured_latency_ms:.1f} ms\nSpeedup: {extrap_res.jetf_speedup_at_2m:.0f}x",
                xy=(2003310, extrap_res.jetf_2m_measured_latency_ms),
                xytext=(300000, 1500),
                arrowprops=dict(facecolor="#008080", shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f7f7", edgecolor="#008080"),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Library Size (N, Number of Spectra)", fontsize=11)
    ax.set_ylabel("Query Latency (ms, log scale)", fontsize=11)
    ax.set_title("Library Scaling & Latency Extrapolation (1-to-N Search)", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] 吞吐量外推曲线图已保存: {output_path}")


def plot_speedup_comparison(
    scaling_res,
    extrap_res,
    output_path: Path,
) -> None:
    """绘制不同库规模下的 JET-Forest 相对 BLINK 加速比柱状图。"""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    labels = [str(s) for s in scaling_res.scales]
    speedups = list(scaling_res.speedup_mean)

    # 加入外推规模 (仅展示真实测量的 2.0M 规模快照，避免未测规模的假想中介乘数)
    for s, b_lat in zip(extrap_res.target_scales, extrap_res.blink_extrapolated_latency_ms):
        if s == 2003310 and extrap_res.jetf_2m_measured_latency_ms is not None:
            labels.append("2.0M*")
            speedups.append(b_lat / extrap_res.jetf_2m_measured_latency_ms)

    x = np.arange(len(labels))
    colors = ["#2b5c8f" if "*" not in l else "#d95f02" for l in labels]

    bars = ax.bar(x, speedups, color=colors, width=0.6, edgecolor="black", alpha=0.85)

    for bar, sp in zip(bars, speedups):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.05,
            f"{sp:.1f}x" if sp < 100 else f"{int(round(sp))}x",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Library Size (* = Extrapolated BLINK vs Measured JET-Forest)", fontsize=11)
    ax.set_ylabel("Speedup Factor (JETF QPS / BLINK QPS, log scale)", fontsize=11)
    ax.set_title("JET-Forest Speedup over BLINK Across Library Sizes", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # 标注图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", label="Measured Speedup"),
        Patch(facecolor="#d95f02", label="Extrapolated Speedup"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] 加速比柱状图已保存: {output_path}")


# -----------------------------------------------------------------------------
# 报告持久化函数
# -----------------------------------------------------------------------------

def save_json_and_csv(
    report_dict: dict,
    json_path: Path,
    csv_path: Path,
) -> None:
    """将评测结果保存为结构化 JSON 与扁平化 CSV 表格。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 过滤不可序列化的 numpy array
    clean_dict = {}
    for k, v in report_dict.items():
        if isinstance(v, dict):
            clean_sub = {}
            for sk, sv in v.items():
                if isinstance(sv, np.ndarray):
                    clean_sub[sk] = sv.tolist()[:100]  # 仅存部分或标量
                elif isinstance(sv, (np.floating, np.integer)):
                    clean_sub[sk] = float(sv) if isinstance(sv, np.floating) else int(sv)
                else:
                    clean_sub[sk] = sv
            clean_dict[k] = clean_sub
        else:
            clean_dict[k] = v

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2, ensure_ascii=False)
    print(f"[Report] JSON 报告已保存: {json_path}")

    # 2. 导出 CSV 总结
    rows = []
    # 精度部分
    if "pairwise_accuracy" in report_dict:
        pa = report_dict["pairwise_accuracy"]
        rows.append(["Pairwise Accuracy", "Sampled Pairs", pa["n_pairs"], "-", "Sample count"])
        rows.append(["Pairwise Accuracy", "MAE", f"{pa['mae']:.6f}", f"{pa['jetf_mae']:.6f}", "Mean Absolute Error"])
        rows.append(["Pairwise Accuracy", "RMSE", f"{pa['rmse']:.6f}", f"{pa['jetf_rmse']:.6f}", "Root Mean Squared Error"])
        rows.append(["Pairwise Accuracy", "Pearson r", f"{pa['pearson_r']:.4f}", f"{pa.get('jetf_pearson_r', 1.0):.4f}", "Linear Correlation"])
        rows.append(["Pairwise Accuracy", "Spearman rho", f"{pa['spearman_rho']:.4f}", f"{pa.get('jetf_spearman_rho', 1.0):.4f}", "Rank Correlation"])
        rows.append(["Pairwise Accuracy", "Discrepancy Rate (>0.001)", f"{pa['discrepancy_rate']*100:.2f}%", f"{pa['jetf_discrepancy_rate']*100:.2f}%", "Fraction |diff| > 1e-3"])
        rows.append(["Pairwise Accuracy", "Match Agreement Rate", f"{pa['match_agreement_rate']*100:.2f}%", f"{pa['jetf_match_agreement_rate']*100:.2f}%", "Exact match count rate"])
        rows.append(["Pairwise Accuracy", "Bias (Mean Diff)", f"{pa['bias']:+.6f}", f"{pa['jetf_bias']:+.6f}", "Signed mean difference"])

    # 检索精度部分
    if "retrieval_accuracy" in report_dict:
        ra = report_dict["retrieval_accuracy"]
        rows.append(["Retrieval Accuracy", "Top-K Jaccard (JETF vs BLINK)", "-", f"{ra['mean_jaccard_jetf_blink']:.4f}", f"Top-{ra['k']} overlap"])
        rows.append(["Retrieval Accuracy", "Recall@K vs matchms", f"{ra['mean_recall_blink_mms']:.4f}", f"{ra['mean_recall_jetf_mms']:.4f}", f"Recall@Top-{ra['k']}"])

    # 混淆矩阵部分
    if "confusion_matrix" in report_dict:
        cm = report_dict["confusion_matrix"]
        b_cm = cm["blink"]
        j_cm = cm["jetf"]
        rows.append(["Confusion Matrix", "F1 Score", f"{b_cm['f1']:.4f}", f"{j_cm['f1']:.4f}", "Threshold score>=0.7, m>=6"])
        rows.append(["Confusion Matrix", "Accuracy", f"{b_cm['accuracy']*100:.2f}%", f"{j_cm['accuracy']*100:.2f}%", "Overall classification accuracy"])
        rows.append(["Confusion Matrix", "Precision", f"{b_cm['precision']*100:.2f}%", f"{j_cm['precision']*100:.2f}%", "Precision on similar pairs"])
        rows.append(["Confusion Matrix", "Recall", f"{b_cm['recall']*100:.2f}%", f"{j_cm['recall']*100:.2f}%", "Recall on similar pairs"])

    # 伸缩性部分
    if "scaling" in report_dict:
        sc = report_dict["scaling"]
        for idx, s in enumerate(sc["scales"]):
            b_lat = sc["blink_latency_mean_ms"][idx]
            j_lat = sc["jetf_latency_mean_ms"][idx]
            sp = sc["speedup_mean"][idx]
            rows.append(["Throughput Scaling", f"N={s} Latency (ms)", f"{b_lat:.2f}", f"{j_lat:.2f}", f"Speedup: {sp:.1f}x"])

    # 外推部分
    if "extrapolation" in report_dict:
        ex = report_dict["extrapolation"]
        rows.append(["Extrapolation", "Model Beta (Exponent)", f"{ex['beta']:.3f}", "Sublinear", "log-log slope"])
        rows.append(["Extrapolation", "Fit R^2", f"{ex['r_squared']:.4f}", "-", "Goodness of fit"])
        if ex.get("jetf_2m_measured_latency_ms"):
            b_2m = ex["blink_extrapolated_latency_ms"][-1]
            j_2m = ex["jetf_2m_measured_latency_ms"]
            sp_2m = ex["jetf_speedup_at_2m"]
            rows.append(["2M Scaling", "2.0M Spectra Latency (ms)", f"{b_2m:.1f} (extrap)", f"{j_2m:.2f} (measured)", f"2M Speedup: {sp_2m:.1f}x"])

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Metric", "BLINK", "JET-Forest", "Notes / Baseline"])
        writer.writerows(rows)
    print(f"[Report] CSV 报告已保存: {csv_path}")


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    check_blink_available()

    print("=" * 70)
    print(" JET-Forest vs BLINK 基准评测套件 (Benchmark Suite)")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"MGF 数据集: {args.mgf}")
    print(f"200万 快照: {args.snapshot}")
    print(f"图表输出目录: {args.plot_dir}")
    print("=" * 70)

    plot_dir = Path(args.plot_dir)
    json_path = Path(args.json)
    csv_path = Path(args.csv)

    report_dict: dict = {
        "timestamp": datetime.now().isoformat(),
        "mgf_path": args.mgf,
        "snapshot_path": args.snapshot,
        "mode": args.mode,
    }

    # 1. 载入数据集
    print(f"\n[1/4] 载入 MGF 库并构建基准索引 (size={args.dataset_size})...")
    t0 = time.perf_counter()
    dataset = load_benchmark_dataset(
        mgf_path=args.mgf,
        library_size=args.dataset_size,
    )
    print(f"数据集载入成功: {dataset.n_spectra} 条谱，耗时 {time.perf_counter() - t0:.2f}s")

    # 2. 精度评测
    if args.mode in ("all", "accuracy"):
        print(f"\n[2/4] 执行逐对打分与检索精度评测 (采样谱对={args.n_pairs})...")
        # 构造采样对
        rng = np.random.default_rng(42)
        n_lib = dataset.n_spectra
        idx1 = rng.integers(0, n_lib, size=args.n_pairs)
        idx2 = rng.integers(0, n_lib, size=args.n_pairs)
        pairs = [
            (dataset.library.peaks.spectrum_at(int(i)), dataset.library.peaks.spectrum_at(int(j)))
            for i, j in zip(idx1, idx2)
        ]

        # 逐对打分精度
        pw_res = evaluate_blink_pairwise_accuracy(pairs, tolerance=0.01, bin_width=0.001)
        print(f"  * BLINK MAE: {pw_res.mae:.6f} | RMSE: {pw_res.rmse:.6f} | Pearson r: {pw_res.pearson_r:.4f}")
        print(f"  * JET-Forest MAE: {pw_res.jetf_mae:.6f} | RMSE: {pw_res.jetf_rmse:.6f} | Discrepancy: {pw_res.jetf_discrepancy_rate*100:.2f}%")
        print(f"  * BLINK 异动率 (|diff|>1e-3): {pw_res.discrepancy_rate * 100:.2f}% | 匹配峰一致率: {pw_res.match_agreement_rate * 100:.1f}%")

        # 检索 Top-K 精度
        ret_res = evaluate_blink_retrieval_accuracy(
            dataset, n_queries=min(args.n_queries, 30), k=10
        )
        print(f"  * Top-10 Jaccard (JETF vs BLINK): {ret_res.mean_jaccard_jetf_blink:.4f}")
        print(f"  * Recall@10 (JETF vs matchms GT): {ret_res.mean_recall_jetf_mms:.4f}")
        print(f"  * Recall@10 (BLINK vs matchms GT): {ret_res.mean_recall_blink_mms:.4f}")

        # 混淆矩阵
        blink_cm, jetf_cm = evaluate_blink_confusion_matrix(pairs, score_thresh=0.7, match_thresh=6)
        print(f"  * BLINK 混淆矩阵指标: F1={blink_cm.f1:.4f}, Accuracy={blink_cm.accuracy*100:.1f}%, Prec={blink_cm.precision*100:.1f}%, Rec={blink_cm.recall*100:.1f}%")
        print(f"  * JET-Forest 混淆矩阵指标: F1={jetf_cm.f1:.4f}, Accuracy={jetf_cm.accuracy*100:.1f}%, Prec={jetf_cm.precision*100:.1f}%, Rec={jetf_cm.recall*100:.1f}%")

        report_dict["pairwise_accuracy"] = {
            "n_pairs": pw_res.n_pairs,
            "mae": pw_res.mae,
            "rmse": pw_res.rmse,
            "pearson_r": pw_res.pearson_r,
            "spearman_rho": pw_res.spearman_rho,
            "discrepancy_rate": pw_res.discrepancy_rate,
            "match_agreement_rate": pw_res.match_agreement_rate,
            "bias": pw_res.bias,
            "jetf_mae": pw_res.jetf_mae,
            "jetf_rmse": pw_res.jetf_rmse,
            "jetf_discrepancy_rate": pw_res.jetf_discrepancy_rate,
            "jetf_match_agreement_rate": pw_res.jetf_match_agreement_rate,
            "jetf_bias": pw_res.jetf_bias,
            "jetf_pearson_r": pw_res.jetf_pearson_r,
            "jetf_spearman_rho": pw_res.jetf_spearman_rho,
        }
        report_dict["retrieval_accuracy"] = {
            "n_queries": ret_res.n_queries,
            "k": ret_res.k,
            "mean_jaccard_jetf_blink": ret_res.mean_jaccard_jetf_blink,
            "mean_recall_jetf_mms": ret_res.mean_recall_jetf_mms,
            "mean_recall_blink_mms": ret_res.mean_recall_blink_mms,
        }
        report_dict["confusion_matrix"] = {
            "blink": {
                "tp": blink_cm.tp, "fp": blink_cm.fp, "fn": blink_cm.fn, "tn": blink_cm.tn,
                "precision": blink_cm.precision, "recall": blink_cm.recall, "f1": blink_cm.f1, "accuracy": blink_cm.accuracy,
            },
            "jetf": {
                "tp": jetf_cm.tp, "fp": jetf_cm.fp, "fn": jetf_cm.fn, "tn": jetf_cm.tn,
                "precision": jetf_cm.precision, "recall": jetf_cm.recall, "f1": jetf_cm.f1, "accuracy": jetf_cm.accuracy,
            },
        }

        # 绘制精度图表
        plot_score_scatter(pw_res, plot_dir / "blink_score_scatter.png")
        plot_confusion_matrices(blink_cm, jetf_cm, plot_dir / "blink_confusion_matrix.png")

    # 3. 伸缩性与吞吐评测
    if args.mode in ("all", "throughput"):
        print("\n[3/4] 执行库规模伸缩性与外推吞吐量评测...")
        scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]

        scaling_res = benchmark_blink_scaling(
            dataset, scales=scales, n_queries=args.n_queries, top_k=10
        )
        print("\n  规模评测结果汇总:")
        for idx, s in enumerate(scaling_res.scales):
            print(
                f"    N={s:6d}: BLINK={scaling_res.blink_latency_mean_ms[idx]:6.2f} ms (QPS={scaling_res.blink_qps[idx]:5.1f}), "
                f"JETF={scaling_res.jetf_latency_mean_ms[idx]:6.2f} ms (QPS={scaling_res.jetf_qps[idx]:6.1f}) -> 加速比 {scaling_res.speedup_mean[idx]:5.1f}x"
            )

        report_dict["scaling"] = {
            "scales": scaling_res.scales,
            "blink_latency_mean_ms": scaling_res.blink_latency_mean_ms,
            "blink_latency_p50_ms": scaling_res.blink_latency_p50_ms,
            "blink_latency_p95_ms": scaling_res.blink_latency_p95_ms,
            "blink_qps": scaling_res.blink_qps,
            "blink_disc_mean_ms": scaling_res.blink_disc_mean_ms,
            "blink_score_mean_ms": scaling_res.blink_score_mean_ms,
            "jetf_latency_mean_ms": scaling_res.jetf_latency_mean_ms,
            "jetf_latency_p50_ms": scaling_res.jetf_latency_p50_ms,
            "jetf_latency_p95_ms": scaling_res.jetf_latency_p95_ms,
            "jetf_qps": scaling_res.jetf_qps,
            "speedup_mean": scaling_res.speedup_mean,
        }

        # 200 万实测
        jetf_2m_ms = None
        if Path(args.snapshot).is_file():
            print(f"\n  检测到 200 万快照 ({args.snapshot})，正在进行实测验证...")
            jetf_2m_ms, jetf_2m_qps = measure_jetf_snapshot_throughput(
                args.snapshot, n_queries=min(args.n_queries, 20), top_k=10
            )
            print(f"  * JET-Forest 200万 (2,003,310) 真实检索时延: {jetf_2m_ms:.2f} ms, QPS: {jetf_2m_qps:.1f}")

        # 外推模型计算
        extrap_res = extrapolate_blink_throughput(
            scaling_res.scales,
            scaling_res.blink_latency_mean_ms,
            target_scales=(50000, 100000, 500000, 2003310),
            jetf_2m_latency_ms=jetf_2m_ms,
        )

        print(f"\n  外推模型: log10(T_ms) = {extrap_res.alpha:.3f} + {extrap_res.beta:.3f} * log10(N), R^2={extrap_res.r_squared:.4f}")
        for s, lat, q in zip(extrap_res.target_scales, extrap_res.blink_extrapolated_latency_ms, extrap_res.blink_extrapolated_qps):
            print(f"    外推规模 N={s:8d}: BLINK 预测时延 = {lat:8.1f} ms, 预测 QPS = {q:5.2f}")
        if extrap_res.jetf_speedup_at_2m is not None:
            print(f"  >>> JET-Forest 200万真实吞吐相对 BLINK 外推加速比: {extrap_res.jetf_speedup_at_2m:.1f}x <<<")

        report_dict["extrapolation"] = {
            "alpha": extrap_res.alpha,
            "beta": extrap_res.beta,
            "r_squared": extrap_res.r_squared,
            "target_scales": extrap_res.target_scales,
            "blink_extrapolated_latency_ms": extrap_res.blink_extrapolated_latency_ms,
            "blink_extrapolated_qps": extrap_res.blink_extrapolated_qps,
            "jetf_2m_measured_latency_ms": extrap_res.jetf_2m_measured_latency_ms,
            "jetf_2m_measured_qps": extrap_res.jetf_2m_measured_qps,
            "jetf_speedup_at_2m": extrap_res.jetf_speedup_at_2m,
        }

        # 绘制伸缩性图表
        plot_throughput_extrapolation(scaling_res, extrap_res, plot_dir / "throughput_extrapolation.png")
        plot_speedup_comparison(scaling_res, extrap_res, plot_dir / "speedup_comparison.png")

    # 4. 保存报告
    print("\n[4/4] 保存评测报告...")
    save_json_and_csv(report_dict, json_path, csv_path)

    print("\n" + "=" * 70)
    print(" JET-Forest vs BLINK 基准评测顺利完成！")
    print(f" JSON 报告: {json_path}")
    print(f" CSV 报告:  {csv_path}")
    print(f" 生成图表:  {plot_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
