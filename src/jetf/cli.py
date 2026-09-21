"""JET-Forest 命令行工具 (CLI)。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from jetf.builder import build_forest_index
from jetf.cleaning import MatchmsCleanConfig, clean_parsed_library, silence_matchms_logging
from jetf.mgf import parse_mgf
from jetf.preprocessing import preprocess_library
from jetf.serialization import load_forest_snapshot, save_forest_snapshot
from jetf.structure import ForestSpec


def cmd_build(args: argparse.Namespace) -> int:
    """构建森林索引快照。"""
    # 静音 matchms 日志，避免处理百万谱图时输出数十万条 WARNING 刷屏阻塞 I/O
    silence_matchms_logging()

    mgf_path = Path(args.mgf)
    out_path = Path(args.output)
    if out_path.exists() and not getattr(args, "force", False):
        print(f"[ERROR] 快照文件已存在: {out_path}。若需覆盖请指定 --force / -f 参数。")
        return 1

    keep_rejected = getattr(args, "keep_rejected", False)

    if getattr(args, "clean", True):
        clean_cfg = MatchmsCleanConfig(
            max_peaks=args.clean_max_peaks,
            min_relative_intensity=args.clean_min_rel,
        )
        print(f"[*] 解析并流式执行 matchms 工业级谱图清洗: {mgf_path} (max_peaks={args.clean_max_peaks}, min_rel={args.clean_min_rel})...")
        t0 = time.perf_counter()
        parsed = parse_mgf(mgf_path, clean_config=clean_cfg, keep_rejected=keep_rejected)
        t_clean = time.perf_counter() - t0
        print(f"    解析与清洗就绪: {parsed.n_spectra:,} 条有效谱，共 {parsed.n_peaks:,} 峰 (耗时 {t_clean:.2f}s)")
    else:
        print(f"[*] 解析质谱文件 (跳过清洗): {mgf_path}")
        t0 = time.perf_counter()
        parsed = parse_mgf(mgf_path, keep_rejected=keep_rejected)
        t_parse = time.perf_counter() - t0
        print(f"    解析完成: {parsed.n_spectra:,} 条谱，共 {parsed.n_peaks:,} 峰 (耗时 {t_parse:.2f}s)")

    print("[*] 预处理与 L2 归一化...")
    t0 = time.perf_counter()
    library = preprocess_library(parsed)
    del parsed
    import gc
    gc.collect()
    t_prep = time.perf_counter() - t0
    print(f"    预处理完成 (耗时 {t_prep:.2f}s)")

    spec = ForestSpec(
        tree_capacity=args.tree_capacity,
        leaf_capacity=args.leaf_capacity,
        summary_grid_da=args.grid_da,
    )

    print(f"[*] 构建包络森林 (Tree={spec.tree_capacity}, Leaf={spec.leaf_capacity}, Grid={spec.summary_grid_da} Da)...")
    t0 = time.perf_counter()
    forest = build_forest_index(library, spec)
    del library
    gc.collect()
    t_build = time.perf_counter() - t0
    print(f"    构建完成: {forest.n_trees:,} 棵小树，{forest.n_nodes:,} 个节点 (耗时 {t_build:.2f}s)")

    print(f"[*] 保存快照到: {out_path}")
    save_forest_snapshot(forest, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] 快照保存成功! 文件大小: {size_mb:.2f} MB")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """打印快照信息。"""
    snap_path = Path(args.snapshot)
    print(f"[*] 加载快照: {snap_path}")
    forest = load_forest_snapshot(snap_path)
    print("=" * 50)
    print(f"总覆盖谱数:   {forest.n_spectra}")
    print(f"森林小树数:   {forest.n_trees}")
    print(f"节点总数:     {forest.n_nodes}")
    print(f"硬分区数:     {len(forest.partitions)}")
    print(f"零能量谱数:   {forest.zero_energy_members.n_members}")
    print(f"规格版本:     {forest.spec.versioned_id}")
    print(f"树容量配置:   {forest.spec.tree_capacity}")
    print(f"叶容量配置:   {forest.spec.leaf_capacity}")
    print(f"摘要网格宽度: {forest.spec.summary_grid_da} Da")
    print("=" * 50)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """运行 JET-Forest 与 matchms 的性能及一致性对比基准。"""
    from jetf.benchmarks.adapter import check_matchms_available
    from jetf.benchmarks.consistency import evaluate_pairwise_consistency, evaluate_retrieval_consistency
    from jetf.benchmarks.dataset import (
        load_benchmark_dataset,
        sample_query_spectra,
        sample_query_spectra_from_forest,
    )
    from jetf.benchmarks.reporter import (
        format_pairwise_consistency_table,
        format_pairwise_throughput_table,
        format_retrieval_consistency_table,
        format_retrieval_throughput_table,
        generate_full_markdown_report,
        save_json_report,
        save_csv_report,
    )
    from jetf.benchmarks.throughput import benchmark_pairwise_throughput, benchmark_retrieval_throughput
    from jetf.query import QueryConfig, SearchMode
    from jetf.types import IonMode

    print("=" * 70)
    print("  JET-Forest vs matchms 全景基准评测 (吞吐量与结果一致性)")
    print("=" * 70)

    # 1. 优先分支：直接从脱机快照评测 (零 MGF 依赖，秒级载入)
    if getattr(args, "snapshot", None):
        snap_path = Path(args.snapshot)
        if not snap_path.is_file():
            print(f"[ERROR] 指定的快照文件不存在: {snap_path}")
            return 1

        if args.mode == "consistency":
            print("[ERROR] 脱机快照评测当前仅支持 --mode throughput。如需与 matchms 进行全库检索排序与零漏检一致性校验，请指定 MGF 输入模式。")
            return 1

        print(f"[*] 直接从脱机快照加载森林索引: {snap_path}...")
        t0 = time.perf_counter()
        forest = load_forest_snapshot(snap_path)
        print(f"    快照就绪: {forest.n_spectra:,} 条谱，加载耗时 {time.perf_counter() - t0:.2f}s")

        seed = getattr(args, "seed", 2026)
        print(f"[*] 从快照中抽样代表性查询谱 ({args.n_queries} 条, seed={seed})...")
        queries = sample_query_spectra_from_forest(forest, n_queries=args.n_queries, seed=seed)
        print(f"    抽样完成: {len(queries)} 条查询谱")

        skip_matchms = getattr(args, "skip_matchms", False)
        if skip_matchms:
            print("[*] 已启用 --skip-matchms: 仅评测 JET-Forest 自身检索吞吐量与时延 (跳过 matchms 穷举)")
        else:
            check_matchms_available()

        retrieval_tp_list = []
        pairwise_tp = None
        pairwise_cons = None
        retrieval_cons_list = []

        if args.mode in ("all", "throughput"):
            print("\n" + "-" * 70)
            print("  阶段: 1-to-N 全库开放检索宏观吞吐量与时延 (Macrobenchmark)")
            print("-" * 70)

            # 场景 A: 开放检索 Top-10 (全库无限制)
            tp_queries_top10 = []
            for row, q in queries:
                meta = forest.spectra[row]
                c = QueryConfig(
                    mode=SearchMode.TOP_K,
                    k=10,
                    fragment_tolerance_da=args.tolerance,
                    ion_mode=meta.ion_mode,
                )
                tp_queries_top10.append((row, q, c))
            tp_open_top10 = benchmark_retrieval_throughput(
                forest,
                tp_queries_top10,
                mode_name="开放检索 Top-10 (全库无限制)",
                skip_matchms=skip_matchms,
                concurrency=args.concurrency,
            )
            retrieval_tp_list.append(tp_open_top10)

            # 场景 B: 开放检索 Top-5 (全库无限制)
            tp_queries_top5 = []
            for row, q in queries:
                meta = forest.spectra[row]
                c = QueryConfig(
                    mode=SearchMode.TOP_K,
                    k=5,
                    fragment_tolerance_da=args.tolerance,
                    ion_mode=meta.ion_mode,
                )
                tp_queries_top5.append((row, q, c))
            tp_open_top5 = benchmark_retrieval_throughput(
                forest,
                tp_queries_top5,
                mode_name="开放检索 Top-5 (全库无限制)",
                skip_matchms=skip_matchms,
                concurrency=args.concurrency,
            )
            retrieval_tp_list.append(tp_open_top5)

            # 场景 C: 开放检索 Threshold >= 0.50 (全库无限制)
            tp_queries_thresh = []
            for row, q in queries:
                meta = forest.spectra[row]
                c = QueryConfig(
                    mode=SearchMode.THRESHOLD,
                    threshold=0.50,
                    fragment_tolerance_da=args.tolerance,
                    ion_mode=meta.ion_mode,
                )
                tp_queries_thresh.append((row, q, c))
            tp_open_thresh = benchmark_retrieval_throughput(
                forest,
                tp_queries_thresh,
                mode_name="开放检索 Threshold >= 0.50",
                skip_matchms=skip_matchms,
                concurrency=args.concurrency,
            )
            retrieval_tp_list.append(tp_open_thresh)

            print(format_retrieval_throughput_table(retrieval_tp_list))

        config_meta = {
            "mode": args.mode,
            "library_size": forest.n_spectra,
            "n_queries": len(queries),
            "n_pairs": 0,
            "tolerance": args.tolerance,
            "tolerance_da": args.tolerance,
            "concurrency": args.concurrency,
            "seed": seed,
            "snapshot": str(snap_path),
            "skip_matchms": skip_matchms,
            "source_type": "snapshot",
        }

        if args.output_md:
            md_content = generate_full_markdown_report(
                pairwise_consistency=pairwise_cons,
                retrieval_consistency=retrieval_cons_list,
                pairwise_throughput=pairwise_tp,
                retrieval_throughput=retrieval_tp_list,
            )
            out_p = Path(args.output_md)
            out_p.write_text(md_content, encoding="utf-8")
            print(f"\n[OK] 完整 Markdown 评测报告已保存至: {out_p.resolve()}")

        if args.output_json:
            out_json_p = Path(args.output_json)
            save_json_report(
                out_json_p,
                pairwise_consistency=pairwise_cons,
                retrieval_consistency=retrieval_cons_list,
                pairwise_throughput=pairwise_tp,
                retrieval_throughput=retrieval_tp_list,
                config_metadata=config_meta,
            )
            print(f"\n[OK] 结构化 JSON 评测结果已保存至: {out_json_p.resolve()}")

        if getattr(args, "output_csv", None):
            out_csv_p = Path(args.output_csv)
            csv_paths = save_csv_report(
                out_csv_p,
                retrieval_throughput=retrieval_tp_list,
                retrieval_consistency=retrieval_cons_list,
                pairwise_throughput=pairwise_tp,
                pairwise_consistency=pairwise_cons,
            )
            for p in csv_paths:
                print(f"[OK] 结构化 CSV 评测结果已保存至: {p.resolve()}")

        print("\n[OK] 快照脱机基准评测全部完成！")
        return 0

    # 2. 常规分支：从 MGF 解析与清洗构建基准数据集
    if not getattr(args, "skip_matchms", False):
        check_matchms_available()

    print(f"[*] 加载基准数据集 (目标库容量: {args.library_size:,} 条谱)...")
    t0 = time.perf_counter()
    clean_cfg = (
        MatchmsCleanConfig(
            max_peaks=args.clean_max_peaks,
            min_relative_intensity=args.clean_min_rel,
        )
        if getattr(args, "clean", True)
        else None
    )
    if clean_cfg:
        print(f"[*] 已启用 matchms 工业级清洗 (max_peaks={args.clean_max_peaks}, min_rel={args.clean_min_rel})")
    else:
        print("[*] 跳过 matchms 清洗 (--no-clean 指定)")
    seed = getattr(args, "seed", 2026)
    dataset = load_benchmark_dataset(
        mgf_path=args.mgf,
        library_size=args.library_size,
        clean_config=clean_cfg,
        max_records=getattr(args, "max_records", None),
        seed=seed,
    )
    print(f"    数据集就绪: {dataset.n_spectra:,} 条谱，构建森林耗时 {time.perf_counter() - t0:.2f}s")

    print(f"[*] 抽样代表性查询谱 ({args.n_queries} 条, seed={seed})...")
    queries = sample_query_spectra(dataset.library, n_queries=args.n_queries, seed=seed)
    print(f"    抽样完成: {len(queries)} 条查询谱")

    # 构建代表性谱对（用于算子微基准与单对一致性测试，采用分层抽样）
    pairs: list[tuple[Any, Any]] = []

    # 1. 近前体谱对 (前体相近 <= 0.5 Da, 排除自匹配)
    rng = np.random.default_rng(seed)
    prec_list = [
        (i, dataset.library.spectra[i].precursor_mz)
        for i in range(dataset.library.n_spectra)
        if dataset.library.spectra[i].precursor_mz is not None
    ]
    prec_list.sort(key=lambda x: x[1])  # type: ignore[arg-type]
    near_pairs = []
    for idx in range(len(prec_list) - 1):
        if prec_list[idx][0] != prec_list[idx + 1][0]:
            if prec_list[idx + 1][1] - prec_list[idx][1] <= 0.5:  # type: ignore[operator]
                near_pairs.append((prec_list[idx][0], prec_list[idx + 1][0]))
    if near_pairs:
        chosen_near = rng.choice(len(near_pairs), size=min(100, len(near_pairs)), replace=False)
        for c_idx in chosen_near:
            i, j = near_pairs[c_idx]
            pairs.append((dataset.library.peaks.spectrum_at(int(i)), dataset.library.peaks.spectrum_at(int(j))))

    # 2. 随机库谱对（大多不相交，排除自配对并去重）
    remaining = max(0, args.n_pairs - len(pairs))
    if remaining > 0:
        seen_pairs = {(near_pairs[c][0], near_pairs[c][1]) for c in (chosen_near if near_pairs else [])}
        while len(pairs) < args.n_pairs:
            batch_sz = max(100, remaining * 2)
            r1 = rng.choice(dataset.library.n_spectra, size=batch_sz, replace=True)
            r2 = rng.choice(dataset.library.n_spectra, size=batch_sz, replace=True)
            for i, j in zip(r1, r2):
                if i != j:
                    pkey = (min(int(i), int(j)), max(int(i), int(j)))
                    if pkey not in seen_pairs:
                        seen_pairs.add(pkey)
                        pairs.append((dataset.library.peaks.spectrum_at(int(i)), dataset.library.peaks.spectrum_at(int(j))))
                        if len(pairs) >= args.n_pairs:
                            break

    pairwise_cons = None
    retrieval_cons_list = []
    pairwise_tp = None
    retrieval_tp_list = []

    # 1. 结果一致性评测
    if args.mode in ("all", "consistency"):
        print("\n" + "-" * 70)
        print("  阶段 1: 结果一致性评测 (Result Consistency)")
        print("-" * 70)

        print("[*] 正在评测单对谱打分一致性 (Pairwise Scoring Equivalence)...")
        pairwise_cons = evaluate_pairwise_consistency(pairs, tolerance_da=args.tolerance)
        print(format_pairwise_consistency_table(pairwise_cons))

        print("\n[*] 正在评测 1-to-N 全库开放检索排序一致性与零漏检 (Open Search)...")
        # 场景 A: 开放检索 Top-10 (排除自身)
        open_queries_top10 = []
        for row, q in queries:
            meta = dataset.library.spectra[row]
            c = QueryConfig(
                mode=SearchMode.TOP_K,
                k=10,
                fragment_tolerance_da=args.tolerance,
                ion_mode=meta.ion_mode,
                exclude_spectrum_id=meta.external_id,
            )
            open_queries_top10.append((row, q, c))

        summary_open_top10 = evaluate_retrieval_consistency(
            dataset, open_queries_top10, mode_name="开放检索 Top-10 (排除自身)"
        )
        retrieval_cons_list.append(summary_open_top10)

        # 场景 B: 开放检索 Top-5 (排除自身)
        open_queries_top5 = []
        for row, q in queries:
            meta = dataset.library.spectra[row]
            c = QueryConfig(
                mode=SearchMode.TOP_K,
                k=5,
                fragment_tolerance_da=args.tolerance,
                ion_mode=meta.ion_mode,
                exclude_spectrum_id=meta.external_id,
            )
            open_queries_top5.append((row, q, c))

        summary_open_top5 = evaluate_retrieval_consistency(
            dataset, open_queries_top5, mode_name="开放检索 Top-5 (排除自身)"
        )
        retrieval_cons_list.append(summary_open_top5)

        # 场景 C: 开放检索 Threshold >= 0.50 (排除自身)
        open_queries_thresh = []
        for row, q in queries:
            meta = dataset.library.spectra[row]
            c = QueryConfig(
                mode=SearchMode.THRESHOLD,
                threshold=0.50,
                fragment_tolerance_da=args.tolerance,
                ion_mode=meta.ion_mode,
                exclude_spectrum_id=meta.external_id,
            )
            open_queries_thresh.append((row, q, c))

        summary_open_thresh = evaluate_retrieval_consistency(
            dataset, open_queries_thresh, mode_name="开放检索 Threshold >= 0.50 (排除自身)"
        )
        retrieval_cons_list.append(summary_open_thresh)

        print(format_retrieval_consistency_table(retrieval_cons_list))

    # 2. 吞吐量与性能评测
    if args.mode in ("all", "throughput"):
        print("\n" + "-" * 70)
        print("  阶段 2: 吞吐量与耗时基准评测 (Throughput & Latency)")
        print("-" * 70)

        print("[*] 正在测量逐对谱打分算子微基准 (Kernel Microbenchmark)...")
        pairwise_tp = benchmark_pairwise_throughput(pairs, tolerance_da=args.tolerance)
        print(format_pairwise_throughput_table(pairwise_tp))

        print("\n[*] 正在测量 1-to-N 全库开放检索端到端吞吐量与时延 (Macrobenchmark)...")
        # 场景 A: 开放检索 Top-10 (全库无限制)
        tp_queries_top10 = []
        for row, q in queries:
            meta = dataset.library.spectra[row]
            c = QueryConfig(
                mode=SearchMode.TOP_K,
                k=10,
                fragment_tolerance_da=args.tolerance,
                ion_mode=meta.ion_mode,
            )
            tp_queries_top10.append((row, q, c))
        skip_matchms = getattr(args, "skip_matchms", False)
        tp_open_top10 = benchmark_retrieval_throughput(
            dataset,
            tp_queries_top10,
            mode_name="开放检索 Top-10 (全库无限制)",
            skip_matchms=skip_matchms,
            concurrency=args.concurrency,
        )
        retrieval_tp_list.append(tp_open_top10)

        # 场景 B: 开放检索 Top-5 (全库无限制)
        tp_queries_top5 = []
        for row, q in queries:
            meta = dataset.library.spectra[row]
            c = QueryConfig(
                mode=SearchMode.TOP_K,
                k=5,
                fragment_tolerance_da=args.tolerance,
                ion_mode=meta.ion_mode,
            )
            tp_queries_top5.append((row, q, c))
        tp_open_top5 = benchmark_retrieval_throughput(
            dataset,
            tp_queries_top5,
            mode_name="开放检索 Top-5 (全库无限制)",
            skip_matchms=skip_matchms,
            concurrency=args.concurrency,
        )
        retrieval_tp_list.append(tp_open_top5)

        # 场景 C: 开放检索 Threshold >= 0.50 (全库无限制)
        tp_queries_thresh = []
        for row, q in queries:
            meta = dataset.library.spectra[row]
            c = QueryConfig(
                mode=SearchMode.THRESHOLD,
                threshold=0.50,
                fragment_tolerance_da=args.tolerance,
                ion_mode=meta.ion_mode,
            )
            tp_queries_thresh.append((row, q, c))
        tp_open_thresh = benchmark_retrieval_throughput(
            dataset,
            tp_queries_thresh,
            mode_name="开放检索 Threshold >= 0.50",
            skip_matchms=skip_matchms,
            concurrency=args.concurrency,
        )
        retrieval_tp_list.append(tp_open_thresh)

        print(format_retrieval_throughput_table(retrieval_tp_list))

    # 3. 报告输出
    config_meta = {
        "mode": args.mode,
        "library_size": dataset.n_spectra,
        "n_queries": len(queries),
        "n_pairs": len(pairs),
        "tolerance": args.tolerance,
        "tolerance_da": args.tolerance,
        "seed": getattr(args, "seed", 2026),
        "mgf": str(dataset.parsed.source_path),
        "clean": bool(getattr(args, "clean", True)),
        "clean_max_peaks": args.clean_max_peaks if getattr(args, "clean", True) else None,
        "clean_min_rel": args.clean_min_rel if getattr(args, "clean", True) else None,
    }

    if args.output_md:
        md_content = generate_full_markdown_report(
            pairwise_consistency=pairwise_cons,
            retrieval_consistency=retrieval_cons_list,
            pairwise_throughput=pairwise_tp,
            retrieval_throughput=retrieval_tp_list,
        )
        out_p = Path(args.output_md)
        out_p.write_text(md_content, encoding="utf-8")
        print(f"\n[OK] 完整 Markdown 评测报告已保存至: {out_p.resolve()}")

    if args.output_json:
        out_json_p = Path(args.output_json)
        save_json_report(
            out_json_p,
            pairwise_consistency=pairwise_cons,
            retrieval_consistency=retrieval_cons_list,
            pairwise_throughput=pairwise_tp,
            retrieval_throughput=retrieval_tp_list,
            config_metadata=config_meta,
        )
        print(f"\n[OK] 结构化 JSON 评测结果已保存至: {out_json_p.resolve()}")

    if getattr(args, "output_csv", None):
        out_csv_p = Path(args.output_csv)
        csv_paths = save_csv_report(
            out_csv_p,
            retrieval_throughput=retrieval_tp_list,
            retrieval_consistency=retrieval_cons_list,
            pairwise_throughput=pairwise_tp,
            pairwise_consistency=pairwise_cons,
        )
        for p in csv_paths:
            print(f"[OK] 结构化 CSV 评测结果已保存至: {p.resolve()}")

    print("\n[OK] 基准评测全部完成！")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jetf",
        description="JET-Forest: 前体自适应浅层包络森林质谱检索引擎",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # build 子命令
    p_build = subparsers.add_parser("build", help="从 MGF 文件编译并构建森林快照")
    p_build.add_argument("mgf", type=str, help="输入 MGF 文件路径")
    p_build.add_argument("-o", "--output", type=str, required=True, help="输出 .npz 快照路径")
    p_build.add_argument("-f", "--force", action="store_true", help="若输出文件已存在则强制覆盖")
    p_build.add_argument("--tree-capacity", type=int, default=64, help="单树容量上限 (默认 64)")
    p_build.add_argument("--leaf-capacity", type=int, default=16, help="叶节点容量目标 (默认 16)")
    p_build.add_argument("--grid-da", type=float, default=0.02, help="网格摘要宽度 Da (默认 0.02)")
    p_build.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        default=True,
        help="构建索引前执行 matchms 工业级谱图清洗 (默认开启)",
    )
    p_build.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="跳过 matchms 谱图清洗 (输入已预清洗时使用)",
    )
    p_build.add_argument("--clean-max-peaks", type=int, default=300, help="清洗时单谱最多保留峰数 (默认 300)")
    p_build.add_argument("--clean-min-rel", type=float, default=0.001, help="清洗时相对强度阈值 (默认 0.001)")
    p_build.add_argument(
        "--keep-rejected",
        action="store_true",
        default=False,
        help="保留被隔离谱图的详细记录列表 (默认关闭以节省大库内存)",
    )

    # info 子命令
    p_info = subparsers.add_parser("info", help="查看已构建快照的元数据信息")
    p_info.add_argument("snapshot", type=str, help="快照 .npz 文件路径")

    # benchmark 子命令
    p_bench = subparsers.add_parser("benchmark", help="运行与 matchms 的吞吐量与一致性全景对比评测")
    p_bench.add_argument(
        "-s",
        "--snapshot",
        type=str,
        default=None,
        help="已编译的 ForestIndex 快照路径 (.npz)，指定时跳过 MGF 解析直接脱机评测",
    )
    p_bench.add_argument(
        "--skip-matchms",
        action="store_true",
        default=False,
        help="跳过 matchms 穷举对比，仅评测 JET-Forest 自身检索吞吐量与时延 (百万级大库推荐)",
    )
    p_bench.add_argument(
        "--mode",
        type=str,
        choices=["all", "consistency", "throughput"],
        default="all",
        help="评测模式: all (全部), consistency (仅一致性), throughput (仅吞吐量)",
    )
    p_bench.add_argument("--mgf", type=str, default=None, help="参考库 MGF 文件路径 (默认自动查找 GNPS-LIBRARY.mgf)")
    p_bench.add_argument("--library-size", type=int, default=2000, help="测试参考库容量大小 (默认 2000)")
    p_bench.add_argument("--n-queries", type=int, default=30, help="抽样查询谱数量 (默认 30)")
    p_bench.add_argument("--n-pairs", type=int, default=1000, help="算子微基准测试谱对数 (默认 1000)")
    p_bench.add_argument("--tolerance", type=float, default=0.02, help="匹配容差 Da (默认 0.02)")
    p_bench.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发查询数 (默认 1，多核吞吐评测推荐 >= 2)",
    )
    p_bench.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        default=True,
        help="基准评测前执行 matchms 工业级谱图清洗 (默认开启)",
    )
    p_bench.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="跳过 matchms 谱图清洗",
    )
    p_bench.add_argument("--clean-max-peaks", type=int, default=300, help="清洗时单谱最多保留峰数 (默认 300)")
    p_bench.add_argument("--clean-min-rel", type=float, default=0.001, help="清洗时相对强度阈值 (默认 0.001)")
    p_bench.add_argument("-o", "--output-md", type=str, default=None, help="输出 Markdown 报告路径")
    p_bench.add_argument(
        "-j",
        "--output-json",
        type=str,
        default=None,
        help="输出结构化 JSON 实验结果文件路径",
    )
    p_bench.add_argument(
        "-c",
        "--output-csv",
        type=str,
        default=None,
        help="输出结构化 CSV 实验结果文件路径",
    )
    p_bench.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="从 MGF 文件最大读取谱图数 (默认对于大文件自动智能限制)",
    )
    p_bench.add_argument("--seed", type=int, default=2026, help="基准评测随机数种子 (默认 2026)")

    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
