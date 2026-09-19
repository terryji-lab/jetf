"""JET-Forest 命令行工具 (CLI)。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from jetf.builder import build_forest_index
from jetf.mgf import parse_mgf
from jetf.preprocessing import preprocess_library
from jetf.query import PrecursorWindow, QueryConfig, SearchMode
from jetf.search import search_forest
from jetf.serialization import load_forest_snapshot, save_forest_snapshot
from jetf.structure import ForestSpec


def cmd_build(args: argparse.Namespace) -> int:
    """构建森林索引快照。"""
    mgf_path = Path(args.mgf)
    out_path = Path(args.output)

    print(f"[*] 解析质谱文件: {mgf_path}")
    t0 = time.perf_counter()
    parsed = parse_mgf(mgf_path)
    t_parse = time.perf_counter() - t0
    print(f"    解析完成: {parsed.n_spectra} 条谱，共 {parsed.n_peaks} 峰 (耗时 {t_parse:.2f}s)")

    print("[*] 预处理与 L2 归一化...")
    t0 = time.perf_counter()
    library = preprocess_library(parsed)
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
    t_build = time.perf_counter() - t0
    print(f"    构建完成: {forest.n_trees} 棵小树，{forest.n_nodes} 个节点 (耗时 {t_build:.2f}s)")

    print(f"[*] 保存快照到: {out_path}")
    save_forest_snapshot(forest, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[✓] 快照保存成功! 文件大小: {size_mb:.2f} MB")
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
    p_build.add_argument("--tree-capacity", type=int, default=64, help="单树容量上限 (默认 64)")
    p_build.add_argument("--leaf-capacity", type=int, default=16, help="叶节点容量目标 (默认 16)")
    p_build.add_argument("--grid-da", type=float, default=0.02, help="网格摘要宽度 Da (默认 0.02)")

    # info 子命令
    p_info = subparsers.add_parser("info", help="查看已构建快照的元数据信息")
    p_info.add_argument("snapshot", type=str, help="快照 .npz 文件路径")

    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
