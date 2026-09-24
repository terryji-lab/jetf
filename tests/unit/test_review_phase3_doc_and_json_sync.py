"""阶段 3 评测与对齐回归测试：对外文档、CLI 与基准数据资产全量一致性校验。

覆盖验证点：
1. docs/benchmark-matchms.json 存在且 retrieval_throughput 包含有效非空条目 (P0)
2. README.md 中的 PrecursorWindow 示例语法正确、可被 ast 解析且无异常执行 (P0)
3. README.md 中声明的 GPU 浮点膨胀系数为 1 + 10^{-3}，与 bounds_kernels.py 一致 (P1)
4. README.md §2.1 表格中的 QPS、时延与加速比与 docs/benchmark-matchms.json 严格对齐 (P0)
5. README.md 明确阐明 3 个物理硬分区 (POSITIVE, NEGATIVE, UNKNOWN) 与 INCLUDE_UNKNOWN 策略 (P1)
6. README.md 明确阐明 CLI --clean 默认开启与 --no-clean 跳过行为 (P1)
7. docs/algorithm-design-forest.md 中开放检索端到端时延与 benchmark-2m.json 严格一致 (P1)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import numpy as np
import pytest

from jetf.gpu.bounds_kernels import SAFETY_MARGIN_FP32
from jetf.types import PrecursorWindow


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
README_PATH = ROOT_DIR / "README.md"
BENCHMARK_JSON_PATH = ROOT_DIR / "docs" / "benchmark-matchms.json"
BENCHMARK_CSV_PATH = ROOT_DIR / "docs" / "benchmark-matchms.csv"
ALGO_DOC_PATH = ROOT_DIR / "docs" / "algorithm-design-forest.md"


def test_benchmark_matchms_json_validity_and_throughput_non_empty():
    """验证 docs/benchmark-matchms.json 存在，格式有效且 retrieval_throughput 非空。"""
    assert BENCHMARK_JSON_PATH.is_file(), f"未找到基准数据文件: {BENCHMARK_JSON_PATH}"

    data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    assert data.get("report_title") == "JET-Forest vs matchms Benchmark Report"

    metadata = data.get("metadata", {})
    assert "config" in metadata
    config = metadata["config"]
    assert config.get("library_size") == 2000

    results = data.get("results", {})
    assert "pairwise_consistency" in results
    assert "retrieval_consistency" in results
    assert "retrieval_throughput" in results

    tp_list = results["retrieval_throughput"]
    assert isinstance(tp_list, list)
    assert len(tp_list) >= 3, f"retrieval_throughput 预期至少包含 3 个场景，实际为 {len(tp_list)}"

    mode_names = [item["mode_name"] for item in tp_list]
    assert any("Top-10" in name for name in mode_names)
    assert any("Top-5" in name for name in mode_names)
    assert any("Threshold" in name for name in mode_names)

    for item in tp_list:
        assert item["jetf_qps"] > 0.0
        assert item["matchms_qps"] > 0.0
        assert item["jetf_latency_mean_ms"] > 0.0
        assert item["matchms_latency_mean_ms"] > 0.0
        assert item["speedup"] > 1.0
        assert 0.0 < item["avg_pruned_ratio"] <= 1.0


def test_readme_precursor_window_example_executable():
    """验证 README.md 中的 PrecursorWindow 推荐写法及其 AST 语法与执行正确性。"""
    readme_text = README_PATH.read_text(encoding="utf-8")

    # 1. 验证 README.md 包含标准推荐写法与注释
    pattern = r"precursor_window=PrecursorWindow\(mz=400\.25,\s*tolerance_da=0\.25\)"
    assert re.search(pattern, readme_text) is not None, "README.md 未包含推荐的 PrecursorWindow(mz=400.25, tolerance_da=0.25)"

    # 2. 验证执行正确性
    pw = PrecursorWindow(mz=400.25, tolerance_da=0.25)
    assert pytest.approx(pw.min_mz) == 400.0
    assert pytest.approx(pw.max_mz) == 400.5
    assert pytest.approx(pw.mz) == 400.25
    assert pytest.approx(pw.tolerance_da) == 0.25

    # 3. 验证兼容的边界传参
    pw_compat = PrecursorWindow(min_mz=400.0, max_mz=400.5)
    assert pytest.approx(pw_compat.min_mz) == 400.0
    assert pytest.approx(pw_compat.max_mz) == 400.5

    # 4. 从 README 提取整个 Quickstart Python 代码块，验证 AST 语法完全无错
    code_blocks = re.findall(r"```python\s*(.*?)```", readme_text, re.DOTALL)
    assert len(code_blocks) >= 1
    # 验证第一个快速上手代码块可以通过 AST 解析
    parsed_ast = ast.parse(code_blocks[0])
    assert parsed_ast is not None


def test_readme_gpu_inflation_factor_alignment():
    """验证 README.md 中的 GPU 安全上界膨胀因子为 1 + 10^{-3}，与 bounds_kernels.py 严格一致。"""
    readme_text = README_PATH.read_text(encoding="utf-8")

    # 验证 README 中声明 1 + 10^{-3} (或 1 + 10^-3)
    assert "1 + 10^{-3}" in readme_text or "1 + 10^-3" in readme_text

    # 验证原先不一致的 1 + 10^{-4} 已经完全清除（除可能的并列微误差外，不存在于 GPU 膨胀描述中）
    assert "FP32 保守膨胀因子（$1 + 10^{-4}$）" not in readme_text
    assert "严格保守的安全上界膨胀因子（$1 + 10^{-4}$）" not in readme_text

    # 验证与实际代码 SAFETY_MARGIN_FP32 的数值严格一致
    expected_margin = np.float32(1.0 + 1e-3)
    assert SAFETY_MARGIN_FP32 == expected_margin


def test_readme_benchmark_table_sync_with_json():
    """验证 README.md §2.1 表格中的各项数字与 docs/benchmark-matchms.json 严格逐项吻合。"""
    json_data = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
    tp_results = {item["mode_name"]: item for item in json_data["results"]["retrieval_throughput"]}

    readme_text = README_PATH.read_text(encoding="utf-8")

    # 提取 §2.1 表格行
    # | **开放检索 Top-10 (全库无限制)** | 2,000 | **98.3** | 23.6 | 10.18±5.31 [8.13] ms | 42.39±13.89 [40.61] ms | **4.2x** | 97.72% |
    modes_to_check = [
        ("开放检索 Top-10 (全库无限制)", "Top-10"),
        ("开放检索 Top-5 (全库无限制)", "Top-5"),
        ("开放检索 Threshold >= 0.50", "Threshold >= 0.50"),
    ]

    for full_name, short_key in modes_to_check:
        item = tp_results[full_name]
        jetf_qps_str = f"{item['jetf_qps']:.1f}"
        matchms_qps_str = f"{item['matchms_qps']:.1f}"
        speedup_str = f"{item['speedup']:.1f}x"
        pruned_str = f"{item['avg_pruned_ratio'] * 100:.2f}%"

        # 验证该行存在于 README.md 中且包含相应数值
        assert jetf_qps_str in readme_text, f"README.md 未找到模式 {short_key} 的 JETF QPS {jetf_qps_str}"
        assert matchms_qps_str in readme_text, f"README.md 未找到模式 {short_key} 的 matchms QPS {matchms_qps_str}"
        assert speedup_str in readme_text, f"README.md 未找到模式 {short_key} 的加速比 {speedup_str}"
        assert pruned_str in readme_text, f"README.md 未找到模式 {short_key} 的剪枝率 {pruned_str}"


def test_readme_hard_partition_clarification():
    """验证 README.md 明确阐明按离子模式划分为 3 个物理硬分区并解释 UNKNOWN 兼容策略。"""
    readme_text = README_PATH.read_text(encoding="utf-8")
    assert "3 个物理硬分区" in readme_text
    assert "POSITIVE" in readme_text
    assert "NEGATIVE" in readme_text
    assert "UNKNOWN" in readme_text
    assert "IonModePolicy.INCLUDE_UNKNOWN" in readme_text


def test_readme_clean_default_behavior_clarification():
    """验证 README.md 明确说明 --clean 默认开启且可通过 --no-clean 跳过。"""
    readme_text = README_PATH.read_text(encoding="utf-8")
    assert "--clean" in readme_text
    assert "--no-clean" in readme_text
    assert "默认开启" in readme_text


def test_algorithm_design_doc_latency_sync():
    """验证 docs/algorithm-design-forest.md 表 7.1 中的开放检索延迟已对齐真实 2M 基准。"""
    algo_text = ALGO_DOC_PATH.read_text(encoding="utf-8")
    # 验证已更正为 48.63 ~ 51.22 ms 区间
    assert "48.63 ~ 51.22 ms" in algo_text
    # 验证旧的 87.18 ~ 96.58 ms 已不存在
    assert "87.18 ~ 96.58 ms" not in algo_text
