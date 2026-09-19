# JET-Forest (JETF)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passed%20(100%25)-brightgreen.svg)]()

**JET-Forest**（JETF）是一个独立、高吞吐、零漏检（Zero False Dismissals）的串联质谱（MS/MS）相似度检索引擎。它实现了**前体自适应浅层包络森林与 BVH-SAH 紧致索引**（Precursor-Binned Envelope Forest with SAH Leaf Partitioning）算法体系。

---

## 核心设计与数学原理

传统质谱检索在大规模开放检索或大前体容差检索下，面临着巨大的多级剪枝与缓存开销。JET-Forest 采用前体自适应浅层森林与连续紧凑结构：

```
       [Precursor Sorting / Buckets (Capacity 64)]
                        |
       +----------------+----------------+
       |                                 |
 [Tree 0 (Root)]                   [Tree 1 (Root)]
   /          \                      /          \
[Leaf 0]    [Leaf 1] (SAH <= 16)  [Leaf 2]    [Leaf 3]
```

1. **两级前体分桶 (Precursor Bins)**：
   - 离子模式硬分区：正/负离子模式在顶层隔离。
   - 模式内按前体质量严格排序，切分为容量为 64 的浅层小树（树高仅 2~3 层）。
   - **模式 A（身份检索）**：通过前体窗口二分检索快速定位候选树范围，不触碰无关前体树。
   - **模式 B（开放检索）**：采用全局 Best-First 优先队列跨树展开，天然支持任意质量偏移与修饰搜索。

2. **0.02 Da 空间 BVH-SAH 启发式紧致切分**：
   - 叶节点目标容量 $\le 16$ 谱。
   - 借鉴光线追踪 BVH-SAH（Surface Area Heuristic）代价模型，将 0.02 Da 网格单元的能量分布映射到高维几何空间，最小化包络体积与重叠面积，最大化子包络剪枝效力。

3. **双层安全包络上界体系 (Conservative Envelopes)**：
   - 叶包络：从成员谱逐坐标提取最大单峰幅度 $m_h$ 与最大网格能量 $e_h$，单次上偏 $1 + 10^{-12}$。
   - 根包络：由各子叶包络逐网格取 $\max$ 合并，严格保序。
   - 三级层次剪枝（根剪枝 $\to$ 叶剪枝 $\to$ 单谱精确上界 $U_{ind}$ 剪枝），对 Greedy Cosine 评分满足数学证明的严格上界安全保真定理（$U \ge \text{Score}$），保证 **100% 零漏检**。

4. **连续紧凑内存布局 (Cache-Friendly Postings)**：
   - 节点所有成员的精评峰按节点内部连续排布，完全规避指针解引用与分散内存访问，大幅提升 SIMD 与缓存命中率。
   - 支持原子 `.npz` 快照持久化，支持即时保存与秒级加载。

---

## 安装

推荐使用现代 Python 包管理器 [`uv`](https://github.com/astral-sh/uv)：

```bash
git clone https://github.com/terryji-lab/jetf.git
cd jetf

# 创建独立虚拟环境并安装依赖
uv venv --python 3.10
uv pip install -e ".[dev]"
```

---

## 快速上手

### 1. Python API

```python
from pathlib import Path
from jetf import (
    parse_mgf,
    preprocess_library,
    build_forest_index,
    ForestSpec,
    QueryConfig,
    SearchMode,
    PrecursorWindow,
    search_forest,
    save_forest_snapshot,
    load_forest_snapshot,
)

# 1. 解析与标准化预处理
library_path = Path("GNPS-LIBRARY.mgf")
parsed = parse_mgf(library_path)
library = preprocess_library(parsed)

# 2. 构建森林索引
spec = ForestSpec(tree_capacity=64, leaf_capacity=16, summary_grid_da=0.02)
forest = build_forest_index(library, spec)

# 3. 持久化与加载
save_forest_snapshot(forest, "forest_index.npz")
loaded_forest = load_forest_snapshot("forest_index.npz")

# 4. 执行身份检索 (Identity Search with Precursor Window)
query_peaks = library.peaks.spectrum_at(0)
config = QueryConfig(
    mode=SearchMode.TOP_K,
    k=10,
    precursor_window=PrecursorWindow(mz=library.spectra[0].precursor_mz, tolerance_da=0.5),
)
outcome = search_forest(query_peaks, loaded_forest, library, config)

print(f"检索命中数: {len(outcome.hits)}")
for hit in outcome.hits:
    print(f"  Hit: Spectrum {hit.spectrum_index}, Score = {hit.score:.4f}")
```

### 2. 命令行工具 (CLI)

```bash
# 从 MGF 文件编译构建索引快照
jetf build path/to/library.mgf -o library_forest.npz --tree-capacity 64 --leaf-capacity 16

# 查看快照元数据与统计信息
jetf info library_forest.npz
```

---

## 验证与测试

本项目采用确定性分层抽样基准与穷举基线对齐：

```bash
uv run pytest tests/
```

- **全量测试 100% 通过 (17/17 passed)**：
  - `tests/unit/test_bvh_sah.py`：BVH-SAH 切分有效性、几何退化保护与单轴投影自检。
  - `tests/unit/test_scoring.py`：确定性 Greedy Cosine 与单谱 $U_{ind}$ 严格上界定理校验。
  - `tests/unit/test_structure.py`：森林列式拓扑不变量、包络上界安全性及 `.npz` 往返序列化。
  - `tests/unit/test_search.py`：身份检索与开放检索（双模态）与穷举基线逐项逐位 100% 吻合。
  - `tests/e2e/test_exhaustive_consistency.py`：全流程端到端穷举一致性验证。

---

## 许可证

本项目依据 [Apache-2.0](LICENSE) 许可证开源。
