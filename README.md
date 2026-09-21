# JET-Forest (JETF)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-87%20passed%20(100%25)-brightgreen.svg)]()

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
   - 离子模式硬分区：正/负离子模式在顶层物理隔离。
   - 模式内按前体质量严格排序，切分为容量为 64 的浅层小树（树高仅 2~3 层），根包络保持高度稀疏。
   - **全库开放检索 (Open Search)**：结合 SIMD 批量根求值（`batch_root_bounds`）与全局 Best-First 优先队列跨树动态展开，天然支持任意质量偏移与修饰搜索，以高剪枝率实现极致加速。

2. **0.02 Da 空间 BVH-SAH 启发式紧致切分**：
   - 叶节点目标容量 $\le 16$ 谱。
   - 借鉴光线追踪 BVH-SAH（Surface Area Heuristic）代价模型，将 0.02 Da 网格单元的能量分布映射到高维几何空间，最小化包络体积与重叠面积，最大化子包络剪枝效力。

3. **双层安全包络上界体系 (Conservative Envelopes)**：
   - 叶包络：从成员谱逐坐标提取支持集 $Z(B)$ 与最大单峰幅度 $m_h$，单次上偏 $1 + 10^{-12}$。
   - 根包络：由各子叶包络逐网格取 $\max$ 合并，严格保序（不上偏，直接继承子包络上偏）。
   - 三级层次剪枝（根剪枝 $\to$ 叶剪枝 $\to$ 单谱精确上界 $U_{ind}$ 剪枝），对 Greedy Cosine 评分满足数学证明的严格上界安全保真定理（$U \ge \text{Score}$），保证 **100% 零漏检**。

4. **连续紧凑内存布局 (Cache-Friendly Postings)**：
   - 节点所有成员的精评峰按节点内部连续排布，完全规避指针解引用与分散内存访问，大幅提升 SIMD 与缓存命中率。
   - 支持原子 `.npz` 快照持久化，支持即时保存与秒级加载。

5. **标准化 matchms 谱图清洗流水线与多方言支持 (Data Cleaning & Dialect Support)**：
   - 深度集成行业标准 `matchms` 清洗流程，支持元数据标准化、相对强度去噪（如 $\ge 0.1\%$）、高强 Top-N 峰截断（默认 300 峰）及无效谱剔除，削减 50%~70% 的低信噪比长尾峰，大幅压缩内存开销并提升包络稀疏度与剪枝率。
   - 原生兼容 MGF 多方言前体质量（`PEPMASS`、`PRECURSOR_MZ`、`PRECURSORMZ`、`PARENT_MASS`）及外部标识符（`SPECTRUM_ID`、`TITLE` 等），无缝直读各类开源与仪器导出的超大规模真实质谱库。

---

## 安装

推荐使用现代 Python 包管理器 [`uv`](https://github.com/astral-sh/uv)：

```bash
git clone https://github.com/terryji-lab/jetf.git
cd jetf

# 创建独立虚拟环境并安装核心库（包含 matchms 清洗管线）与测试依赖
uv venv --python 3.10
uv pip install -e ".[dev]"

# 若需运行全套可视化基准评测报告导出，安装 benchmark 依赖组：
uv pip install -e ".[dev,benchmark]"
```

---

## 快速上手

### 1. Python API

```python
from pathlib import Path
from jetf import (
    parse_mgf,
    clean_parsed_library,
    MatchmsCleanConfig,
    preprocess_library,
    build_forest_index,
    ForestSpec,
    QueryConfig,
    SearchMode,
    search_forest,
    save_forest_snapshot,
    load_forest_snapshot,
    PrecursorWindow,
)

# 1. 解析与标准化预处理 (支持原生 MGF 多方言字段)
library_path = Path("GNPS-LIBRARY.mgf")
parsed = parse_mgf(library_path)

# 可选：执行 matchms 谱图清洗（过滤弱噪声、截断 Top-300 峰以极致优化内存与剪枝率）
cleaned = clean_parsed_library(parsed, MatchmsCleanConfig(max_peaks=300, min_relative_intensity=0.001))
library = preprocess_library(cleaned)

# 2. 构建森林索引
spec = ForestSpec(tree_capacity=64, leaf_capacity=16, summary_grid_da=0.02)
forest = build_forest_index(library, spec)

# 3. 持久化与加载
save_forest_snapshot(forest, "forest_index.npz")
loaded_forest = load_forest_snapshot("forest_index.npz")

# 4. 执行全库开放检索 (Open Top-K Search，支持独立脱机检索，无需再次提供原始库)
query_peaks = library.peaks.spectrum_at(0)
config = QueryConfig(
    mode=SearchMode.TOP_K,
    k=10,
    ion_mode=library.spectra[0].ion_mode,
)
outcome = search_forest(query_peaks, loaded_forest, library=None, config=config)

print(f"全库开放检索命中数: {len(outcome.hits)}")
for hit in outcome.hits:
    print(f"  Hit: [{hit.external_id}] Spectrum {hit.spectrum_index}, Score = {hit.score:.4f}, Matched Peaks = {hit.n_matched}")

# 5. 执行前体靶向检索 (Targeted Precursor Search，二分过滤仅评估相关树)
target_config = QueryConfig(
    mode=SearchMode.TOP_K,
    k=5,
    ion_mode=library.spectra[0].ion_mode,
    precursor_window=PrecursorWindow(min_mz=400.0, max_mz=400.5),
)
target_outcome = search_forest(query_peaks, loaded_forest, library=None, config=target_config)
print(f"前体靶向命中数: {len(target_outcome.hits)}")
```

### 2. 命令行工具 (CLI)

```bash
# 从 MGF 文件编译构建索引快照 (指定 --clean 可启用 matchms 标准化去噪与峰数截断)
jetf build path/to/library.mgf -o library_forest.npz --clean --clean-max-peaks 300 -f

# 查看快照元数据与统计信息
jetf info library_forest.npz

# 运行与 matchms 的全景性能与一致性对比基准 (同时导出 Markdown 与结构化 JSON)
jetf benchmark --mode all --library-size 2000 --clean -o docs/benchmark-matchms.md -j docs/benchmark-matchms.json
```

---

## 与 matchms 性能对比 (Benchmark vs matchms)

以代谢组学界工业级参考库 [matchms](https://github.com/matchms/matchms)（`CosineGreedy`, Numba JIT）为基准，在真实 GNPS 质谱库（目标 $N=2,000$，经 matchms 工业级清洗后实际可用 $N=1,991$ 谱，`seed=2026`）及 200 万级全量大库快照（$N=2,003,310$ 谱，快照大小 1.52 GB）上系统评测了**结果一致性**与**检索吞吐量**：

### 1. 结果一致性 (Result Consistency)

- **单对打分等价性**：在 $1,000$ 对真实质谱匹配中，JET-Forest 与 matchms 平均绝对误差 $\text{MAE} = 5.93 \times 10^{-7}$，中位数分差 $0.00$，$95\%$ 谱对分差 $\le 6.94 \times 10^{-18}$，$99\%$ 谱对分差 $\le 2.23 \times 10^{-16}$，匹配峰数吻合率达 **99.80%**。存在 3 对谱（0.3%）出现微小差异（$\text{Max AE} = 3.54 \times 10^{-4}$），源于并列峰贪心匹配时的 tie-breaking 顺序差异。
- **全库检索零漏检**：在排除自配对的严格近邻检索场景下（开放 Top-K 与阈值检索，共 90 组查询），**Recall@K 恒为 100.00%**，漏检总数为 **0**（**Zero False Dismissals** 严格成立），最大打分差 $\le 8.88 \times 10^{-16}$。

### 2. 吞吐量与加速比 (Throughput & Latency)

#### 2.1 2,000 谱库全景对比 (vs matchms 穷举基准)

| 检索场景 | 库容量 ($N$) | JETF QPS | matchms QPS | JETF 时延 (Mean±Std [Med] ms) | matchms 时延 (Mean±Std [Med] ms) | 加速比 (Speedup) | 包络剪枝率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **开放检索 Top-10 (全库无限制)** | 1,991 | **62.5** | 28.3 | 16.00±9.05 [14.80] ms | 35.35±11.94 [37.08] ms | **2.2x** | 98.16% |
| **开放检索 Top-5 (全库无限制)** | 1,991 | **159.0** | 28.3 | 6.29±4.08 [5.58] ms | 35.36±12.12 [37.44] ms | **5.6x** | 99.08% |
| **开放检索 Threshold >= 0.50** | 1,991 | **303.2** | 28.6 | 3.30±4.13 [1.82] ms | 34.94±11.68 [36.75] ms | **10.6x** | 99.76% |

#### 2.2 200 万谱全量 GNPS 库宏观检索吞吐量 (2,003,310 谱脱机秒级加载，二次性能跃升版)

| 检索场景 | 库容量 ($N$) | JETF 单核 QPS | JETF 时延 (Mean±Std [Med] ms) | P95 时延 (ms) | 包络剪枝率 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **开放检索 Top-10 (全库无限制)** | 2,003,310 | **7.8** | 128.72±103.43 [**96.58**] ms | 350.81 ms | **99.98%** |
| **开放检索 Top-5 (全库无限制)** | 2,003,310 | **9.3** | 107.36±81.46 [**87.18**] ms | 217.20 ms | **99.98%** |
| **开放检索 Threshold >= 0.50** | 2,003,310 | **1.7** | 591.01±642.97 [**347.24**] ms | 2023.08 ms | **99.74%** |

> **基准复现与报告生成**：可通过 CLI 评测套件随时重跑并导出最新 Markdown 与 JSON 报告：
> ```bash
> # 运行 200 万全库检索基准测试并生成最新报告
> jetf benchmark --snapshot all_gnps_forest.npz -o docs/benchmark-2m.md -j docs/benchmark-2m.json --skip-matchms
>
> # 运行与 matchms 严格一致性及对齐评测
> jetf benchmark GNPS-LIBRARY.mgf --mode all --clean -o docs/benchmark-matchms.md -j docs/benchmark-matchms.json
> ```

---

## 验证与测试

本项目采用确定性分层抽样基准、穷举基线与 matchms 双重对齐：

```bash
uv run pytest tests/
```

- **全量测试 100% 通过 (87/87 passed)**：
  - `tests/unit/test_accelerated_search.py`：二次加速计划深度校验（JIT 叶上界单调二分与 AABB 剪枝数值等价性、树根零上界短路、Top-K 动态门槛 $\theta$ 预植入安全保真性、多线程并发检索结果逐项吻合）。
  - `tests/unit/test_review_p0_p1_fixes.py`：量化偏置极限距离边缘用例零漏检验证（$m_{lib}=100.0-10^{-13}, m_q=99.98-10^{-13}$）、全 0 强度谱建库列等长校验与微块对齐、阈值检索临界分一致性 $[threshold - 10^{-12}, threshold - 10^{-13}]$ 及 0 峰空谱快速短路。
  - `tests/unit/test_review_fixes_20260920.py`：包络上界浮点上偏（A1）、0.02 Da 网格单元边界浮点截断安全（A2）、幂变换边界防御（A3）、MGF 严格定界符与 UTF-8 BOM 处理（D1, D3）、电荷 0 与多电荷解析（D4）、快照谱元数据往返与独立脱机检索（D2）、子抽样去重与 rejected 审计保留（D5）、一致性评测剔除自身与空 GT 召回修正（E1-E4）。
  - `tests/unit/test_benchmark_json.py`：系统软硬件元数据采集、结构化 JSON 报告序列化与 CLI `--output-json` 参数集成自测。
  - `tests/unit/test_cleaning.py`：matchms 标准化谱图清洗流水线、参数边界校验、Top-N 峰截断及紧凑列式拓扑不变量自检。
  - `tests/unit/test_matchms_scoring.py`：与 matchms `CosineGreedy` 的自匹配、不相交及随机重叠数值等价性（误差 $<10^{-10}$）。
  - `tests/e2e/test_matchms_retrieval.py`：全流程对比 matchms 真实候选，实证 100% 召回与零漏检（Zero False Dismissals）。
  - `tests/unit/test_bvh_sah.py`：BVH-SAH 切分有效性、几何退化保护与单轴投影自检。
  - `tests/unit/test_scoring.py`：确定性 Greedy Cosine 与单谱 $U_{ind}$ 严格上界定理校验。
  - `tests/unit/test_structure.py`：森林列式拓扑不变量、包络上界安全性及 `.npz` 往返序列化。
  - `tests/unit/test_search.py`：开放检索 Top-K 与 Threshold 模式与穷举基线逐项逐位 100% 吻合。
  - `tests/e2e/test_exhaustive_consistency.py`：针对极值谱与零门槛全流程端到端穷举一致性验证。
  - `tests/unit/test_batch_root_bounds.py`：向量化批量树根求值与逐个求值的数值严格等价性。
  - `tests/unit/test_regression_review.py`：输入防护（负强度拦截、mass 升序校验、库指纹防错配、L2 归一化断言、MGF `PRECURSOR_MZ`/`PARENT_MASS`/`SPECTRUM_ID` 多方言兼容）回归防护测试。
  - `tests/unit/test_lean_build_and_snapshot.py`：精简构建与快照序列化往返、指纹校验、库-索引一致性断言。
  - `tests/unit/test_uind_numba.py`：单谱 Uind 上界 Numba JIT 与 NumPy 向量化内核等价性。
  - `tests/unit/test_identity_search.py`：身份检索全模式覆盖自测。

---

## 许可证

本项目依据 [Apache-2.0](LICENSE) 许可证开源。
