# JET-Forest (JETF)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-184%20passed%20(100%25)-brightgreen.svg)]()

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
   - **离子模式 3 个物理硬分区**：在顶层划分为正离子 (`POSITIVE`)、负离子 (`NEGATIVE`) 以及未知模式 (`UNKNOWN`) 3 个物理硬分区，实现跨模式零交叉物理隔离；同时支持 `IonModePolicy.INCLUDE_UNKNOWN` 策略，允许查询谱在指定模式匹配的同时兼容检索 UNKNOWN 分区。
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
   - 提供完备的坏谱隔离审计机制（`RejectedRecord` 与 `RejectReason`），清晰记录因空峰、非正强度、NaN 异常或清洗过滤而被隔离的谱图。

6. **无锁并发与多核硬件加速 (Lock-Free Concurrency & Multi-Core JIT)**：
   - **查询级多线程无锁并发**：`ForestIndex` 与 `PreprocessedLibrary` 均为只读不可变的紧凑列式结构化 NumPy 数组。`search_forest` 在遍历检索时为每个线程独立维护私有的 `ResultSet` 优先队列与局部状态栈，**零全局可变状态、零写锁争用（Lock-Free）**，天然安全支持 Python `ThreadPoolExecutor` 并发检索多个查询谱。
   - **内核级释放 GIL 与数据并行**：底层 Numba JIT 算子内核（叶上界计算、单谱精确上界 $U_{ind}$、贪心余弦精评）均显式声明 `nogil=True`，在密集数值计算期间主动释放 Python 全局解释器锁，使多线程可真实跑满 CPU 多物理核；批量树根求值内核 `_batch_root_bounds_numba` 还支持 `parallel=True`（OpenMP 树间数据并行）。

7. **GPU 端到端分层加速与纯寄存器保真流水线 (GPU Multi-Layer Acceleration & Pure-Register Filtering)**：
   - **零 C++ 依赖的高性能 CUDA 流水线**：依托 Numba 0.67 原生 CUDA JIT，直接在 Python 解释器内编译与调度核函数，零外部 C++/NVCC 编译器构建负担。
   - **FP32 紧致与保守安全膨胀**：显存内包络与峰表采用连续 FP32 紧致存储，在 K1（根包络）、K2（叶包络）与 K3a（单谱精确上界 $U_{ind}$）计算中应用严格保守的安全上界膨胀因子（$1 + 10^{-3}$，即 1000 ppm / 0.1%），彻底消除单精度浮点截断引起的任何漏检假阴性，**数学保真 100% 零漏检**。
   - **K3a 纯寄存器双指针求交**：在 GPU 端批量执行 $U_{\text{ind}}$ 单谱精确上界估算，利用 GPU 线程级高速纯寄存器执行贪心双指针扫描，零共享内存开销、零全局内存往返访存，毫秒级过滤 $>99.9\%$ 的候选谱图。
   - **Top-K 贪心探测与动态门槛预植入 (Speculative Probe Preheating)**：
     - **设计机理**：在 Top-K 模式初期 $\theta = -\infty$ 时，优先贪心探测顶层前体候选树，精评前 $K$ 个及高置信谱（至多 $\text{probe\_trees} \times \text{tree\_capacity}$ 条），将初始搜索阈值 $\theta$ 从 $-\infty$ 快速抬升至真实下界，换取后续遍历中 $>99.9\%$ 的极致剪枝率；
     - **开销完全透明**：探测阶段实际精评的谱数已如实计入统计指标 `SearchStats.n_scored`，并在 `SearchStats.probe_scored` 中单独透明追踪，使检索成本清晰可审计；
     - **计算零冗余**：被探测的树在 probe 阶段已完成其全部叶节点的评估与剪枝，后续不会重复入堆展开，全局计算零冗余。
   - **零显存浪费驻留管理**：完全剔除无用的 `postings.energy` 列，在 GNPS 百万库规模下直接节省约 40% 峰表显存，近百万级谱图快照仅需约 1 GB 显存驻留。

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
    precursor_window=PrecursorWindow(mz=400.25, tolerance_da=0.25),  # 检索窗口 [400.0, 400.5] Da (亦支持 min_mz=400.0, max_mz=400.5 边界传参)
)
target_outcome = search_forest(query_peaks, loaded_forest, library=None, config=target_config)
print(f"前体靶向命中数: {len(target_outcome.hits)}")

# 6. 多线程无锁并发批量检索 (Official Batch Search API)
# 官方提供 search_forest_batch，自动执行自适应 JIT 线程解耦，消除过度订阅，安全无锁并发
from jetf import search_forest_batch

queries = [library.peaks.spectrum_at(i) for i in range(10)]
batch_results = search_forest_batch(
    queries, loaded_forest, library=None, config=config, concurrency=4
)
print(f"多线程并发检索完成: {len(batch_results)} 条查询")

# 7. 进阶过滤策略 (离子模式策略、最小匹配峰数、自身排除)
from jetf import IonModePolicy

advanced_config = QueryConfig(
    mode=SearchMode.TOP_K,
    k=10,
    ion_mode=library.spectra[0].ion_mode,
    ion_mode_policy=IonModePolicy.INCLUDE_UNKNOWN,  # 允许与库中 UNKNOWN 离子模式谱匹配 (默认)
    min_matched_peaks=3,                           # 至少匹配 3 个碎片峰，过滤偶发孤峰假阳性
    exclude_spectrum_id="CCMSLIB00000001",         # 留一法/泛化评测时排除自身谱图
)

# 8. GPU 端到端加速批量检索 (GPU Accelerated Batch Search)
# 依托 Numba 0.67 原生 CUDA 支持，零 C++ 编译负担，一键迁移至显存
from jetf.gpu import (
    GpuForestIndex,
    search_forest_batch_gpu,
    search_threshold_batch_gpu,
    search_topk_batch_gpu,
)

# 一键上传森林至 GPU 显存 (自动完成 FP32 紧致与 energy 列剔除节省显存)
gpu_forest = GpuForestIndex.from_forest(loaded_forest)

# 执行 GPU 批处理阈值检索 (K1+K2 双层包络 + K3a 纯寄存器 Uind 过滤)
threshold_config = QueryConfig(mode=SearchMode.THRESHOLD, threshold=0.7)
gpu_th_outcomes = search_threshold_batch_gpu(
    queries, gpu_forest, config=threshold_config, batch_size=256
)

# 执行 GPU 批处理 Top-K 检索 (Speculative Probe 预植入 + 批量剪枝)
topk_config = QueryConfig(mode=SearchMode.TOP_K, k=10)
gpu_topk_outcomes = search_topk_batch_gpu(
    queries, gpu_forest, config=topk_config, batch_size=256
)

# 或使用统一调度器 search_forest_batch_gpu 自动按 QueryConfig 模式分流
unified_outcomes = search_forest_batch_gpu(
    queries, gpu_forest, config=topk_config, batch_size=256
)
```

### 2. 命令行工具 (CLI)

```bash
# 从 MGF 文件编译构建索引快照 (--clean 在构建快照时为默认开启，执行 matchms 标准化去噪与 Top-300 峰截断；若输入已预清洗可指定 --no-clean 跳过)
jetf build path/to/library.mgf -o library_forest.npz --clean --clean-max-peaks 300 -f

# 编译大库时可选开启被隔离谱图详细记录审计 (--keep-rejected)
jetf build path/to/library.mgf -o library_forest.npz --clean --keep-rejected -f

# 查看快照元数据与统计信息
jetf info library_forest.npz

# 运行与 matchms 的全景性能与一致性对比基准 (同时导出结构化 JSON 与 CSV)
jetf benchmark --mode all --library-size 2000 --clean -j docs/benchmark-matchms.json -c docs/benchmark-matchms.csv

# 运行 200 万大库脱机多线程高并发吞吐量压测 (指定 --concurrency 4 启用 4 线程并发检索)
jetf benchmark --snapshot all_gnps_forest.npz --mode throughput --concurrency 4 --skip-matchms -j docs/benchmark-2m.json -c docs/benchmark-2m.csv

# 运行 GPU 批量吞吐量基准压测套件 (对比 CPU 1T / CPU MT vs RTX 4060 GPU，导出结构化 JSON)
python -m jetf.benchmarks.gpu_throughput --snapshot cleaned_forest.npz --n-queries 64 --output-json docs/benchmark_gpu_64.json
```

---

## 与 matchms 性能对比 (Benchmark vs matchms)

以代谢组学界工业级参考库 [matchms](https://github.com/matchms/matchms)（`CosineGreedy`, Numba JIT）为基准，在真实 GNPS 质谱库（目标 $N=2,000$，经 matchms 工业级清洗后实际可用 $N=1,991$ 谱，`seed=2026`）及 200 万级全量大库快照（$N=2,003,310$ 谱，快照大小 1.52 GB）上系统评测了**结果一致性**与**检索吞吐量**：

### 1. 结果一致性 (Result Consistency)

- **单对打分等价性**：在 $1,000$ 对真实质谱匹配中，JET-Forest 与 matchms 平均绝对误差 $\text{MAE} = 1.03 \times 10^{-7}$，中位数分差 $0.00$，$95\%$ 谱对分差 $\le 6.94 \times 10^{-18}$，$99\%$ 谱对分差 $\le 2.22 \times 10^{-16}$，匹配峰数吻合率达 **99.90%**。存在 1 对谱（0.1%）出现微小差异（$\text{Max AE} = 1.03 \times 10^{-4}$），源于并列峰贪心匹配时的 tie-breaking 顺序差异。
- **全库检索零漏检**：在排除自配对的严格近邻检索场景下（开放 Top-K 与阈值检索，共 90 组查询），**Recall@K 恒为 100.00%**，漏检总数为 **0**（**Zero False Dismissals** 严格成立）。

### 2. 吞吐量与加速比 (Throughput & Latency)

#### 2.1 2,000 谱库全景对比 (vs matchms 穷举基准)

| 检索场景 | 库容量 ($N$) | JETF QPS | matchms QPS | JETF 时延 (Mean±Std [Med] ms) | matchms 时延 (Mean±Std [Med] ms) | 加速比 (Speedup) | 包络剪枝率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **开放检索 Top-10 (全库无限制)** | 2,000 | **171.0** | 27.7 | 5.85±3.05 [4.75] ms | 36.16±9.61 [37.22] ms | **6.2x** | 97.72% |
| **开放检索 Top-5 (全库无限制)** | 2,000 | **255.6** | 27.2 | 3.91±2.95 [2.99] ms | 36.79±10.44 [38.00] ms | **9.4x** | 98.70% |
| **开放检索 Threshold >= 0.50** | 2,000 | **678.6** | 26.6 | 1.47±1.84 [0.68] ms | 37.56±10.50 [39.38] ms | **25.5x** | 99.80% |

#### 2.2 200 万谱全量 GNPS 库宏观检索吞吐量与并发性能 (2,003,310 谱脱机秒级加载，100 张谱大样本评测)

##### 单谱低延迟模式 (单查询独占多核，concurrency=1)
| 检索场景 | 库容量 ($N$) | JETF QPS | JETF 时延 (Mean±Std [Med] ms) | P95 时延 (ms) | P99 时延 (ms) | 包络剪枝率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **开放检索 Top-10 (全库无限制)** | 2,003,310 | **10.2** | 97.91±137.80 [**51.22**] ms | 370.47 ms | 675.81 ms | **99.98%** |
| **开放检索 Top-5 (全库无限制)** | 2,003,310 | **11.7** | 85.61±113.56 [**48.63**] ms | 331.32 ms | 635.85 ms | **99.98%** |
| **开放检索 Threshold >= 0.50** | 2,003,310 | **2.6** | 388.67±522.91 [**161.31**] ms | 1216.69 ms | 2461.13 ms | **99.76%** |

##### 高并发批量批处理模式 (自适应解耦，concurrency=16)
| 检索场景 | 库容量 ($N$) | JETF QPS | JETF 批处理单谱时延 (Mean [Med] ms) | P95 时延 (ms) | P99 时延 (ms) | 包络剪枝率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **开放检索 Top-10 (全库无限制)** | 2,003,310 | **8.8** | 1446.52 [**484.78**] ms | 7408.05 ms | 9605.59 ms | **99.98%** |
| **开放检索 Top-5 (全库无限制)** | 2,003,310 | **10.6** | 1174.44 [**381.24**] ms | 6377.66 ms | 7801.50 ms | **99.98%** |
| **开放检索 Threshold >= 0.50** | 2,003,310 | **1.6** | 8896.20 [**3915.04**] ms | 32453.53 ms | 48262.86 ms | **99.76%** |

> **基准复现与报告生成**：可通过 CLI 评测套件随时重跑并导出最新结构化 JSON 与 CSV 结果：
> ```bash
> # 运行 200 万全库检索基准测试并生成最新结果
> jetf benchmark --snapshot all_gnps_forest.npz -j docs/benchmark-2m.json -c docs/benchmark-2m.csv --skip-matchms
>
> # 运行与 matchms 严格一致性及对齐评测
> jetf benchmark GNPS-LIBRARY.mgf --mode all --clean -j docs/benchmark-matchms.json -c docs/benchmark-matchms.csv
> ```

---

## GPU 加速批量检索性能评测 (GPU Throughput & Acceleration Benchmark)

基于近百万级真实 GNPS 质谱库快照（`cleaned_forest.npz`，$N=820,482$ 谱，12,821 棵小树，87,167 个节点，快照大小 763.09 MB），在消费级独立显卡环境上系统测试了显存驻留开销、开放式阈值检索与开放式 Top-K 检索的端到端吞吐量与加速比。

### 1. 硬件与软件运行环境

- **GPU 设备**：NVIDIA GeForce RTX 4060 Laptop GPU（8.00 GB GDDR6 显存，Ada Lovelace 架构，Compute Capability 8.9）
- **CPU 设备**：24 核 Intel 处理器（支持 24 线程高并发）
- **软件运行时**：Python 3.10.6，Numba 0.67 原生 CUDA JIT（**零 C++ 依赖、零外部编译构建链**）
- **评测数据集**：真实 GNPS 质谱库快照 `cleaned_forest.npz`（820,482 谱，606,682 正离子谱 + 213,800 负离子谱）

### 2. 显存驻留与 Host-to-Device 迁移统计 (Device Memory Summary)

得益于 FP32 紧致压缩与**完全剔除无用 postings.energy 能量列**的设计，近百万级谱图在 GPU 显存中仅占用 **0.968 GB**（仅占 8GB 总显存的 **12.1%**），可在消费级显卡乃至边缘移动端 GPU 上轻松常驻：

| 显存驻留组件 (VRAM Component) | 占用显存 (MB) | 占用显存 (GB) | 占总显存比例 | 设计优化与说明 |
|:---|:---:|:---:|:---:|:---|
| **Envelopes (双层包络与网格幅值)** | 304.38 MB | 0.297 GB | 30.7% | FP32 紧致压缩，保存 0.02 Da 空间网格幅值 |
| **Trees (森林前体质量与根叶拓扑)** | 0.96 MB | 0.001 GB | 0.1% | 前体二分区间与根/叶节点索引偏移表 |
| **Nodes (BVH-SAH 紧凑节点描述符)** | 2.08 MB | 0.002 GB | 0.2% | 叶节点布尔标志与内部 ID 边界指针 |
| **Postings (连续峰表 mass/intensity/norm)** | 683.41 MB | 0.667 GB | 69.0% | 密集排布，去能量列；强度转换为 FP32 |
| **TOTAL Device Resident (GPU 驻留总计)** | **990.83 MB** | **0.968 GB** | **12.1% (总 VRAM)** | **H2D 搬运耗时 0.32s (3,104 MB/s 吞吐)** |
| **VRAM Energy Omission Savings (显存削减)** | **268.36 MB** | **0.262 GB** | **节省 ~40% 峰表** | 完全剔除 postings.energy 列，零精度损失节省显存 |

### 3. 开放式阈值检索性能基准 (Threshold Search: $\theta = 0.70$)

在不限制前体质量（全库开放式扫描）的苛刻检索场景下，对比 CPU 单线程、CPU 24 线程多核并发与 GPU 批处理在不同 `batch_size` 下的耗时与吞吐（$N=64$ 张真实抽样质谱）：

| 检索后端 / 配置 | 批容量 (Batch) | 耗时 (Wall Time) | 吞吐量 (QPS) | 单查询均值时延 | P50 时延 | P95 时延 | 加速比 vs 1T | 加速比 vs MT | 包络剪枝率 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CPU (1 Thread)** | 1 | 3.650 s | 17.5 | 41.28 ms | 16.58 ms | 167.01 ms | 1.00x | 1.29x | 99.93% |
| **CPU (24 Threads)** | 24 | 4.708 s | 13.6 | 73.56 ms | 539.09 ms | 3084.24 ms | 0.78x | 1.00x | 99.93% |
| **GPU (batch_size=64)** | 64 | 2.236 s | 28.6 | 34.94 ms | 11.38 ms | 60.10 ms | 1.63x | 2.11x | 99.93% |
| **GPU (batch_size=128)** | 128 (eff 64) | 1.488 s | 43.0 | 23.25 ms | 4.39 ms | 52.56 ms | 2.45x | 3.16x | 99.93% |
| **GPU (batch_size=256)** | 256 (eff 64) | **1.469 s** | **43.6** | **22.95 ms** | **4.15 ms** | **52.79 ms** | **2.49x** | **3.21x** | **99.93%** |
| **GPU (batch_size=512)** | 512 (eff 64) | 1.815 s | 35.3 | 28.36 ms | 4.93 ms | 53.56 ms | 2.01x | 2.59x | 99.93% |

> **关键观察**：GPU 批处理通过流水线双缓冲与 K1/K2/K3a 多级剔除，在 `batch_size=256` 时达到 **43.6 QPS**，单查询中位数时延仅 **4.15 ms**，相比 24 线程 CPU 提速 **3.21x**，相比单线程 CPU 提速 **2.49x**，剪枝率高达 **99.93%**。

### 4. 开放式 Top-K 检索性能基准 (Top-K Search: $K = 10$)

在开放式 Top-10 检索中，结合**探针树投机预植入（Speculative Probe Preheating）**机制：贪心探测顶层前体候选树，精评前 $K$ 个及高置信谱（至多 $\text{probe\_trees} \times \text{tree\_capacity}$ 条，其实际精评数已透明记录于 `SearchStats.probe_scored` 并计入 `n_scored`），将初始搜索门槛 $\theta$ 从 $-\infty$ 快速抬升至真实下界。被 probe 的树已完成全部叶节点精评与剪枝，不重复入堆，计算零冗余，换取后续遍历 $>99.9\%$ 的极致剪枝率与全量树根批量短路：

| 检索后端 / 配置 | 批容量 (Batch) | 耗时 (Wall Time) | 吞吐量 (QPS) | 单查询均值时延 | P50 时延 | P95 时延 | 加速比 vs 1T | 加速比 vs MT | 包络剪枝率 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CPU (1 Thread)** | 1 | 2.593 s | 24.7 | 23.71 ms | 7.77 ms | 61.01 ms | 1.00x | 1.11x | 99.97% |
| **CPU (24 Threads)** | 24 | 2.878 s | 22.2 | 44.97 ms | 254.98 ms | 1220.61 ms | 0.90x | 1.00x | 99.97% |
| **GPU (batch_size=64)** | 64 | 3.064 s | 20.9 | 47.87 ms | 16.61 ms | 23.65 ms | 0.85x | 0.94x | 99.95% |
| **GPU (batch_size=128)** | 128 (eff 64) | 2.705 s | 23.7 | 42.27 ms | 10.80 ms | 17.83 ms | 0.96x | 1.06x | 99.95% |
| **GPU (batch_size=256)** | 256 (eff 64) | 2.660 s | 24.1 | 41.57 ms | 9.99 ms | 16.57 ms | 0.97x | 1.08x | 99.95% |
| **GPU (batch_size=512)** | 512 (eff 64) | **2.546 s** | **25.1** | **39.78 ms** | **7.97 ms** | **14.74 ms** | **1.02x** | **1.13x** | **99.95%** |

### 5. 正确性与数学保真核验 (Correctness & Zero False Dismissals)

对 GPU 检索结果与 CPU 逐位基准执行严格对拍（包括召回率、分差绝对误差、命中谱匹配度）：

| 评测检索模式 | 测试查询数 | 零漏检状态 (Zero Miss) | 平均召回率 (Recall@K) | 漏检谱总数 | 最大绝对分差 (Max \|ΔScore\|) | 平均绝对分差 (MAE) | 命中山头完全吻合率 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **THRESHOLD ($\theta=0.7$)** | 64 | **PASS (Zero Miss)** | **100.00%** | **0** | **0.00e+00** | **0.00e+00** | **100.00%** |
| **TOP-K ($K=10$)** | 64 | **PASS (Zero Miss)** | **100.00%** | **0** | **0.00e+00** | **0.00e+00** | **100.00%** |

> **实测结论**：
> 1. **零精度滑坡**：GPU 算子计算结果与 CPU 基准实现的绝对分差 $\le 10^{-12}$（实测为精确 $0.00$），命中外部 ID 与谱图序号 **100.00% 完全对齐**。
> 2. **零漏检成立**：FP32 保守膨胀因子（$1 + 10^{-3}$，即 1000 ppm / 0.1%）数学保真成立，未发生任何因单精度截断导致的漏检（False Dismissals = 0）。

---

## 验证与测试

本项目采用确定性分层抽样基准、穷举基线与 matchms 双重对齐：

```bash
uv run pytest tests/
```

- **全量测试 100% 通过 (184/184 passed)**：
  - `tests/unit/test_review_phase3_doc_and_json_sync.py`：阶段 3 对外文档、CLI 调度与 2,000 谱 benchmark JSON/CSV 基准资产全量对齐与回归校验。
  - `tests/unit/test_review_phase2_benchmark_fixes.py`：阶段 2 真实抽样大库评测、BLINK/matchms 评测基线一致性与自排除过滤回归测试。
  - `tests/unit/test_review_phase1_gpu_fixes.py`：阶段 1 GPU 核函数与流水线并发安全防御、网格边界与零漏检回归测试。
  - `tests/unit/gpu/test_device.py`：GPU 显存驻留数据结构 `GpuForestIndex` 拓扑不变性、异步 CUDA 流搬运与切分子库测试。
  - `tests/unit/gpu/test_bounds_kernels.py`：K1 树根上界与 K2 叶上界 CUDA 算子与 CPU 结果逐位对齐、边界用例及多查询批量测试。
  - `tests/unit/gpu/test_uind_kernel.py`：K3a 纯寄存器 $U_{\text{ind}}$ 密集与稀疏配对核函数在合成数据与 GNPS 子集上的数学正确性与吞吐测试。
  - `tests/e2e/gpu/test_gpu_threshold_search.py`：端到端 GPU 批量阈值检索（$\theta=0.5, 0.7, 0.9$）零漏检与逐位吻合性测试。
  - `tests/e2e/gpu/test_gpu_topk_search.py`：端到端 GPU 批量 Top-K 检索（$K=1, 5, 10$）100% 召回率、穷举对拍与探针预植入校验。
  - `tests/unit/test_adaptive_concurrency.py`：自适应多线程并发解耦校验（`adaptive_numba_threads` 动态调整与严格现场恢复、`search_forest_batch` 批量并发与串行逐项逐位完全一致性）。
  - `tests/unit/test_accelerated_search.py`：二次加速计划深度校验（JIT 叶上界单调二分与 AABB 剪枝数值等价性、树根零上界短路、Top-K 动态门槛 $\theta$ 预植入安全保真性、多线程并发检索结果逐项吻合）。
  - `tests/unit/test_review_p0_p1_fixes.py`：量化偏置极限距离边缘用例零漏检验证（$m_{lib}=100.0-10^{-13}, m_q=99.98-10^{-13}$）、全 0 强度谱建库列等长校验与微块对齐、阈值检索临界分一致性 $[threshold - 10^{-12}, threshold - 10^{-13}]$ 及 0 峰空谱快速短路。
  - `tests/unit/test_review_fixes_20260920.py`：包络上界浮点上偏（A1）、0.02 Da 网格单元边界浮点截断安全（A2）、幂变换边界防御（A3）、MGF 严格定界符与 UTF-8 BOM 处理（D1, D3）、电荷 0 与多电荷解析（D4）、快照谱元数据往返与独立脱机检索（D2）、子抽样去重与 rejected 审计保留（D5）、一致性评测剔除自身与空 GT 召回修正（E1-E4）。
  - `tests/unit/test_benchmark_json.py`：系统软硬件元数据采集、结构化 JSON/CSV 评测结果导出、CLI 参数集成及 Markdown 报告彻底移除验证。
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

