> [!NOTE]
> **归档与修复决议说明 (2026-09-21)**：
> 本审查报告中指出的 16 个问题（P0×1、P1×3、P2×4、P3×8）已全部完成修复：
> - **P0**（负强度查询漏检）：已通过三重防线修复（构造层拒绝 + 预处理截断 + 检索入口校验），见 `tests/unit/test_regression_review.py`。
> - **P1-1**（库谱 mass 有序性）：已在 `scoring.py` 入口添加升序断言。
> - **P1-2**（快照/库错配）：已在 `search.py` 入口添加谱数与指纹比对。
> - **P1-3**（未归一化查询）：已在 `types.py` `validate_query` 中添加 L2 范数校验。
> - **P2-1**（批量根求值未实现）：已实现 `batch_root_bounds`（Numba 并行 + NumPy 降级），见 `bounds.py:297-360`。
> - **P2-2**（SAH 构建慢）：已改为前缀/后缀扫描增量计算，见 `bvh_sah.py:100-125`。
> - **P2-4**（叶容量目标未达成）：已调整 `min_leaf = max(8, target_leaf_size // 2)`，候选位收窄回 `[half-4, half, half+4]`。
> - **P3**（死代码与冗余）：已清理 `max_cell_energy` 死列、双套离子模式解析器统一、双网格配置校验、零引用依赖移除等。
> - 本报告已按既定工程规范正式归档至 `docs/archive/`。

# JET-Forest 实现对照审查报告

> **审查对象**：`src/jetf/` 全部实现（3,652 行，14 个模块）与 `tests/`（5 个测试文件）
> **对照基准**：`docs/algorithm-design-forest.md`（前体自适应浅层包络森林与 BVH-SAH 紧致索引设计规格书）
> **审查日期**：2026-09-20
> **数据规模**：GNPS-LIBRARY.mgf 全库 16,106 谱 / 7,441,781 峰
> **代码版本**：`64a8f52`（含未提交文件 `.zcodeignore`）

---

## 目录

1. [结论摘要](#一结论摘要)
2. [审查方法](#二审查方法)
3. [P0：静默漏检缺陷](#三p0静默漏检缺陷-负强度查询)
4. [P1：静默错误风险](#四p1静默错误风险)
5. [P2：性能与规模实测](#五p2性能与规模实测)
6. [P3：死代码、冗余与工程细节](#六p3死代码冗余与工程细节)
7. [已实证的可靠部分](#七已实证的可靠部分)
8. [未覆盖范围](#八未覆盖范围)
9. [修复优先级建议](#九修复优先级建议)

---

## 一、结论摘要

**文档的三条铁律在代码里真实落地，不是纸面设计。** 在全库 16,106 谱上执行 24 组随机查询（身份检索 12 组 + 开放检索 12 组），森林检索的命中集合与 `search_exhaustive` 穷举基线**逐条一致、零漏检**；浮点余量纪律（叶层上偏一次、根层纯 max 合并不重复上偏）经数值验证，根-叶逐坐标相对偏差恰好为 `0.000e+00`。

本次审查发现：

| 等级 | 数量 | 概要 |
| :--- | :---: | :--- |
| **P0** | 1 | 负强度查询导致剪枝上界失效，产生**可复现的静默漏检**（违反 §8.1 铁律） |
| **P1** | 3 | 静默错误风险：库谱 mass 有序性未校验、快照/库错配无校验、未归一化查询被接受 |
| **P2** | 4 | 性能与规模：构建 105.7 s（83% 在 `sah_cost`）、文档承诺的批量根求值未实现、叶容量目标未达成、存储实测超文档声称 1.4~1.9 倍 |
| **P3** | 8 | 死代码与冗余、双套网格配置不校验、双套离子模式解析器、测试强度低于文档铁律 |

---

## 二、审查方法

| 手段 | 内容 |
| :--- | :--- |
| 测试基线 | `pytest tests/ -q` → 17 passed（209 s，其中大部分为 142 MB MGF 解析） |
| 全量源码精读 | 14 个源模块 + 5 个测试文件，逐行对照文档 §3~§8 |
| 合成库对抗探针 | 27 种配置组合：NaN 前体、零能量谱、负离子/未知离子分区、`min_matched_peaks=0`、`k>N`、`threshold=0.0`、`exclude_spectrum_id`、三种离子模式策略 |
| 全库规模探针 | 全库构建、快照往返、24 组查询的剪枝率与一致性、内存与耗时分解 |
| 包络安全性抽查 | 全库 1,547 个叶节点各取 1 条成员作查询，逐 (查询, 成员) 对比叶界/根界/U_ind 与真实分数 |
| 构建热点剖析 | `cProfile` + `pstats`（2000 谱分层子集） |
| 静默假设验证 | 针对 mass 有序性、未归一化查询、网格错配、快照错配分别构造最小复现 |

---

## 三、P0：静默漏检缺陷 —— 负强度查询

### 3.1 问题

文档 §3.3 定义的节点峰上界 $U_{peak}(B) = \sum_i u_i \cdot \max_{h \in E_B(i)} m_h(B)$，其成为真实分数上界的关键前提是查询侧 $u_i \ge 0$：

$$S(Q,P) = \sum_{(i,j) \in M} u_i v_j \le \sum_i u_i \max_{j \in \text{legal}(i)} v_j$$

右侧推导要求匹配边集合 $M$ 中每个 $i$ 至多用一次**且 $u_i \ge 0$**。当 $u_i < 0$ 时，求和不再是上界，剪枝会切掉真实命中。

该前提在代码里**既不校验、也无文档说明**，并且被两条并行逻辑同时破坏：

| 位置 | 行为 | 后果 |
| :--- | :--- | :--- |
| `src/jetf/scoring.py:127-131` | `positive = weights > 0.0` 过滤候选边 | 负强度查询峰**不参与匹配**（等价于被忽略） |
| `src/jetf/bounds.py:143` | `sum_float64(context.intensity * per_peak)` | 负 $u_i$ **反向拉低**上界 |
| `src/jetf/scoring.py:174` | `single_spectrum_bound` 同样线性加权 | $U_{ind}$ 一并失效 |
| `src/jetf/search.py:105` / `:120` / `:147` | `U < theta` 剪枝判据 | 上界失效直接转为漏检 |

### 3.2 防线缺口

- `parse_mgf` 只在**库侧**拦截负强度（`_reject_illegal_peaks`，`src/jetf/mgf.py:341`），查询侧无任何检查。
- `preprocess_query`（`src/jetf/preprocessing.py:117`）的 L2 归一化**保留符号**，因此"官方"查询预处理路径同样不设防。
- 质谱数据中负强度并不罕见（部分厂商把基线噪声写成小负值），而失效只发生在剪枝路径上，穷举基线不会暴露问题。

### 3.3 最小复现

```python
import numpy as np
from jetf import (IonMode, SpectrumMeta, SourceRef, QueryConfig, SearchMode,
                  SpectrumPeaks, build_forest_index, preprocess_library,
                  search_exhaustive, search_forest)
from jetf.mgf import ParsedLibrary

# 50 条库谱，每条在 100/200 处有峰（100 处很强）
metas, mass, inten, pid, offs = [], [], [], [], [0]
rng = np.random.default_rng(5)
for i in range(50):
    peaks = [(100.0, rng.uniform(500, 1000)), (200.0, rng.uniform(1, 20)), (300.0 + i * 0.5, 30.0)]
    metas.append(SpectrumMeta(external_id=f"L{i}", precursor_mz=500.0, charge=1,
                              ion_mode=IonMode.POSITIVE, source=SourceRef("m", i), raw_metadata={}))
    m = np.array([p[0] for p in peaks]); it = np.array([p[1] for p in peaks])
    mass.append(m); inten.append(it); pid.append(np.arange(3, dtype=np.int64)); offs.append(offs[-1] + 3)

parsed = ParsedLibrary(source_path="m", spectra=tuple(metas), mass=np.concatenate(mass),
                       intensity=np.concatenate(inten), peak_id=np.concatenate(pid),
                       spectrum_offsets=np.array(offs, dtype=np.int64))
library = preprocess_library(parsed)
forest = build_forest_index(library)

# 查询：100 处强度为负
raw = np.array([-50.0, 40.0, 10.0]); n = float(np.sqrt((raw ** 2).sum()))
q = SpectrumPeaks(mass=np.array([100.0, 200.0, 300.0]), intensity=raw / n,
                  energy=(raw / n) ** 2, peak_id=np.arange(3, dtype=np.int64), norm=n)

cfg = QueryConfig(mode=SearchMode.TOP_K, k=5, ion_mode=IonMode.POSITIVE)
rf = search_forest(q, forest, library, cfg)
re_ = search_exhaustive(q, library, cfg)
print("森林:", [(h.external_id, round(h.score, 6)) for h in rf.hits])
print("穷举:", [(h.external_id, round(h.score, 6)) for h in re_.hits])
```

实测输出（两集合完全不相交）：

```
森林: [('L19', 0.015307), ('L20', 0.013773), ('L16', 0.007627), ('L17', 0.00735), ('L18', 0.00479)]
穷举: [('L4', 0.023467), ('L24', 0.022736), ('L25', 0.019348), ('L13', 0.018088), ('L26', 0.017186)]
逐位一致: False
剪枝统计: {'roots_pruned': 0, 'leaves_pruned': 3, 'uind_pruned': 6}

该树根 U_peak = -0.742565  <  成员真实得分 0.016300     <- 上界失效
preprocess_query 后强度: [-0.771517  0.617213  0.154303]  <- 负号被保留
```

### 3.4 修复建议

1. 在 `preprocess_query` 与 `search_forest` 入口将负强度截断为 0（`np.maximum(intensity, 0.0)`），或在 `SpectrumPeaks` 层面拒绝负强度并给出明确错误。
2. 文档 §3.3 补充说明 $u_i \ge 0$ 是定理前提；§8.2 的"正确性纪律"增加该不变量的校验要求。
3. 回归测试建议直接使用 §3.3 的复现用例作为夹具。

---

## 四、P1：静默错误风险

三条都属于"输入不合规但无校验、无文档说明"，共同特征是**静默**：不报错、不告警，只给出错误结果。

### 4.1 `score_greedy_cosine` / `single_spectrum_bound` 隐含要求库谱 mass 升序

`src/jetf/scoring.py:77-78` 对 `library.mass` 执行 `np.searchsorted` 枚举合法匹配边，这要求库谱峰按质量升序排列，但函数没有校验、docstring 与文档都没有写明该前提。实验（同一谱，仅调整行序）：

| 库谱行序 | score | n_matched | U_ind |
| :--- | ---: | ---: | ---: |
| mass 升序 | 1.000000 | 3 | 1.000000000001 |
| 同一谱乱序 | 0.666667 | 2 | 0.666667（**已低于真实分数**） |

检索内部数据来自 `parse_mgf`（每条记录都经 `np.lexsort((peak_id, mass))` 排序），因此索引路径安全；但这两个函数通过 `jetf/__init__.py` 导出为公共 API，用户传入自建 `SpectrumPeaks` 会拿到静默错误的结果，且上界可能低于真实分数。

**修复建议**：入口加单调性断言（`np.all(np.diff(mass) >= 0)`）或内部排序；至少在 docstring 中声明前提。

### 4.2 快照与库错配无任何校验

分数取自 `forest.postings.spectrum_at(iid)`（`src/jetf/search.py:139`），标签取自 `library.spectra[row]`（`src/jetf/search.py:130`），两者的一致性从不验证。实验（A 库快照 + B 库检索，峰位整体平移 100 Da）：

```
错配 top3 (external_id, score): [('B0', 0.5526), ('B1', 0.1558), ('B2', 0.1538)]
真值 top3 (external_id, score): [('B0', 1.0000), ('B1', 0.1039), ('B2', 0.1025)]
complete = True, 无任何告警
```

即：**分数取自 A 的峰数据、标签取自 B 的元数据**，结果静默错误。若两者谱数不同，则视大小关系静默错配或抛 `IndexError`；若快照只覆盖部分库，检索会**谎报 `complete=True`**。

现成资源未被使用：`forest.n_spectra` 已存入快照（`src/jetf/serialization.py:43`），`check_forest_index` 也已实现（`src/jetf/structure.py:288`），但 `load_forest_snapshot` 与 `search_forest` 都没有调用或断言。

**修复建议**：`ForestSpec` 或快照中增加库指纹（`source_path` + 峰总数 + external_id 集合哈希），`search_forest` 入口断言 `forest.n_spectra == library.n_spectra` 且指纹一致。

### 4.3 未归一化查询被静默接受

`search_forest` 假定传入的 `SpectrumPeaks` 已完成 L2 归一化（文档 §3.1 要求 $\sum_j v_j^2 = 1$），但既不校验也不归一化。实验（查询强度整体 ×1000）：

```
原始强度查询: 森林 top1 得分 = 105.1255, 穷举 top1 = 105.1255
森林/穷举命中集合一致: True
```

剪枝一致性未破坏（上界与分数同步线性缩放），但**报出的分数已不是 Cosine 语义**（105 ≫ 1），且 `preprocess_query` 是公开 API 却不在检索路径上调用，用户很容易漏掉这一步。

**修复建议**：`search_forest` 内部调用 `preprocess_query`，或校验 `abs(np.linalg.norm(query.intensity) - 1.0) < 1e-9` 并给出明确错误。

---

## 五、P2：性能与规模实测

### 5.1 全库实测数据 vs 文档声称

规模与文档完全吻合：**16,106 谱 / 7,441,781 峰 / 251 棵树**（文档 §6.2、§7.2 均以该规模举例）。

| 项目 | 文档声称 | 实测 | 判定 |
| :--- | :--- | :--- | :---: |
| 解析 + 预处理 | — | 4.1 s + 0.7 s | ✓ |
| 森林构建 | — | **105.7 s** | 见 5.2 |
| 包络内存 | 150 MB（§7.2） | 213.7 MB | ✗ 1.4× |
| 微块 posting 缓冲 | 120 MB（§7.2） | 227.3 MB | ✗ 1.9× |
| 索引总常驻 | < 300 MB（§7.2） | **441 MB** | ✗ |
| 构建期峰值内存 | 未提及 | **2,462 MB** | 见 5.6 |
| 快照文件 | — | 170.2 MB | — |
| 快照保存 / 加载 | — | 11.3 s / **1.3 s** | ✓ "秒级加载"成立 |
| 每树叶子数 | 4（§1.2 "Root 64 → Leaves 16"） | **6.16**（叶容量 4/10.3/16） | ✗ 见 5.4 |
| 节点总数 | ≈1,250（§7.2） | 1,798 | ✗ 1.4× |
| 开放检索根剪枝率 | 70%~86%（§6.2） | **58.8%**（147.5/251） | 略低 |
| 开放检索叶剪枝率 | 75%~90%（§1.2） | **96.8%**（对已求值叶） | ✓ 超出 |
| 精评调用削减 | 50%~64%（§7.1） | **99.58%**（开放）/ **99.89%**（身份） | ✓ 远超 |
| 身份检索候选树 | "仅留 1~2 棵"（§6.1） | **1.50 / 251 = 0.60%** | ✓ |
| 身份检索根层整树排除 | §6.1 流程图核心步骤 | **恒为 0 次/query** | ✗ 见 5.5 |

微块 227.3 MB 是必然结果：7,441,781 峰 × 4 列 × 8 字节 = 238 MB。文档 §7.2 的 120 MB 隐含 16 字节/峰（float32 量级），与 `src/jetf/types.py:15-19` 全 float64/int64 的实现不符。

### 5.2 构建耗时 105.7 s，83% 花在 `sah_cost`

`cProfile`（2000 谱分层子集，绝对时间被插桩放大，关注占比）：

```
8155462 function calls in 37.281 seconds
  ncalls  tottime  cumtime  filename:lineno(function)
       1    0.037   37.281  builder.py:145(build_forest_index)
  346/32    2.817   35.422  bvh_sah.py:31(split_sah_bvh)        <- 95%
   20900   21.566   31.045  bvh_sah.py:8(sah_cost)              <- 83%
   20900    8.635    8.635  {method 'union' of 'set' objects}
       1    1.533    1.534  builder.py:86(_extract_spec_cells_and_amps)   <- 4%
```

`src/jetf/bvh_sah.py:8` 的 `sah_cost` 对**每个候选 (轴, 切分位)** 都从头执行 `set().union(*cells_list)` 并重建 max 幅度字典（`src/jetf/bvh_sah.py:87-99`）。2000 谱触发 20,900 次调用，全库约 17 万次；单桶 64 谱 ≈ 10 轴 × 7 位置 × 2 侧 = 140 次全量重算。

这是文档 §4.2 伪码的直接翻译，属于**算法本身的复杂度问题**。经典 BVH-SAH 做法是每个候选轴排序一次后做前缀扫描增量维护 union/max，把 $O(\text{位置数} \times n \times \text{cells})$ 降到 $O(n \times \text{cells} + \text{位置数} \times \text{cells})$，输出不变，预计 5~10× 提速。

### 5.3 文档承诺的"批量根求值"未实现，且这是当前最大瓶颈

文档 §6.2 第 1 步要求 **"批量根求值（SIMD Batch Root Evaluation）"**，代码实现是 Python for 循环逐根调用（`src/jetf/search.py:177-192`）。实测单次节点包络求值成本：

| 操作 | 实测耗时 |
| :--- | ---: |
| 根包络 `build_query_context` + `peak_bound`（8,920 cells / 218 查询峰） | **49.4 µs** |
| 叶包络同操作（1,118 cells） | 40.8 µs |
| `single_spectrum_bound` 单谱上界 | 25.3 µs |
| `score_greedy_cosine` 完整精评 | 28.2 µs |

**一次根包络求值比一次完整精评还贵 1.8 倍。** 251 个根求值 = **17.6 ms，占开放检索 100% 的查询耗时**（同查询精评仅 0.2 ms）。根因是固定开销主导：每次求值约 10~12 个 NumPy 小数组调用，而 `build_query_context` 中的 `window_cells`（`src/jetf/bounds.py:98`）**只依赖查询与网格、与节点无关，却被重算 251 次以上**。

可兑现的优化（按收益排序）：
1. 把查询侧的 `cell_lower` / `cell_upper` 提到节点循环之外，只传一次。
2. 把全部根的支持集拼成一个大扁平数组，用一次 `np.searchsorted` 完成批量求值（即文档承诺的 SIMD 批量根求值）。
3. 叶层同理：身份检索每查询约 9 次叶求值 × 41 µs，也有可观节省。

### 5.4 叶容量目标未达成（6.16 叶/树 vs 设计 4 叶/树）

实测每树叶子数 `mean=6.16, min=3, max=8`，叶容量 `4/10.3/16`（设计值 16），节点总数 1,798（设计约 1,255）。根因在 `src/jetf/bvh_sah.py:68` 与 `:80-85`：

- 候选切分位比文档 §4.2 的 `[half-4, half, half+4]` **更宽**：代码为 `half ± {2, 4, 6}`；
- `min_leaf = max(4, leaf_capacity // 4) = 4`，允许切到 4 条谱的极小叶。

SAH 于是一路挑不均衡切分（64 → 26/38 → 9/17 → 8/9 …），切出更多更小的叶。后果：节点与微块数 +43%，包络内存与每查询叶层求值次数同步上浮，叶层剪枝收益被摊薄。

**修复建议**：候选集收窄回 ±4，或对不均衡加惩罚项；`min_leaf` 提到 `leaf_capacity // 2`。注意 `tests/unit/test_structure.py:62` 的断言是 `4 <= count <= 24`（较宽松），修改后仍应通过。

### 5.5 身份检索下根包络剪枝恒为 0（文档叙事与实测不符）

实测（±0.5 Da 身份检索，12 组查询平均）：

```
候选树 1.50 / 251 (=森林 0.60%)
根层整树排除 0.00/query
叶求值 9.08 个, 整叶跳过 0.58 -> 叶剪枝率 6.4%
U_ind 拦截 4.8, 精评 18.1/query, 命中 7.6/query
精评削减 99.89%
```

前体二分已把候选压缩到 1.50 棵树，根包络求值永远满足 $U \ge \theta$（**实测 0 次整树排除**），真正的拦截来自 U_ind。文档 §6.1 的流程图把 `U_peak(Root) < θ → 整树排除` 画成身份检索的核心步骤，实际起作用的是**前体二分 + U_ind**；根/叶包络的价值集中在**开放检索**（根 58.8% / 叶 96.8%）。该结论的耗时口径与"包络层在身份检索下是纯开销"的量化见 §5.6。

**修复建议**：文档按"哪一层服务哪个模态"重写 §1.2、§6.1、§6.2；§7.1 的"精评减少 50%~64%"应更新为实测的 99.6%~99.9%。

### 5.6 吞吐量分解：加速比的口径与"决策层"常数

单查询实测（全库 16,106 谱，8 核机器单线程）：

| 口径 | 身份检索 ±0.5 Da | 开放检索 θ=0.30 |
| :--- | :--- | :--- |
| 森林单查询 | 1.4 ~ 2.1 ms | 17.7 ~ 50 ms |
| 单核吞吐 | ~480 ~ 710 q/s | ~20 ~ 56 q/s |
| 穷举基线单查询 | 11.8 ms | 283.2 ms |
| 穷举基线**精评条数** | **6 / 16,106** | 13,761 / 16,106 |
| 加速比 | 5 ~ 8× | 6 ~ 16× |
| 森林精评条数 | 6 | **4 / 13,761 = 99.97% 削减** |
| 森林耗时构成 | 上界层 0.9 ms + 精评 0.3 ms | **上界层 17.0 ms** + 精评 0.3 ms |

**口径校正（重要）**：身份检索下穷举基线 11.8 ms 中只有 6 条谱被真正评分（`is_eligible` 的前体窗口过滤在评分前就跳过了其余全部），其余约 11.6 ms 全是 Python 元数据扫描开销（实测纯元数据扫描 16,106 条需 4.1 ms，`is_eligible` 的三层函数调用约 0.73 µs/条）。因此"5.1× 加速"是**实现与实现之间**的数字，而非算法级数字；不宜作为架构优势的对外宣称。

**结构性结论：吞吐不由剪枝率决定，而由"决策层"的常数决定。**

1. **决策成本 ≈ 所省工作的数十倍。** 开放检索 17.7 ms 中，17.0 ms 用于判断该不该精评，只有 0.3 ms 是真正的精评；即剪枝决策花了所避免工作的约 57 倍代价。
2. **U_ind 的边际收益接近零。** `single_spectrum_bound` 实测 25.3 µs，而它守护的 `score_greedy_cosine` 只有 28.2 µs —— 过滤器的成本是它所避免工作的 90%。它只在剪掉大比例候选时才划算：开放检索剪掉 615.7 条/query（值得），身份检索只剪掉 4.8/(18.1+4.8) ≈ 21%（几乎白干）。
3. **身份检索的包络层基本是纯开销。** 该实例中根/叶上界求值花 0.9 ms，却一次都没剪掉东西；真正把候选从 64 条压到 6 条的是**前体窗口的元数据过滤**，不是包络。12 组查询平均叶剪枝率 6.4%，其可省上限（被剪叶的成员数 × 28 µs）远低于 0.37 ms 的叶层求值成本。

**可兑现的优化及预期量级**（按收益排序）：

| 优化 | 依据 | 预期 |
| :--- | :--- | :--- |
| 批量根求值（文档 §6.2 承诺未实现） | 251 次独立调用 × 49.4 µs = 17.6 ms；批量后的下界是一次顺序流读 `cell_index` 32.4 MB ≈ 1.5~2 ms | 开放检索 17~50 ms → 3~8 ms |
| `cell_index` 改用 int32 | 根扫描数据量 32.4 MB → 16.2 MB | 与上条叠加 |
| 身份检索关掉或限流 U_ind | U_ind 25.3 µs vs 精评 28.2 µs，仅剪 21% | 身份检索约 -0.1~0.3 ms/query |
| 去掉 $e_h$ 死列 | 见 §六 P3-1 | 常驻 441 → ~371 MB（查询带宽不变，构建/加载/快照受益） |
| SAH 前缀扫描增量代价 | 见 §5.2 | 构建 105.7 s → 十几秒 |

**规模外推**：开放检索的根扫描是 $O(M)$ 且**每次查询都要重扫全部根**，与 θ、与命中数无关。M 随库线性增长：

| 库规模 | 树数 M | 开放检索根扫描 |
| :--- | ---: | ---: |
| 16,106 谱 | 251 | 17.6 ms |
| 160,000 谱（外推） | ~2,500 | ~176 ms |

这是当前唯一的可扩展性隐患——剪枝已经把精评砍掉 99.97%，但查询成本仍随库规模线性增长。要打断这个线性关系，需要比"两层森林"更粗的顶层摘要（第三层）或批量化，而不是继续调剪枝阈值。

### 5.7 规模外推提醒

`jetf build` 全库约 2 分钟、峰值内存 2.4 GB。若按同样路径处理 4.5 GB 的 `ALL_GNPS.mgf`，构建时间与内存将线性以上增长（SAH 中间结构是 Python `set`/`dict`，内存密度低），而文档 §7.2 的存储结论只对 16k 规模的 GNPS-LIBRARY 成立。

---

## 六、P3：死代码、冗余与工程细节

| # | 项 | 位置 | 说明 |
| :---: | :--- | :--- | :--- |
| 1 | `max_cell_energy`（$e_h$）是完整死列 | `src/jetf/bounds.py:29`、`builder.py:372`、`serialization.py:66` | 构建、校验、序列化、快照往返全都有，**没有任何上界读它**（文档 §3.2 将其定义为包络第三列）。占约 1/3 包络内存（≈70 MB）。要么按 §6 的 "Block-Max 跳跃" 用起来，要么删除 |
| 2 | `QueryContext.support_energy` | `src/jetf/bounds.py:91,123` | 计算后从未被读取 |
| 3 | `SearchStats.seed_count` / `skipped_block_intervals` | `src/jetf/results.py:105-106` | 恒为 `0` / `()`，前代 JET-MS 残留 |
| 4 | 包络偏移数组两份拷贝 | `src/jetf/structure.py:166` 与 `:191` | `ForestNodes.envelope_offsets` 与 `ForestEnvelopes.node_envelope_offsets` 是同一数组，两份都进快照，只有后者被 `envelope_of` 读取 |
| 5 | 双套网格配置互不校验 | `src/jetf/preprocessing.py:159` vs `builder.py:150` | `PreprocessSpec.grid_da` 与 `ForestSpec.summary_grid_da` 错配（0.05 vs 0.02）时构建不报错，$e_h$ 列被静默算错（0.05 网格的 cell 下标散列到 0.02 网格 support 上，实测全部落到 slot 0，总和 2.000 → 1.998）。今天无害（$e_h$ 无用），一旦启用 $e_h$ 即成静默错误。建议 `build_forest_index` 入口断言两者相等 |
| 6 | 双套离子模式解析器 | `src/jetf/types.py:33` vs `src/jetf/mgf.py:256` | `IonMode.from_str` 是死代码且与实际使用的 `_parse_ion_mode` **词表分歧**（前者接受 `pos`/`neg`/`+`/`1`，后者只认 `positive`/`negative`）。实测 GNPS-LIBRARY.mgf 只有 13,761 positive + 2,345 negative，当前无影响；但两套并存意味着切换任一方都会改变硬分区归属，进而改变 EXACT 策略召回。建议删除其一或统一到 `from_str` |
| 7 | 声明但零引用的依赖 | `pyproject.toml:13,19` | `pydantic>=2.0`（运行时）与 `hypothesis`（dev）全代码库零引用（已 grep 确认） |
| 8 | 热循环内的对象构造开销 | `src/jetf/structure.py:211`、`:244` | `envelope_of` 每次构造 `NodeEnvelope` 跑 3 次 `check_column`；`postings.spectrum_at` 每次构造 `SpectrumPeaks` 跑 4 次。在"节点求值已是瓶颈"（见 5.3）的前提下值得省 |

补充说明：

- `time.perf_counter()` 实测 55 ns/次（Windows QPC），每节点 2 次插桩对 28~49 µs 的求值开销可忽略，**这一条经实测可排除**，无需优化。
- **测试强度低于文档铁律**：`tests/e2e/test_exhaustive_consistency.py:75` 用 `pytest.approx(abs=1e-12)`，而文档 §8.1 要求"逐位相同（Bit-exact）"。由于两条路径都调用同一个 `score_greedy_cosine` 处理同一批 float64 值，`==` 实际成立；建议改成精确相等断言，让铁律真正被钉住。
- **"100% test coverage" 无法自证**：提交信息与 README 徽章（"17 passed (100%)"）指的是**通过率**；仓库内无覆盖率工具依赖、无配置、无 CI。
- **CLI 静默覆盖输出**：`jetf build -o <已存在文件>` 会无提示覆盖。

---

## 七、已实证的可靠部分

以下不是纸面结论，而是本次审查实测通过的项，建议在后续重构中作为回归基线保留：

1. **定理 1 的包含链逐条落地**：根包络是子叶的**严格逐坐标 max**（相对偏差 `0.000e+00`，确认没有重复上偏）；对全库 1,547 个叶节点各取 1 条成员作查询、共 200 对 (查询, 成员) 抽查，叶界/根界/U_ind **零漏检**。
2. **浮点余量纪律**：`inflate_upper_bounds` 只在叶层调用一次（`src/jetf/builder.py:115-126`），根层纯 max 合并（`src/jetf/builder.py:129-142`）；剪枝判据严格 `U < θ`，相等时不剪（`src/jetf/search.py:105,120,147`）。
3. **top-k 堆语义正确**：`_HeapItem.__lt__` 反向比较 + `(score, external_id, spectrum_index)` 全序破并列，推演确认最终堆内容**与插入顺序无关**，这正是双路径结果可比的前提。
4. **结构不变量完整**：`check_forest_index` 校验叶区间连续并集覆盖根区间、`internal_to_row`/`row_to_internal` 互反、零能量侧车 + 森林成员 = 全谱数（`src/jetf/structure.py:288-338`）；另经独立验证，分区三轴 (trees, nodes, ids) tiling 无缝隙。
5. **确定性**：SAH 候选轴的并列破并列、切分位候选顺序、兜底均分都做了确定性处理，两次切分输出完全相同。
6. **文档未提及的自造机制"零能量谱侧车"可靠**：在 27 种配置组合下（NaN 前体、零能量谱、三种离子模式策略、`min_matched_peaks=0`、`k>N`、`threshold=0.0`、`exclude_spectrum_id`、`min_matched_peaks>1`）与穷举基线逐位一致。
7. **NaN 前体的兜底是必需的**：实测 `np.searchsorted` 在含 NaN 的 `precursor_min/max` 数组上行为"未定义但确定"——`side="right"` 会把 NaN 树纳入候选范围，全靠 `src/jetf/search.py:92` 的显式 NaN 检查兜住。建议加注释标明它承担正确性职责，避免被后人当冗余删除。
8. **全库规模下 24 组查询零不一致**（身份 + 开放各 12 组），候选树仅占森林 0.60%。

---

## 八、未覆盖范围

- **未构造"向量化线性扫描基线"做对照**：仓库内的 `search_exhaustive` 是 Python 逐条循环，其身份检索耗时（11.8 ms）主要由元数据扫描开销构成（见 §5.6 口径校正）。一个把全库上界计算批量化（如对 7.44M 个 posting cell 做一次 `np.add.reduceat` 分段聚合）的暴力基线没有任何量级更差的理由，因此**森林相对于"良好工程化的暴力基线"的净优势未被直接测量**，§5.6 的 5~16× 只能读作"相对于本仓库 Python 穷举"的数字。
- 文档 §1.2 的两项几何指标（"叶间包络重叠率砍半至 10.9%"、"包络体积压缩 52.7%"）仓库内无度量工具，**未独立验证**。
- 非默认 `tree_capacity` / `leaf_capacity` / `max_candidate_axes` 配置未做扫描。
- 4.5 GB 的 `ALL_GNPS.mgf` 与 2.5 GB 的 `cleaned_spectra.mgf` 未实测（仅用 GNPS-LIBRARY.mgf 全库 16,106 谱）。
- 未验证多进程/多线程并发检索场景（`search_forest` 本身无状态、不修改索引，推测可重入，但未实测）。
- 未评估 `fragment_tolerance_da` 非 0.02 时与网格宽度的交互（`window_cells` 用 `nextafter` 做保守超集，理论上安全方向）。

---

## 九、修复优先级建议

| 顺序 | 项 | 工作量 | 收益 |
| :---: | :--- | :--- | :--- |
| 1 | **P0** 负强度守卫（`preprocess_query` / `search_forest` 入口截断或拒绝）+ 文档补充 $u_i \ge 0$ 前提 | 数行 | 消除一类静默漏检 |
| 2 | **P1-1** mass 有序性断言 | 数行 | 公共 API 不再静默错误 |
| 3 | **P1-2** 快照库指纹 + `n_spectra` 断言 | 十数行 | 消除整类错配事故 |
| 4 | **P1-3** 查询归一化校验或内部归一 | 数行 | 分数语义可信 |
| 5 | **P2-3** 查询侧 cell 上下界提出循环 + 根层批量求值 | 中等 | 开放检索主要耗时（17.6 ms/query）可大幅下降 |
| 6 | **P2-2** SAH 前缀扫描增量代价 | 中等 | 构建从 105.7 s 降至十几秒 |
| 7 | **P2-4 / 5.5** 叶容量与文档叙事同步 | 视决策 | 收敛节点数、包络内存、文档可信度 |
| 8 | **P3** 死代码清理（$e_h$、`support_energy`、`seed_count`、重复偏移数组、零引用依赖、双套解析器） | 小 | 包络内存约 -70 MB，代码路径收敛 |

建议 1~4 项连同 §3.3 的复现用例一起作为回归夹具，先钉住正确性再动性能优化。

---

*本报告中的全部数值均来自本轮实测：全库 GNPS-LIBRARY.mgf（16,106 谱 / 7,441,781 峰）、2000 谱分层子集、以及针对每个结论构造的最小合成复现。*
