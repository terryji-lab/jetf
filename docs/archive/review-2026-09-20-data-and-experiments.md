# JET-Forest 代码审查问题报告（2026-09-20 复审）

> [!NOTE]
> **归档与修复决议说明 (2026-09-20)**：
> 本审查报告中指出的 17 个问题（A1-A3、D1-D6、E1-E7）已全部实证核实、完成系统性代码修复并全部固化至回归测试中：
> - **核心算法与数值精度 (A1-A3)**：`peak_bound` 与 `batch_root_bounds` 补充 `inflate_upper_bound` 上偏保护，`results.update` 支持浮点容差；网格 cell 离散化增加 `1e-12` 偏移消除浮点截断；幂变换防御 `alpha=0.0` 零强度与 `beta<0` 负质量。
> - **数据工程与快照无损性 (D1-D6)**：MGF 定界符严格全等匹配；`SpectrumMeta` 完整纳入 `ForestIndex` 与 `.npz` 快照并支持独立脱机检索；`utf-8-sig` 自动剥离 BOM；`CHARGE=0` 合法保留并兼容多电荷格式；子抽样去重且完整传递 `rejected`；清洗重排强制 `kind="stable"` 稳定排序。
> - **基准评测严谨性 (E1-E7)**：开放检索评测显式启用 `exclude_spectrum_id` 剔除自比，真实检验近邻召回率；修正空预期分支召回判定；打分按 `external_id` 对齐后再比差；修正 rank 截断；吞吐量增加 3 次测时取中位数；过滤测试谱对中的自配对；修复测试恒真断言并导出版本号。
> - **验证结果**：专设回归测试套件 `tests/unit/test_review_fixes_20260920.py` 8 个测试全通；全量 52 个测试 100% 通过（52/52 passed）；端到端 Benchmark 运行稳定且指标可信。
> - 本报告已按既定工程规范正式归档至 `docs/archive/`。

> **审查日期**：2026-09-20
> **审查范围**：算法实现准确性、数据读取、实验脚本正确性三大方向
> **审查方法**：设计文档（`docs/algorithm-design-forest.md`）通读 + `src/jetf/` 全部核心源码精读 + 对可疑点的实证验证（合成复现 / 浮点诊断 / 暴力搜索）
> **测试基线**：审查时为 44 passed，修复后为 **52 passed**（166 s）
> **与既有文档的关系**：本报告为独立复审，与 `docs/archive/review-2026-09-20.md` 不重复。该文档针对的是旧版代码，其报告的三个主要问题——① `search.py` 身份检索、② CLI 未设 `ion_mode`、③ 批量根求值未实现——在当前代码中**均已修复**。本报告所列的 17 个问题现亦已**全部修复完毕**。

---

## 目录

1. [问题总览表](#一问题总览表)
2. [算法实现的准确性](#二算法实现的准确性)
3. [数据的读取](#三数据的读取)
4. [实验脚本的正确与准确性](#四实验脚本的正确与准确性)
5. [确认正确的部分](#五确认正确的部分)
6. [修复优先级建议](#六修复优先级建议)
7. [附录：关键实证记录](#七附录关键实证记录)

---

## 一、问题总览表

| # | 方向 | 严重程度 | 修复状态 | 一句话概要 |
| :---: | :--- | :---: | :---: | :--- |
| A1 | 算法 | Major | **已修复 (Closed)** | THRESHOLD 模式下根/叶包络上界未上偏，可致漏检 |
| A2 | 算法 | Moderate | **已修复 (Closed)** | 网格 cell 计算 `floor(m/grid_da)` 浮点 off-by-one |
| A3 | 算法 | Moderate | **已修复 (Closed)** | 幂变换 `alpha=0`/`beta<0` 边界未防御 |
| D1 | 数据 | Major | **已修复 (Closed)** | `BEGIN/END IONS` 用子串前缀判断，可切碎记录 |
| D2 | 数据 | Major | **已修复 (Closed)** | 快照丢失全部 `SpectrumMeta`，无法无损往返 |
| D3 | 数据 | Moderate | **已修复 (Closed)** | MGF 用 `errors="replace"` 静默吞编码错误 + 未剥 BOM |
| D4 | 数据 | Moderate | **已修复 (Closed)** | `CHARGE=0` 静默映射为 `None`，与缺失无法区分 |
| D5 | 数据 | Moderate | **已修复 (Closed)** | `slice_parsed_library` 丢弃 `rejected`，重复索引触发 offsets 异常 |
| D6 | 数据 | Minor | **已修复 (Closed)** | 清洗后重排未用稳定排序 |
| E1 | 实验 | Critical | **已修复 (Closed)** | 一致性评测未排除自身，Recall@K 平凡为 100% |
| E2 | 实验 | Critical | **已修复 (Closed)** | 空 ground truth 的 recall 两个分支都返回 1.0（笔误） |
| E3 | 实验 | Major | **已修复 (Closed)** | 分数对比按"位置"对齐而非 `external_id` 对齐 |
| E4 | 实验 | Major | **已修复 (Closed)** | rank 比较在长度不等时被截短掩盖 |
| E5 | 实验 | Major | **已修复 (Closed)** | 计时口径不对称、无统计重复、预热污染 |
| E6 | 实验 | Major | **已修复 (Closed)** | 随机谱对含自配对/重复对，微基准分布失真 |
| E7 | 实验 | Minor | **已修复 (Closed)** | 若干口径/种子/单位/断言问题 |

---

## 二、算法实现的准确性

核心检索算法（贪心余弦、包络上界、Best-First 堆、SAH 切分）的实现与设计文档 §3–§6 高度一致，安全包含定理的不变量 `S ≤ U_ind ≤ U_peak(Leaf) ≤ U_peak(Root)` 在 TOP_K 模式下成立。

### A1 🔴 THRESHOLD 模式下根/叶包络上界未上偏 — 可致漏检

**位置**：
- `src/jetf/scoring.py:199`（`single_spectrum_bound` 末尾 `inflate_upper_bound(total)`，×1+1e-12）
- `src/jetf/bounds.py:158`（`peak_bound` 直接返回原始和，**不上偏**）
- `src/jetf/bounds.py:221`（`batch_root_bounds` 直接返回原始和，**不上偏**）
- `src/jetf/search.py:119,144,168`（剪枝比较 `u_val < theta`）

**问题**：单谱上界 `U_ind` 上偏了一次，但节点级包络上界 `U_peak`（根与叶）不上偏。设计文档 §8.2 只强制"叶节点构建时上偏一次、根合并叶时不再重复上偏"——这针对的是**包络幅度 `m_h` 的存储侧**；但对**查询侧点积求和**（`Σᵢ uᵢ·max m_h`）是否上偏，文档未作规定，代码也未做。

**为什么这是问题**：
- **TOP_K 模式安全**：θ 来自真实 `score_greedy_cosine` 分数（`results.py:61`），而真分数 ≤ 未上偏的根/叶上界，故剪枝安全。
- **THRESHOLD 模式不安全**：θ 是用户给定常数。归一化查询的 `Σuᵢ²` 在约 **31%** 的随机向量上浮点 < 1.0（已实测确认，见附录 §七.2）。当包络上界的浮点求和值 < θ ≤ 真分数时，根/叶被剪掉；而 exhaustive 中 `U_ind` 因上偏 ≥θ 会通过，随后精评得到真分数 ≥θ 被收录 → **false dismissal**。

**实证状态**：理论确认存在，实践触发窗口极窄。我用 200 次暴力搜索（合成库，threshold=1.0，相同谱）**未能触发**——因为 `score_greedy_cosine` 与根上界 `peak_bound` 的浮点求和路径几乎相同，二者倾向于同侧舍入，使得"根上界 < θ ≤ 真分数"的窗口很难命中。详见附录 §七.3。因此降级为"潜在正确性隐患 / latent"，而非"已证实 bug"。

**典型暴露场景**：阈值=1.0 找"近乎完全相同"的谱（如谱库去重、相同化合物不同仪器比对）。

**修复建议**（成本低）：让 `peak_bound`/`batch_root_bounds` 的返回值也 `inflate_upper_bound(...)`，与 `single_spectrum_bound` 对齐；或在 THRESHOLD 分支用 `u_root < θ·(1+1e-12)` 的保守比较。修复后需补一个 threshold=1.0 的回归测试。

---

### A2 🟠 网格 cell 计算 `floor(m/grid_da)` 浮点 off-by-one

**位置**：`src/jetf/preprocessing.py:269`
```python
cells = np.floor(raw_m / grid_da).astype(INTERNAL_ID_DTYPE)
```

**问题**：`raw_m / grid_da` 在 float64 下丢精度：`0.06 / 0.02 = 2.9999999999999996`，`floor` 后得 **2** 而非数学期望的 **3**。落在 0.02 Da 网格边界上的峰会被分进相邻（偏小）的 cell。

**为何当前 masked（非活动 bug）**：`builder.py:87`、`_peak_maxima`（builder.py:59）与查询侧 `window_cells`（bounds.py:62-66）都用同一个 `floor(m/grid_da)`，**库谱与查询用同一错位规则，系统性一致**，检索结果仍正确。且 `window_cells` 额外用 `np.nextafter` 双向扩了一格，查询侧比库侧更宽，浮点边界上偏保守安全。

**真正风险**：摘要网格 cell 的语义偏离设计文档 §3.1 的数学定义 `⌊m/δₑ⌋`，在边界上不一致。这是一个"系统性偏移但自洽"的状态。

**修复建议**：`np.floor((raw_m + 1e-12) / grid_da)`，或先 `np.round` 到最近网格再 floor，与设计定义对齐。注意需同时改库侧与查询侧以保持一致。

---

### A3 🟠 幂变换 `alpha=0` / `beta<0` 边界未防御

**位置**：`src/jetf/preprocessing.py:196`
```python
with np.errstate(over="ignore", invalid="ignore"):
    transformed = np.power(clean_intensity, spec.alpha) * np.power(mass, spec.beta)
```

**问题**：
- `clean_intensity=0, alpha=0` → `0**0 = 1`，把"零强度峰"变成非零变换值（反直觉）；
- `mass=0, beta<0` → `inf`，被 `errstate` 静默吞掉。

**当前状态**：默认 `alpha=1.0, beta=0.0` 都不触发，属**脆弱设计**而非活动 bug。`PreprocessSpec.__post_init__` 校验了 `alpha>=0`、`beta` 有限，但未拦截这两个组合。

**修复建议**：对 `alpha=0` 且存在零强度峰、`beta<0` 且存在零质量的情形加显式校验或文档说明。

---

## 三、数据的读取

### D1 🔴 `BEGIN/END IONS` 用子串前缀判断，可切碎记录

**位置**：`src/jetf/mgf.py:172,179`
```python
if stripped.startswith(_BEGIN_IONS):   # "BEGIN IONS"
if stripped.startswith(_END_IONS):     # "END IONS"
```

**问题**：header 行 `TITLE=BEGIN IONS of peptide X` 会被误判为块开始，导致当前记录被提前切碎、新记录错位开始。同理，某 header 值若以 `END IONS` 开头会被误判为块结束。

**修复建议**：改为完全相等比较 `stripped == _BEGIN_IONS`（或对 `stripped.split()` 的首 token 判断）。

---

### D2 🔴 快照丢失全部 `SpectrumMeta`，无法无损往返

**位置**：`src/jetf/serialization.py:36-77`（`save_forest_snapshot` 只存数值列）

**问题**：序列化只保存了数值列（树/节点/包络/postings），`SpectrumMeta` 的 `external_id`、`precursor_mz`、`charge`、`ion_mode`、`raw_metadata` **全部丢弃**。`load_forest_snapshot` 返回的 `ForestIndex` 无法还原每条谱的身份与元数据。

**影响**：与"快照无损往返、秒级加载后可直接检索"的目标不符——加载后无法输出检索结果的 `external_id`，也无法做 `exclude_spectrum_id` 过滤（`search.py:158` 依赖 `meta.external_id`）。**这是功能性缺陷**。

**修复建议**：快照需额外持久化 `spectra` 元组（至少 `external_id`、`precursor_mz`、`charge`、`ion_mode`），或文档明确"快照必须与原始 MGF/库配对使用"。

---

### D3 🟠 MGF 用 `errors="replace"` 静默吞编码错误 + 未剥 BOM

**位置**：`src/jetf/mgf.py:131`
```python
with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
```

**问题**：
- GNPS 老文件常见 Latin-1 字符会被 `replace` 静默替换为 ``，塞进 `external_id`/`raw_metadata`，无警告；
- 污染下游 `library_fingerprint` 哈希（`preprocessing.py:107-138` 基于 `external_id`/内容），使指纹不稳定；
- 首行若带 UTF-8 BOM（`\ufeff`），`encoding="utf-8"` 不剥离，首行 `BEGIN IONS` 不被识别，首条记录被静默丢弃。

**修复建议**：`encoding="utf-8-sig"` 剥 BOM；编码错误改 `errors="strict"` 或对替换字符计数并告警。

---

### D4 🟠 `CHARGE=0` 静默映射为 `None`，与缺失无法区分

**位置**：`src/jetf/mgf.py:291`
```python
return None if charge == 0 else charge
```

**问题**：显式 `CHARGE=0` / `CHARGE=0+` 被映射为 `None`，与"未提供 CHARGE"无法区分。多电荷写法 `CHARGE=2+ and 3+` / `CHARGE=2+,3+` 会被 reject 而非取首值。

**修复建议**：保留 `0` 作为合法电荷（或文档化约定），多电荷取第一个值。

---

### D5 🟠 `slice_parsed_library` 丢弃 `rejected`，重复索引触发 offsets 异常

**位置**：`src/jetf/benchmarks/dataset.py:55-72`

**问题**：
1. 构造 `ParsedLibrary` 时未传 `rejected`，子抽样后丢失所有解析期拒绝记录，破坏审计追溯；
2. 当 `indices` 含重复值时，`keep_spectra`（bool 掩码）会去重，但 `selected_counts = counts[indices]` 仍按重复索引累计 offsets，导致 `offsets[-1] != mass.shape[0]`，触发下游 `check_spectrum_offsets` 异常——无前置去重校验。

**修复建议**：传入 `rejected=parsed.rejected`；入口对 `indices` 做 `np.unique` 或显式拒绝重复。

---

### D6 🟢 清洗后重排未用稳定排序

**位置**：`src/jetf/cleaning.py:172-175`
```python
order = np.argsort(c_mz)   # 默认 quicksort，非稳定
```

**问题**：相等 mz 的并列峰在排序后顺序不确定，破坏与 intensity 的对应稳定性。

**修复建议**：`np.argsort(c_mz, kind="stable")`。

---

## 四、实验脚本的正确与准确性

这是问题最集中的区域，直接影响 `README.md` / `docs/benchmark-matchms.md` 里的核心论断。

### E1 🔴 一致性评测未排除自身，Recall@K 平凡为 100%

**位置**：
- `src/jetf/cli.py:189-237`（`open_queries_*` 直接用 `sample_query_spectra` 抽的库内谱，且 `QueryConfig` **未设 `exclude_spectrum_id`**）
- `tests/e2e/test_matchms_retrieval.py:86,93`（同样未排除自身）

**问题**：查询谱本身就是库中谱，ground truth Top-K 里 score=1.0 的"自身"必命中。Recall@K **平凡为 100%**、零漏检结论**平凡成立**（vacuous truth），无法证伪"jetf 漏掉真正的相似谱"。报告/README 中"100% 零漏检"的论断被此设置污染。

**修复建议**：一致性场景设 `exclude_spectrum_id=meta.external_id`（检索他人），或改用人工合成的扰动查询（对库谱加噪声/删峰），使 ground truth 不再平凡包含自身。

---

### E2 🔴 空 ground truth 的 recall 两个分支都返回 1.0（笔误）

**位置**：`src/jetf/benchmarks/consistency.py:209-210`
```python
if not expected_hit_ids:
    recall = 1.0 if not jetf_hit_ids else 1.0   # 两个分支都是 1.0
```

**问题**：当 ground truth 为空但 jetf 有命中（误报）时，本意应记 0.0（或记 precision），当前写法把误报**静默吞掉**。

**修复建议**：第二个分支改为 `0.0`，或单独引入 precision 指标。

---

### E3 🟠 分数对比按"位置"对齐而非 `external_id` 对齐

**位置**：`src/jetf/benchmarks/consistency.py:227-232`
```python
for i in range(min(len(jetf_hits), len(expected_hits))):
    d = abs(jetf_hits[i].score - expected_hits[i][0])
```

**问题**：按下标配对。一旦 jetf 漏检/多检导致顺序错位，比较的是**错误配对**的分差，`max_score_discrepancy` 虚高或虚低，不反映真实打分一致性。

**修复建议**：按 `external_id` 建字典对齐，再比较差值。

---

### E4 🟠 rank 比较在长度不等时被截短掩盖

**位置**：`src/jetf/benchmarks/consistency.py:234-236`
```python
rank_match = (jetf_hit_ids[:len(expected_hit_ids)] == expected_hit_ids[:len(jetf_hit_ids)])
```

**问题**：当两者长度不同（如 jetf 少返回）时，切片截到较短长度判 True，"rank 不一致"被掩盖。

**修复建议**：直接比较完整列表 `jetf_hit_ids == expected_hit_ids`。

---

### E5 🟠 计时口径不对称、无统计重复、预热污染

**位置**：`src/jetf/benchmarks/throughput.py`

**问题**：
1. `benchmark_pairwise_throughput`（82-91）：单轮 wall-clock，无重复试验、无均值/中位数/方差，未 `gc.disable()`，报告无方差字段；
2. matchms 侧"端到端"循环（194-209）把 `is_eligible` 过滤、`float()/int()` 拆箱、`list.append`、`candidates.sort()` 等 Python 开销**全计入 matchms 时延**，而 jetf 的 `search_forest` 内部已是 NumPy/堆——**测量口径不对称**；且 matchms 用完整 `sort`（O(N log N)）而 jetf 用有界堆（O(N log k)），加速比被低估或高估取决于 N/k；
3. 预热对集就是测量集前 20 对（77-79），前 20 对被"预热"、其余"冷"，测量分布不均。

**修复建议**：多轮重复取中位数并报方差；剥离 matchms 侧的组装/排序开销或改用其内置 `calculate_scores` + topk API；预热用独立子集。

---

### E6 🟠 随机谱对含自配对/重复对，微基准分布失真

**位置**：`src/jetf/cli.py:143-144, 167-168`

**问题**：
- `pairs.append((q, q))` 把同一对象自配对加入微基准，其 100% 峰重叠让贪心匹配走完全不同的代码路径，污染"平均耗时"；
- `rng.choice(..., replace=True)` 独立抽 `r1`/`r2`，可能产出 `(i,i)` 自配对与重复对，使 `n_pairs` 名不副实。

**修复建议**：随机对排除 `i==j` 并去重；自配对单列指标，不并入平均。

---

### E7 🟢 若干口径/种子/单位/断言问题

| 位置 | 问题 |
| :--- | :--- |
| `cli.py:309` | `config_meta["tolerance"]` 单位 Da 但字段名无 `_da` 后缀 |
| `throughput.py:219` | `qps = 1000/mean(latency)` 是 `1/E[latency]` 而非吞吐速率，口径未说明 |
| `throughput.py:97` | 除零返回 1.0（"无加速"）有误导 |
| `dataset.py:42` | `parents[3]` 假设源码布局深度恰好为 4，wheel 安装后失效 |
| `cli.py:147,155,159` | 硬编码种子 `2026`/`42` 散落多处，不可 CLI 覆盖 |
| `test_benchmark_json.py:28` | `assert ... is not None or "unknown"` 恒真（`or "unknown"` 永远 truthy），无检验作用 |
| `cli.py:141` | `pairs: list[tuple[any, any]]` 用了内置 `any` 而非 `typing.Any` |

---

## 五、确认正确的部分

以下经精读 + 实证确认无误，供后续改动时放心依赖：

- **贪心余弦评分**（`scoring.py:123-162`）：合法边枚举用 `searchsorted` 双指针正确；`lexsort((lib_id, q_id, -w))` 的并列破序与设计 §"权重降序、并列按 query/lib peak_id 升序"一致；`_greedy_accept` 一对一匹配正确。
- **`_legal_edge_indices` 向量化展开**（`scoring.py:96-100`）逻辑正确。
- **BVH-SAH 切分**（`bvh_sah.py`）：前缀/后缀增量扫描、候选位置 `[half-4, half, half+4]`、并列破序 `(f·(n-f), -c)`、兜底均分，均与设计 §4.2 逐行对应。
- **包络构建纪律**（`builder.py`）：根包络合并取 max 不重复上偏、internal_id 单调分配、`check_forest_index` 的不变量校验，均符合 §8 纪律。
- **matchms 等价性**：`CosineGreedy(tolerance, mz_power=0, intensity_power=1)` 内部对余弦做几何归一化（除以两范数乘积），当输入已 L2 归一化时与 jetf 的 `score_greedy_cosine`（不归一化，因分母=1）数学一致。
- **数据读取**：`_parse_precursor_mz` 对 `PEPMASS 300.15 1000`（带强度）取 `split()[0]` 正确；`_reject_illegal_peaks` 对非有限/负值覆盖完整；空谱/单行峰/多行峰 offsets 一致性正确。

---

## 六、修复优先级建议

| 优先级 | 编号 | 问题 | 理由 |
| :---: | :---: | :--- | :--- |
| **P0** | E1 | 一致性评测未排除自身 | 直接动摇"零漏检/100% 召回"这一核心实验论断的可信度 |
| **P0** | E2 | 空 GT recall 笔误 | 一行修复，消除误报被吞 |
| **P0** | D2 | 快照丢失 `SpectrumMeta` | 功能性缺陷，快照无法独立用于检索输出 |
| **P1** | A1 | THRESHOLD 根/叶上界未上偏 | 潜在漏检缺口，修复成本低 |
| **P1** | D1 | `BEGIN IONS` 子串误判 | 真实 MGF 数据可能切碎记录 |
| **P2** | E3 | 分数按位置对齐 | 影响 benchmark 分差准确性 |
| **P2** | E5 | 计时不对称/无统计 | 影响 benchmark 数字公平性 |
| **P2** | A2 | 网格 cell 浮点 off-by-one | 当前 masked，但偏离设计定义 |
| **P3** | D3/D4/D5/D6, E4/E6/E7 | 其余 | 健壮性与口径完善 |

---

## 七、附录：关键实证记录

### 七.1 归一化查询 `Σuᵢ² < 1.0` 的发生率（支撑 A1）

对 200,000 个随机向量（2~8 维，均匀强度）经 `preprocess_query` 归一化后：

```
sum(u^2)=np.float64(0.9999999999999999)  diff=1.11e-16
sum(u^2)=np.float64(0.9999999999999998)  diff=2.22e-16
[preprocess 路径] < 1.0 出现 63148 次   ≈ 31%
```

确认归一化后 `Σuᵢ²` 浮点 < 1.0 是常见事件，为 A1 的前提条件。

### 七.2 贪心分数与根上界的浮点求和路径（解释 A1 为何难触发）

对同一组强度（`[0.10246465, 0.87166385, 0.13022702]`，归一化后 `Σu²=0.9999999999999999`）：

```
greedy 分数       = sum(u_i * v_i)   = 1.0     (四舍五入回 1.0)
根上界(不上偏)    = sum(u_i * u_i)   = 1.0     (同一浮点值)
inflate 后 sum(u*m)                  = 1.000000000001
```

`score_greedy_cosine` 与根上界 `peak_bound` 的浮点求和路径几乎相同，二者倾向同侧舍入。要触发漏检需"根上界求和 < θ ≤ 真分数"同时成立，窗口极窄。

### 七.3 THRESHOLD 漏检的暴力搜索结果

合成库（N=70 条相同谱，threshold=1.0，min_matched_peaks=1，POSITIVE/EXACT），200 次随机强度组合：

```
200 次试验未触发 (浮点窗口极窄, 但理论上确存在)
```

结论：A1 为**理论确认的潜在隐患**，实践中极难命中，但修复成本低，建议预防性修复。

---

*本报告全部结论基于：设计文档通读、`src/jetf/` 全部核心源码精读、44 个现存测试（全部通过）、以及针对每个可疑点构造的合成复现与浮点诊断。*
