# JET-Forest 算法设计与原理：前体自适应浅层包络森林与 BVH-SAH 紧致索引

> **前体自适应浅层包络森林与表面积启发式质谱检索（Precursor-Binned Envelope Forest with SAH Leaf Partitioning，JET-Forest）**
> 
> 本文档是 JET-MS 范式升华后的完整架构与算法设计规格书。它详细定义了从“深层全局聚类单树”转向“前体自适应浅层包络森林”的数学原理、几何分裂准则、内存布局与检索执行流程。

---

## 目录

1. [背景与范式升华：从单树到森林](#一背景与范式升华从单树到森林)
2. [层次化架构概览（Forest Hierarchy）](#二层次化架构概览forest-hierarchy)
3. [核心数学定义与安全证明](#三核心数学定义与安全证明)
4. [图形学 BVH-SAH 紧致叶切分算法](#四图形学-bvh-sah-紧致叶切分算法)
5. [数据结构与内存布局（Columnar Memory Layout）](#五数据结构与内存布局columnar-memory-layout)
6. [双模态统合型森林检索执行管线（Dual-Mode Search Pipeline）](#六双模态统合型森林检索执行管线dual-mode-search-pipeline)
7. [复杂度与存储开销分析](#七复杂度与存储开销分析)
8. [工程实现规范与正确性纪律](#八工程实现规范与正确性纪律)

---

## 一、背景与范式升华：从单树到森林

### 1.1 历史教训：全局深树的“两大死穴”
在早期 JET-MS 的设计中，试图借鉴空间度量树（Metric Tree）的思想，用 **1 Da 粗网格 Spherical k-means** 构建一棵覆盖全库所有谱的全局递归深树（分支因子 $f=4$，深度 4~5 层）。经过严密的实验探针，该方案被证实存在两大结构性硬伤：
1. **50 倍尺度错位（Scale Mismatch）**：聚类在 1 Da 粗空间进行，而评分是 0.02 Da 极窄容差下的峰匹配，导致几何上“谱形接近”并不蕴含“高 Cosine 分”；
2. **S-Tree 根节点饱和（Root Saturation）**：由于全库所有谱强行汇聚于单个根节点，高层节点的包络（OR-Union）几乎占满了 0~2000 Da 质量轴，退化为全通地毯。**根节点与上层节点的剪枝率恒为 0.0%**。

### 1.2 破局之道：包络森林（Envelope Forest）+ BVH-SAH
JET-Forest 彻底抛弃“全局单根深树”，重塑为三级混合架构：
- **第一级（物理级硬分区）**：按前体质量（Precursor m/z）与离子模式将全库切分为物理互不相交的轻量前体桶，根除跨前体污染；
- **第二级（族群级浅树）**：每个前体桶独立为一棵**两层浅树（Root 64 $\to$ Leaves $\le 16$）**。由于每棵树只覆盖化学前体相近的 64 条谱，根包络依然高度稀疏，**根节点整树排除率实测约 58.8%**（16,106 谱 GNPS-LIBRARY 开放检索）；
- **第三级（微观级致密叶）**：桶内采用图形学光线追踪的 **表面积启发式（Surface Area Heuristic, SAH）** 进行 0.02 Da 空间二分，目标叶容量 $C_{leaf} \le 16$（典型单树动态生成 5~6 个叶节点，容量分布在 8~16，均值约 10.3），将叶间包络重叠率砍半至 **10.9%**（早期设计估算值，未独立验证），包络体积压缩 **52.7%**（早期设计估算值，未独立验证），叶剪枝率实测约 **96.8%**（对已求值叶）；
- **检索内核特化**：当前实现原生特化于**全库开放检索与阈值检索（Open / Threshold Search）**，通过连续内存的 SIMD 批量根求值（`batch_root_bounds`）与全局 Best-First 优先队列跨树动态推进，以零漏检（Zero False Dismissals）实现大幅超越传统穷举的高吞吐检索。

---

## 二、层次化架构概览（Forest Hierarchy）

```
                                  全 库 谱 集 (N = 16,106)
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               ▼                             ▼                             ▼
       [ 正离子模式分区 ]             [ 负离子模式分区 ]            [ 未知模式分区 ]
       (Positive Partition)          (Negative Partition)          (Unknown Partition)
               │
      按 Precursor m/z 严格升序排序并物理分桶 (每个桶固定覆盖 C_tree = 64 条谱)
               │
  ┌────────────┴────────────┬─────────────────────────┬────────────────────────────┐
  ▼                         ▼                         ▼                            ▼
【浅树 T_0】             【浅树 T_1】              【浅树 T_2】       ...     【浅树 T_M】
(覆盖谱 0~63)            (覆盖谱 64~127)           (覆盖谱 128~191)                (最后剩余谱)
  │                         │
  ├─ Root 包络              ├─ Root 包络 (覆盖 64 谱)
  │  (Z_root, m_root)       │
  │                         │  [ 桶内执行 BVH-SAH 紧致二分 ]
  ▼                         ▼
  ├─ Leaf 0 (8~16 谱, 0.02 Da 紧致包络)
  ├─ Leaf 1 (8~16 谱, 0.02 Da 紧致包络)
  ├─ ...
  └─ Leaf K (8~16 谱, 典型 5~6 叶/树, 目标 <= 16)

       │
       ▼ 每个叶节点直接对齐一个物理微块 (MicroBlock, 连续峰数据)
       └─ 全局 Best-First 堆推进 + U_ind 单谱快速过滤 + Greedy Cosine 精评
```

---

## 三、核心数学定义与安全证明

### 3.1 质量网格与精评表示
- **摘要网格（Summary Grid）**：宽度 $\delta_e = 0.02$ Da。连续质量轴被划分为半开区间：
  $$C_h = [h \cdot \delta_e, (h+1) \cdot \delta_e), \quad h \in \mathbb{Z}$$
  峰 $m$ 所属 cell 下标为 $h = \lfloor m / \delta_e \rfloor$。
- **谱表示**：每条谱 $P$ 预处理后满足 $\sum_{j} v_j^2 = 1$，单谱 cell 能量为 $b_h(P) = \sum_{j \in C_h} v_j^2$。

### 3.2 节点包络摘要（Node Envelope）
对于任意节点 $B$（包含成员谱集合 $B \subset \mathcal{L}$），其包络摘要由两列等长数组构成（仅存储非零 cell，彻底消除稀疏存储浪费）：
1. **支持集（Support Set）$Z(B)$**：
   $$Z(B) = \bigcup_{P \in B} \{ h : P \text{ 在 } C_h \text{ 内有非零峰} \}$$
2. **最大单峰幅度（Max Peak Amplitude）$m_h(B)$**：
   $$m_h(B) = (1 + \epsilon_{\text{mach}}) \cdot \max_{P \in B, j \in C_h} v_j, \quad \epsilon_{\text{mach}} = 10^{-12}$$

### 3.3 节点级峰上界（Peak Bound）
设查询谱为 $Q = \{(m_i, u_i)\}$，匹配容差为 $\tau = 0.02$ Da。**前置条件：查询峰强度必须非负（$u_i \ge 0$）且经过 L2 归一化（$\sum_i u_i^2 = 1$）**。
- 查询峰 $i$ 的兼容 cell 集合定义为与区间 $[m_i - \tau, m_i + \tau]$ 相交的所有 cell：
  $$E(i) = \left\{ h \in \mathbb{Z} : \left\lfloor \frac{m_i - \tau}{\delta_e} \right\rfloor \le h \le \left\lfloor \frac{m_i + \tau}{\delta_e} \right\rfloor \right\}$$
- 查询峰在节点 $B$ 内的命中 cell 集合为 $E_B(i) = E(i) \cap Z(B)$。
- **节点峰上界**：
  $$U_{peak}(B) = \sum_{i} u_i \cdot \max_{h \in E_B(i)} m_h(B)$$
  若 $E_B(i) = \emptyset$，则该峰的最大幅度记为 0。

### 3.4 层次化安全包含定理（Safety Invariance Theorem）
> **定理 1（弱对偶与包含安全性）**：
> 设 $P \in \text{Leaf} \subset \text{Root}$，真实确定性 Greedy Cosine 分数为 $S(Q, P)$，单谱精确过滤界为 $U_{ind}(P)$。则对于任意查询 $Q$，恒有：
> $$S(Q, P) \le U_{ind}(P) \le U_{peak}(\text{Leaf}) \le U_{peak}(\text{Root}) \le 1 + 10^{-12}$$
> 
> **证明概要**：
> 1. 由 Greedy Cosine 语义，真实匹配是一对一合法配对集合 $M$。$S(Q, P) = \sum_{(i,j) \in M} u_i v_j \le \sum_{i} u_i \max_{j \in \text{legal}(i)} v_j = U_{ind}(P)$；
> 2. 对任意合法库峰 $j \in C_h$，必有 $h \in E_{\text{Leaf}}(i)$，且 $v_j \le m_h(\text{Leaf})$，故 $U_{ind}(P) \le U_{peak}(\text{Leaf})$；
> 3. 由于 $\text{Leaf} \subset \text{Root}$，支持集 $Z(\text{Leaf}) \subseteq Z(\text{Root})$，且对任意 cell $h$ 有 $m_h(\text{Leaf}) \le m_h(\text{Root})$，故 $U_{peak}(\text{Leaf}) \le U_{peak}(\text{Root})$。
> 证毕。该定理保证了**根节点剪枝和叶节点剪枝绝无假阴性（False Dismissal 恒为 0）**。

---

## 四、图形学 BVH-SAH 紧致叶切分算法

在每个包含 $N \approx 64$ 条谱的前体桶内，我们弃用 k-means，采用**表面积启发式（SAH）**进行递归二分，直到切为 4 个容量为 16 的紧致叶子。

### 4.1 质谱包络代价函数（Envelope SAH Cost）
在 3D 图形学中，光线穿透包围盒的先验概率正比于表面积；在质谱中，任意随机查询峰“击中”节点包络并产生高上界的期望成本，正比于**支持集宽度与峰幅度总和的乘积**：
$$\text{Cost}(B) = |Z(B)| \cdot \sum_{h \in Z(B)} m_h(B)$$
- 当簇内成员的峰高度重合时：$|Z(B)|$ 极小，$\text{Cost}(B)$ 达到极小；
- 当簇内成员错开时：$|Z(B)|$ 剧烈膨胀，$\text{Cost}(B)$ 受到严厉惩罚。

### 4.2 最佳投影轴搜索与增量前缀/后缀切分算法

为消除重复集合并集与幅度字典重算的巨大开销，工程实现采用了**前缀/后缀扫描增量维护（Incremental Prefix/Suffix Scan）**，并将并列破序做到位级确定（Bit-exact Determinism）：

```python
def split_sah_bvh(
    spectrum_indices,
    spec_cells,
    spec_amps,
    target_leaf_size=16,
    max_candidate_axes=10,
):
    """
    输入: 当前桶内的成员行序号列表 (N <= 64)、预抽取的 cell 集合与最大幅度映射
    输出: 切分出的紧致叶节点行列表，每个叶包含 8~16 条谱
    """
    indices = list(spectrum_indices)
    n = len(indices)
    if n <= target_leaf_size:
        return [indices]

    # Step 1: 统计子集内每个 0.02 Da cell 的谱出现频次
    cell_freq = {}
    for idx in indices:
        for c in spec_cells[idx]:
            cell_freq[c] = cell_freq.get(c, 0) + 1

    # Step 2: 挑选“信息量/方差最大”的候选切分轴 (Top-10 cells)
    # 评分准则: f * (n - f)，并列按 -c 破并列 (cell 编号升序，绝对确定性)
    candidate_axes = sorted(
        cell_freq.keys(),
        key=lambda c: (cell_freq[c] * (n - cell_freq[c]), -c),
        reverse=True,
    )[:max_candidate_axes]

    best_split = None
    min_total_cost = float("inf")
    half = n // 2
    min_leaf = max(8, target_leaf_size // 2)

    # Step 3: 在候选轴上寻找最小 SAH Cost 的平衡切分点 [half, half-4, half+4]
    if candidate_axes:
        candidate_positions = [
            pos for delta in (0, -4, 4)
            if min_leaf <= (pos := half + delta) <= n - min_leaf
        ]
        if not candidate_positions and 4 <= half <= n - 4:
            candidate_positions = [half]
        pos_set = set(candidate_positions)

        for axis_cell in candidate_axes:
            # 按谱在该 cell 的峰强度升序排序，并列按原谱序号升序破并列
            sorted_indices = sorted(
                indices,
                key=lambda idx: (spec_amps[idx].get(axis_cell, 0.0), idx),
            )

            # 前缀正向扫描: 增量维护 prefix_cells 与 prefix_amps
            prefix_cost = {}
            p_cells, p_amps = set(), {}
            for i, idx in enumerate(sorted_indices):
                p_cells.update(spec_cells[idx])
                for c, a in spec_amps[idx].items():
                    if c not in p_amps or a > p_amps[c]:
                        p_amps[c] = a
                if (split_len := i + 1) in pos_set:
                    prefix_cost[split_len] = len(p_cells) * sum(p_amps.values())

            # 后缀反向扫描: 增量维护 suffix_cells 与 suffix_amps
            suffix_cost = {}
            s_cells, s_amps = set(), {}
            for i in range(n - 1, -1, -1):
                idx = sorted_indices[i]
                s_cells.update(spec_cells[idx])
                for c, a in spec_amps[idx].items():
                    if c not in s_amps or a > s_amps[c]:
                        s_amps[c] = a
                if i in pos_set:
                    suffix_cost[i] = len(s_cells) * sum(s_amps.values())

            # 评估平衡候选位置
            for split_pos in candidate_positions:
                total_cost = prefix_cost.get(split_pos, float("inf")) + suffix_cost.get(split_pos, float("inf"))
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_split = (sorted_indices[:split_pos], sorted_indices[split_pos:])

    # 兜底: 若无特征候选轴或无法改善，按原顺序直接均分
    if best_split is None:
        best_split = (indices[:half], indices[half:])

    # Step 4: 递归切分左右子集
    left_leaves = split_sah_bvh(best_split[0], spec_cells, spec_amps, target_leaf_size, max_candidate_axes)
    right_leaves = split_sah_bvh(best_split[1], spec_cells, spec_amps, target_leaf_size, max_candidate_axes)
    return left_leaves + right_leaves
```

---

## 五、数据结构与内存布局（Columnar Memory Layout）

为保证极致的 CPU 缓存局部性与跨语言序列化兼容性（完全脱离 Python 对象指针与 GC），整个森林索引采用**扁平化列式数组（Flat NumPy / Arrow 兼容）**存储。

### 5.1 整体内存对象架构图

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ForestIndex (全库森林索引)                       │
├────────────────────────────────────────────────────────────────────────┤
│ 1. 分区列表 (Partitions: tuple[ForestPartition, ...]):                  │
│    - ion_mode:          Positive / Negative / Unknown                  │
│    - (tree_start, tree_end), (node_start, node_end), (id_start, id_end)│
│                                                                        │
│ 2. 森林树数组 (Trees, 长度 M):                                          │
│    - precursor_min:     [100.1, 150.3, ...] (float64, 桶前体下界)     │
│    - precursor_max:     [100.4, 150.8, ...] (float64, 桶前体上界)     │
│    - root_node_id:      [0, 5, 10, ...]     (int64, 指向 Node 列)    │
│    - leaf_offsets:      [0, 4, 8, 12, ...]  (int64, 对应叶段)        │
│    - leaf_node_ids:     [1, 2, 3, 4, 6, 7..](int64, 全部叶序号平铺)  │
│                                                                        │
│ 3. 节点数组 (Nodes, 包含全部 Root 和 Leaf):                             │
│    - is_leaf:           [False, True, True...](bool)                  │
│    - member_count:      [64, 16, 16, 16, 16...](int64)                │
│    - id_start:          [0, 0, 16, 32, 48...] (int64, 内部 ID 起点)   │
│    - id_end:            [64, 16, 32, 48, 64...](int64, 内部 ID 终点)   │
│    - tree_id:           [0, 0, 0, 0, 0, 1...] (int64, 所属树编号)     │
│                                                                        │
│ 4. 平铺包络摘要 (Envelopes Columnar Buffer, 两列稀疏存储):             │
│    - grid_da:           0.02 (float64, 摘要网格宽度)                  │
│    - node_envelope_offsets: [0, 850, 1200, ...] (int64, 节点偏移切片)  │
│    - cell_index:        [5002, 5003, 8100...] (int64, 严格递增支持集) │
│    - max_peak_amplitude:[0.85, 0.12, 0.45...] (float64, 最大单峰幅度) │
│                                                                        │
│ 5. 微块对齐峰缓冲 (MicroBlock Postings Buffer):                         │
│    - 内部 ID 连续递增，每个 Leaf 对应一个物理微块 (MicroBlock)          │
│    - (mass, intensity, energy, peak_id) 四列按成员内升序连续存储       │
│    - spectrum_offsets:  [0, 45, 120...] (int64, 谱峰切片边界)          │
│    - norm:              [1.0, 1.0...]   (float64, 原始 L2 范数)        │
│                                                                        │
│ 6. 双向映射与元数据侧车 (Mapping & Metadata Sidecar):                  │
│    - internal_to_row:   [0, 1, 2...] (int64, 内部紧凑 ID 到库行号)    │
│    - row_to_internal:   [0, 1, 2...] (int64, 库行号到内部紧凑 ID, -1为侧车)│
│    - zero_energy_members: ZeroEnergyMembers (零能量/零峰隔离侧车)     │
│    - library_fingerprint: str (全库采样哈希与元数据复合指纹)           │
│    - spectra:           tuple[SpectrumMeta, ...] | None (无损快照元数据)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 六、双模态统合型森林检索执行管线（Dual-Mode Search Pipeline）

JET-Forest 原生统合了**前体二分靶向检索（Targeted Search）**与**全库开放式检索/全局阈值检索（Open / Threshold Search）**两种检索模态，同一套索引结构，通过输入配置中的 `precursor_window` 自动无缝切换：

1. **模态一：前体二分靶向检索（Targeted Precursor Mode）**：当指定 `QueryConfig(precursor_window=PrecursorWindow(min_mz, max_mz))` 时，利用 `ForestTrees.precursor_min/max` 升序数组在 Stage 0 执行二分定位（$O(\log M)$），瞬间排除 $>96\%$ 的无关小树，候选树仅需展开 1~2 棵；
2. **模态二：全库开放式检索与阈值检索（Open / Threshold Mode）**：当未指定前体窗口（全库盲筛与开放检索）时，直接批量评估所有合法分区树根包络，通过全局 Best-First 优先队列跨树动态推进，以高剪枝率实现极致加速。

```mermaid
flowchart TD
    Q[输入查询谱 Q 与配置 Config] --> Validate[输入合规校验: 非负强度 + L2归一化]
    Validate --> CheckEmpty{Q 是否为空峰谱?}
    CheckEmpty -- 是 (0 峰快速短路) --> SuppZero[零能量侧车补足 (若需) 并立即返回]
    CheckEmpty -- 否 --> CheckPrec{Config 是否包含 precursor_window?}
    CheckPrec -- 是 (前体靶向模式) --> Stage0[Stage 0: 树前体二分查找 searchsorted<br/>定位重叠树区间, 排除 >96% 无关树]
    CheckPrec -- 否 (全库开放模式) --> AllTrees[取分区内全部树作为候选]
    Stage0 --> BatchRoot[Stage 1: SIMD 批量根求值 batch_root_bounds]
    AllTrees --> BatchRoot
    BatchRoot --> RootFilter{U_peak(Root) < theta?}
    RootFilter -- 是 (整树排除) --> PrunedTree[排除无关小树与全量叶子]
    RootFilter -- 否 (候选入堆) --> PushHeap[将幸存树根压入全局优先队列堆]
    PushHeap --> PopHeap[Stage 2: 弹出堆顶未访问最高界节点]
    PopHeap --> CheckTop{Node U_bound < theta?}
    CheckTop -- 是 --> Terminate[堆内所有剩余节点全量剪除! 提前终止]
    CheckTop -- 否 --> CheckLeaf{是否为叶节点?}
    CheckLeaf -- 否 (树根) --> ExpandLeaves[展开该树所有叶子并计算 U_leaf 入堆]
    CheckLeaf -- 是 (叶节点) --> ForMembers[Stage 3: 进入叶内微块 MicroBlock 连续切片]
    ForMembers --> UindFilter[计算单谱精确上界 U_ind]
    UindFilter --> CheckUind{U_ind < theta?}
    CheckUind -- 是 --> SkipExact[跳过精评]
    CheckUind -- 否 --> GreedyCosine[Stage 4: 执行确定性 Greedy Cosine 精评]
    GreedyCosine --> UpdateHeap[更新 Top-k 结果堆并动态抬升门槛 theta]
    UpdateHeap --> PopHeap
```

### 6.1 四阶段（含前体靶向 Stage 0）执行流程
0. **阶段 0：前体二分靶向过滤（可选 Stage 0，前体靶向检索模式）**：
   - 若查询配置中指定了 `precursor_window`，利用森林树数组中已按前体排序的 `precursor_min` 与 `precursor_max`；
   - 通过 `np.searchsorted` 以 $O(\log M)$ 时间复杂度快速定位与查询前体区间重叠的树索引切片 `[t_begin, t_finish)`；
   - 过滤掉边界外树，直接排除 $>96\%$ 的无关小树，通常仅需保留 1~2 棵候选树进入后续阶段；
   - 若未指定 `precursor_window`，则将分区内全部树作为候选直接进入阶段 1。
1. **阶段 1：SIMD 批量根求值（Batch Root Evaluation）**：
   - 针对候选树根（开放模式下全量约 $M \approx 251$ 棵，靶向模式下仅 1~2 棵）；
   - 在循环外一次性计算好查询峰兼容质量区间 `[cell_lower, cell_upper]`；
   - 调用 `batch_root_bounds`，通过多核 Numba JIT / 向量化内存复用求得根节点的 $U_{peak}(\text{Root})$。
2. **阶段 2：根级快速粗剪与建堆**：
   - 凡 $U_{peak}(\text{Root}) < \theta$ 的小树，连同其全部叶子及谱**瞬时排除**；
   - 幸存树根压入全局最大堆 `(-U_peak, entry_count, root_id, is_leaf=False, tree_id)`。
3. **阶段 3：全局最佳优先逐层展开（Best-First Priority Queue）**：
   - 每次从堆顶弹出上界最大候选节点：
     - 若当前节点界 $< \theta$，则堆内所有剩余节点均 $< \theta$，**瞬间剪除堆内全部剩余节点并跳出循环**；
     - 若为树根：遍历其子叶，求出叶包络界 $U_{peak}(\text{Leaf})$，若 $\ge \theta$ 则作为叶节点压入堆中；
     - 若为叶节点：进入该微块进行谱级精评。
4. **阶段 4：微块遍历、单谱 $U_{ind}$ 拦截与精评**：
   - 微块成员谱内部 ID 严格连续，直接零拷贝切片；
   - 逐谱调用 `single_spectrum_bound` 计算 $U_{ind}$，若 $< \theta$ 则拦截；
   - 幸存谱调用 `score_greedy_cosine` 执行精评，将满足条件的命中压入结果集，实时抬高门槛 $\theta$。

---

## 七、复杂度与存储开销分析

### 7.1 检索时间复杂度对比

| 检索阶段 | 全局单树 (M4 Global Tree) | 传统穷举扫描 (Exhaustive Scan) | JET-Forest 开放检索 (Open Best-First) | JET-Forest 前体靶向检索 (Targeted Precursor) |
| :--- | :---: | :---: | :---: | :---: |
| **阶段 0：前体二分过滤** | 不支持（深树全局交织） | $O(N)$ 遍历过滤 | 无（全库开放无限制） | **$O(\log M)$ 二分定位，瞬时排除 $>96\%$ 的树** |
| **阶段 1：根包络批量筛选** | $O(f \cdot \text{Depth}) \approx 36$ 次松散求值 | 无（全量扫描） | **$O(M) \approx 251$ 次 SIMD 向量化求值** | **仅评估 1~2 棵候选树根节点** |
| **阶段 2：整树整叶排除率** | 根部不剪枝（饱和全通） | 0% | **实测开放检索约 58.8%（16,106 谱全库）** | **整树排除率 $> 99\%$** |
| **阶段 3：叶节点展开** | 深度下探（无序） | 全量展开 | **仅展开高潜力幸存树的叶子（Best-First）** | **仅展开 1~2 棵目标树的 5~12 个叶子** |
| **阶段 4：精评调用次数** | 很高（包络松，幸存多） | $100\%$ 全量精评 ($N$ 次) | **极少（精评削减率 $> 98\%$）** | **微量（微块内数条至数十条谱）** |

### 7.2 空间开销与存储模型（16,106 谱全库实测）
- **森林拓扑列**：251 棵树 $\times$ 平均 6.16 叶 $\approx 1,798$ 个节点。节点元数据总计 $< 1$ MB；
- **包络两列平铺**：
  - 根节点与叶节点平铺包络采用双列 (`cell_index`: int64, `max_peak_amplitude`: float64) 稀疏存储；
  - 未清洗全谱实测内存常驻约 **214 MB**；
- **微块 Posting 峰缓冲**：
  - 全库未清洗 7,441,781 峰，采用严格的 64 位类型（`mass`: float64, `intensity`: float64, `energy`: float64, `peak_id`: int64）；
  - 744 万峰 $\times$ 4 列 $\times$ 8 字节 $\approx$ **238 MB**；
- **全库总内存常驻（未清洗原始谱）**：约 **441 MB**，压缩快照文件仅 **170 MB**，支持秒级全量加载（加载仅需 1.3 秒）！
- **工业级清洗后内存优化（matchms Cleaned 推荐配置）**：
  - 经 `matchms` 标准化清洗（Top-300 峰截断、0.1% 相对强度去噪）后，峰数量从 744 万大幅削减至约 200~250 万峰；
  - 稀疏包络尺寸显著缩小，微块峰缓冲降至约 60~80 MB；
  - **16k 库清洗后实际总常驻内存仅约 150 MB**，压缩快照文件降至 **60~70 MB**，内存开销减少超 65%，且完全保留核心高质量碎片特征。

---

## 八、工程实现规范与正确性纪律

1. **绝对一致性铁律（100% Exact Equivalency）**：
   - 无论开启 Top-K 检索、Threshold 检索，无论 $\theta$ 门槛高低，系统检索输出的 `hits` 列表必须与 `search_exhaustive`（穷举基线）**逐位相同（Bit-exact Match）**。任何一处逻辑错误导致的假阴性均判定为重大 Bug。
2. **浮点结合律与余量纪律（Inflation Discipline）**：
   - 叶节点构建时，最大单峰幅度必须调用 `inflate_upper_bounds(x)` 上偏一次（乘以 $1 + 10^{-12}$）；
   - 根节点合并叶节点时，只取 $\max$，严禁存储侧重复上偏；
   - 节点级查询求值 `peak_bound`、`batch_root_bounds` 以及单谱精确上界 `single_spectrum_bound` 在返回前均调用 `inflate_upper_bound` 进行查询侧上偏；
   - 贪心余弦打分在接近 1.0 时整定为 1.0，门槛判断支持 $10^{-12}$ epsilon 浮点容差（$U < \theta$，相等时不剪枝），彻底消除极端边界漏检隐患。
3. **网格离散与数值精度纪律（Grid Discretization Precision）**：
   - 峰幅度网格映射与查询兼容区间采用 `(raw_m + 1e-12) / grid_da` 防御浮点精度抖动，避免 $0.06 / 0.02 = 2.99999...$ 截断误入相邻 cell；
   - 幂变换预处理防御零强度峰在 $\alpha=0$ 下异常为 1.0 以及负质量非有限值。
4. **输入契约与防御纪律（Input Contracts & Invariants）**：
   - 查询谱强度必须非负（$u_i \ge 0$），在进入检索前必须经 L2 归一化（$\sum_i u_i^2 = 1.0$）；
   - 库谱与查询谱峰质量必须按升序严格排列；
   - 检索入口严格比对森林总谱数与全库采样指纹（Fingerprint），坚决防止快照与库错配。
5. **微块内部 ID 单调性与元数据无损快照**：
   - 每一棵树内部、每一个叶节点内部，谱的内部 ID（`internal_id`）必须严格连续单调分配，确保微块范围切片为半开区间 `[id_start, id_end)`；
   - 索引快照支持完整列式 `SpectrumMeta`（包含 `external_id`、`precursor_mz`、`charge`、`ion_mode`、`source_path`、`source_index`）无损持久化，原生支持无需原始 MGF 的独立脱机检索。
6. **数据清洗与拓扑保序纪律（Cleaning & Dialect Robustness）**：
   - 数据接入层原生兼容 MGF 多方言字段（`PRECURSOR_MZ`、`PARENT_MASS`、`PEPMASS`、`SPECTRUM_ID`），支持 UTF-8 BOM 自动剥离与多电荷解析，防止前体丢失导致的分桶退化；
   - 经 `matchms` 流水线清洗（相对强度过滤、Top-N 截断）后，谱图的所有峰必须采用稳定排序（`stable`）重新执行质量轴升序排序与连续 `peak_id` 分配，重编后的 `spectrum_offsets` 必须严格递增并符合连续列式内存不变量。
7. **零能量谱隔离侧车（Zero-Energy Sidecar）**：
   - 零能量谱（空谱）在构建期被隔离到 `ZeroEnergyMembers` 侧车结构中，不进入树索引；
   - 检索时通过 `supplement_zero_score` 按元数据资格补足零分命中，确保与穷举基线一致。
8. **排序确定性保证（Deterministic Tie-Breaking）**：
   - Greedy Cosine 候选边排序：权重降序 → 并列按 query peak_id 升序 → 再按 library peak_id 升序；
   - 检索结果排序：分数降序 → 并列按 external_id 升序 → 再按 spectrum_index 升序；
   - SAH 候选轴选择：`f·(n-f)` 降序 → 并列按 cell 编号升序；
   - SAH 谱排序：按谱在候选 cell 的峰强度升序 → 并列按原谱序号升序。
9. **批量根求值双路径（Batch Root Bounds Dual Path）**：
   - `batch_root_bounds` 提供 Numba JIT 并行内核（`fastmath=False`，确保位级一致性）与 NumPy 向量化降级内核，按环境自动切换；
   - 所有上界计算路径（`peak_bound`、`batch_root_bounds`、`single_spectrum_bound`）统一使用相对余量 `×(1 + 10⁻¹²)`（`inflate_upper_bound`），无额外绝对余量。
