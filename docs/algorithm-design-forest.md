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
6. [双模态检索管线（Dual-Mode Search Pipeline）](#六双模态检索管线dual-mode-search-pipeline)
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
- **第一级（物理级硬分区）**：按前体质量（Precursor m/z）将全库切分为物理互不相交的轻量前体桶，根除跨前体污染；
- **第二级（族群级浅树）**：每个前体桶独立为一棵**两层浅树（Root 64 $\to$ Leaves 16）**。由于每棵树只覆盖化学前体相近的 64 条谱，根包络依然高度稀疏，**根节点整树排除率高达 30% ~ 86%**；
- **第三级（微观级致密叶）**：桶内采用图形学光线追踪的 **表面积启发式（Surface Area Heuristic, SAH）** 进行 0.02 Da 空间二分，将叶间包络重叠率砍半至 **10.9%**，包络体积压缩 **52.7%**，叶剪枝率突破至 **75% ~ 90%**。

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
  ├─ Leaf 0 (16 谱, 0.02 Da 紧致包络)
  ├─ Leaf 1 (16 谱, 0.02 Da 紧致包络)
  ├─ Leaf 2 (16 谱, 0.02 Da 紧致包络)
  └─ Leaf 3 (16 谱, 0.02 Da 紧致包络)
       │
       ▼ 每个叶节点直接对齐一个物理微块 (MicroBlock, 连续峰数据)
       └─ 执行 Block-Max 跳跃 + U_ind 单谱快速过滤 + Greedy Cosine 精评
```

---

## 三、核心数学定义与安全证明

### 3.1 质量网格与精评表示
- **摘要网格（Summary Grid）**：宽度 $\delta_e = 0.02$ Da。连续质量轴被划分为半开区间：
  $$C_h = [h \cdot \delta_e, (h+1) \cdot \delta_e), \quad h \in \mathbb{Z}$$
  峰 $m$ 所属 cell 下标为 $h = \lfloor m / \delta_e \rfloor$。
- **谱表示**：每条谱 $P$ 预处理后满足 $\sum_{j} v_j^2 = 1$，单谱 cell 能量为 $b_h(P) = \sum_{j \in C_h} v_j^2$。

### 3.2 节点包络摘要（Node Envelope）
对于任意节点 $B$（包含成员谱集合 $B \subset \mathcal{L}$），其包络摘要由两列等长数组构成（只存非零 cell）：
1. **支持集（Support Set）$Z(B)$**：
   $$Z(B) = \bigcup_{P \in B} \{ h : P \text{ 在 } C_h \text{ 内有非零峰} \}$$
2. **最大单峰幅度（Max Peak Amplitude）$m_h(B)$**：
   $$m_h(B) = (1 + \epsilon_{\text{mach}}) \cdot \max_{P \in B, j \in C_h} v_j, \quad \epsilon_{\text{mach}} = 10^{-12}$$
3. **最大 cell 能量（Max Cell Energy）$e_h(B)$**：
   $$e_h(B) = (1 + \epsilon_{\text{mach}}) \cdot \max_{P \in B} b_h(P)$$

### 3.3 节点级峰上界（Peak Bound）
设查询谱为 $Q = \{(m_i, u_i)\}$，匹配容差为 $\tau = 0.02$ Da。
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

### 4.2 最佳投影轴搜索与二分切分算法

```python
def split_sah_bvh(spectrum_indices, target_leaf_size=16):
    """
    输入: 当前桶内的谱序号列表 (N <= 64)
    输出: 切分出的叶节点列表，每个叶包含 ~16 条谱
    """
    if len(spectrum_indices) <= target_leaf_size:
        return [spectrum_indices]

    # Step 1: 统计子集内每个 0.02 Da cell 的谱出现频次
    cell_freq = count_cell_frequencies(spectrum_indices)
    N = len(spectrum_indices)

    # Step 2: 挑选“信息量/方差最大”的候选切分轴 (Top-10 cells)
    # 评分准则: f * (N - f)，当一半谱有峰、一半谱无峰时取得最大值
    candidate_axes = sorted(
        cell_freq.keys(), key=lambda c: cell_freq[c] * (N - cell_freq[c]), reverse=True
    )[:10]

    best_split = None
    min_total_cost = float("inf")
    half = N // 2

    # Step 3: 在候选轴上寻找最小 SAH Cost 的切分点
    for axis_cell in candidate_axes:
        # 按谱在该 cell 的峰强度升序排序 (无峰的强度为 0.0)
        sorted_indices = sort_by_intensity(spectrum_indices, cell=axis_cell)

        # 考虑容量平衡约束，仅考察中位数附近的切分位置 [half-4, half, half+4]
        for split_pos in [half - 4, half, half + 4]:
            if 4 <= split_pos <= N - 4:
                left_group = sorted_indices[:split_pos]
                right_group = sorted_indices[split_pos:]

                # 计算左右包络代价和
                cost_l = sah_cost(left_group)
                cost_r = sah_cost(right_group)

                if cost_l + cost_r < min_total_cost:
                    min_total_cost = cost_l + cost_r
                    best_split = (left_group, right_group)

    # 兜底: 若无特征候选轴，按默认前体微细排序直接均分
    if best_split is None:
        best_split = (spectrum_indices[:half], spectrum_indices[half:])

    # Step 4: 递归切分左右子集
    left_leaves = split_sah_bvh(best_split[0], target_leaf_size)
    right_leaves = split_sah_bvh(best_split[1], target_leaf_size)
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
│ 1. 分区元数据 (Partitions):                                             │
│    - partition_offsets: [0, M_pos, M_pos+M_neg, M_total] (int64)      │
│                                                                        │
│ 2. 森林树数组 (Trees, 长度 M):                                          │
│    - tree_precursor_min: [100.1, 150.3, ...] (float64, 桶前体下界)     │
│    - tree_precursor_max: [100.4, 150.8, ...] (float64, 桶前体上界)     │
│    - tree_root_node_id:  [0, 5, 10, ...]     (int64, 指向 Node 列)    │
│    - tree_leaf_offsets:  [0, 4, 8, 12, ...]  (int64, 对应叶段)        │
│    - tree_leaf_node_ids: [1, 2, 3, 4, 6, 7..](int64, 全部叶序号平铺)  │
│                                                                        │
│ 3. 节点数组 (Nodes, 包含全部 Root 和 Leaf):                             │
│    - is_leaf:            [False, True, True...](bool)                  │
│    - member_count:       [64, 16, 16, 16, 16...](int64)                │
│    - internal_id_start:  [0, 0, 16, 32, 48...] (int64)                 │
│    - internal_id_end:    [64, 16, 32, 48, 64...](int64)                │
│    - envelope_offsets:   [0, 850, 1200, ...]   (int64, 指向包络平铺列) │
│                                                                        │
│ 4. 平铺包络摘要 (Envelopes Columnar Buffer):                           │
│    - cell_index:         [5002, 5003, 8100...] (int64, 严格递增支持集) │
│    - max_peak_amplitude: [0.85, 0.12, 0.45...] (float64, 最大单峰幅度) │
│    - max_cell_energy:    [0.72, 0.01, 0.20...] (float64, 最大 cell 能量)│
│                                                                        │
│ 5. 微块对齐峰缓冲 (MicroBlock Postings Buffer):                         │
│    - 内部 ID 连续递增，每个 Leaf 对应一个物理微块 (MicroBlock)          │
│    - (mass, amplitude, energy, peak_id) 四列按成员内升序存储           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 六、双模态检索管线（Dual-Mode Search Pipeline）

JET-Forest 在检索入口处原生区分两种物理场景，消除无谓开销。

### 6.1 场景 A：身份检索（Identity Search，带前体窗口 $[M \pm \tau_{prec}]$）
这是 GNPS / MASST 生产环境中占比 $>90\%$ 的最核心场景。

```mermaid
flowchart TD
    Q[输入查询 Q 与前体窗口 [M ± tau]] --> BinarySearch[二分定位 Forest 的前体桶重叠区间]
    BinarySearch --> PruneDomain[直接排除 99% 的树! 仅留 1~2 棵候选树]
    PruneDomain --> ForTree[遍历幸存树 T_k]
    ForTree --> RootBound[计算根包络界 U_peak(Root)]
    RootBound --> CheckRoot{U_peak < theta?}
    CheckRoot -- 是 (整树排除) --> NextTree[检查下一棵树]
    CheckRoot -- 否 (展开叶子) --> ForLeaves[遍历该树的 4 个 SAH 紧致叶子]
    ForLeaves --> LeafBound[计算叶包络界 U_peak(Leaf)]
    LeafBound --> CheckLeaf{U_peak < theta?}
    CheckLeaf -- 是 (整叶跳过) --> NextLeaf[检查下一叶]
    CheckLeaf -- 否 (微块处理) --> MicroBlock[进入微块 Block-Max 跳跃]
    MicroBlock --> Uind[逐谱计算 U_ind 单谱快速过滤]
    Uind --> CheckUind{U_ind < theta?}
    CheckUind -- 是 --> SkipSpectrum[跳过评分]
    CheckUind -- 否 --> GreedyCosine[执行确定性 Greedy Cosine 精评]
    GreedyCosine --> UpdateHeap[更新 Top-k 堆与动态门槛 theta]
```

### 6.2 场景 B：开放检索 / 全局阈值检索（Open / Threshold Search，无前体限制）
针对全库相似性发现、未知修饰盲筛或分子网络构建。

1. **批量根求值（SIMD Batch Root Evaluation）**：
   - 森林中共有 $M$ 个根节点（以 16,000 谱全库为例，仅 $M \approx 250$ 个树根）；
   - 利用向量化连续内存，并行计算这 250 个根节点的 $U_{peak}(\text{Root})$；
2. **根级快速粗剪**：
   - 凡 $U_{peak}(\text{Root}) < \theta$ 的小树，连同其全部 4 个叶子及 64 条谱**瞬时排除**（实测秒杀率 $70\% \sim 86\%$）；
3. **最佳优先优先队列（Best-First Priority Queue）**：
   - 将幸存树的叶节点压入最大堆 $(U_{peak}(\text{Leaf}), \text{LeafID})$；
   - 优先展开界最高的高潜力叶子，促使 $\theta$ 极速抬升；
4. **叶内微块跳跃与 $U_{ind}$ 拦截**：
   - 沿用微块 $U_{ind}$ 剪枝，完成最终精评。

---

## 七、复杂度与存储开销分析

### 7.1 检索时间复杂度对比

| 检索阶段 | 全局单树 (M4 Global Tree) | 包络森林 (JET-Forest 身份搜索) | 包络森林 (JET-Forest 开放搜索) |
| :--- | :---: | :---: | :---: |
| **阶段 1：前体定位** | 无（需在遍历中逐节点判断） | **$O(\log M) \approx 5$ 次比较** | 无（全量扫描） |
| **阶段 2：候选树/节点筛选** | $O(f \cdot \text{Depth}) \approx 36$ 次求值 | **$O(1) \approx 1 \sim 2$ 次根求值** | **$O(M) \approx 250$ 次轻量向量点积** |
| **阶段 3：叶节点展开** | 深度下探（根部不剪枝） | **仅展开 $1 \sim 2$ 棵树的叶子 ($<8$ 次)** | **仅展开 $15\% \sim 30\%$ 幸存树的叶子** |
| **阶段 4：精评调用次数** | 很高（包络松，幸存多） | **极少（减少 60% 以上）** | **极少（减少 50% ~ 64%）** |

### 7.2 空间开销与存储模型（16,106 谱全库预估）
- **森林拓扑列**：250 棵树 $\times$ 4 叶 $\approx 1,250$ 个节点。节点元数据总计 $< 1$ MB；
- **包络三列平铺**：
  - 根节点包络：250 个 $\times$ 平均 15,000 cells $\approx 3.75 \times 10^6$ entries；
  - SAH 叶节点包络：1,000 个 $\times$ 平均 3,200 cells $\approx 3.20 \times 10^6$ entries；
  - 总平铺大小约 700 万条记录，按三列 (int64, float64, float64) 存储，占内存约 **150 MB**；
- **微块 Posting 峰缓冲**：全库 744 万峰，四列紧凑存储常驻约 **120 MB**；
- **总内存常驻**：$< 300$ MB，单机笔记本即可秒级全量加载！

---

## 八、工程实现规范与正确性纪律

1. **绝对一致性铁律（100% Exact Equivalency）**：
   - 无论开启身份搜索、开放搜索，无论 $\theta$ 门槛高低，系统检索输出的 `hits` 列表必须与 `search_exhaustive`（穷举基线）**逐位相同（Bit-exact Match）**。任何一处逻辑错误导致的假阴性均判定为重大 Bug。
2. **浮点结合律与余量纪律（Inflation Discipline）**：
   - 叶节点构建时，最大单峰幅度与 cell 能量必须调用 `inflate_upper_bounds(x)` 上偏一次（乘以 $1 + 10^{-12}$）；
   - 根节点合并叶节点时，只取 $\max$，**严禁重复上偏**；
   - 门槛判断严格执行 $U < \theta$，相等时不剪枝。
3. **微块内部 ID 单调性**：
   - 每一棵树内部、每一个叶节点内部，谱的内部 ID（`internal_id`）必须严格连续单调分配；
   - 确保微块范围切片为半开区间 `[id_start, id_end)`，零开销内存切片直接对齐底层 posting 数组。
