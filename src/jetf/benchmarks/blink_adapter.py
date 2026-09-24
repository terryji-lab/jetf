"""JET-Forest 与 BLINK 间的数据结构适配器与基准评测引擎。

涵盖：
1. BLINK 环境检测与动态导入 (check_blink_available)
2. JETF 数据结构 (SpectrumPeaks, PreprocessedLibrary) 与 BLINK (2D mzis: shape (2, n_peaks)) 互转
3. 逐对谱打分算子 (score_blink_pair)
4. BLINK 基准引擎 (BlinkBenchmarkEngine)，支持库预离散化、单条 1-to-N 检索、批量打分与离散化/纯打分耗时拆解
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from jetf.types import SpectrumPeaks

if TYPE_CHECKING:
    from jetf.preprocessing import PreprocessedLibrary


def check_blink_available() -> None:
    """确保 BLINK 库可被正确导入。必须将包含真实包的 blink 目录加入 sys.path 首位。"""
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path("D:/JETF/blink"),
        repo_root / "blink",
        Path.cwd() / "blink",
    ]
    for c in candidates:
        if (c / "blink" / "__init__.py").is_file():
            s = str(c.resolve())
            if s not in sys.path or sys.path[0] != s:
                if s in sys.path:
                    sys.path.remove(s)
                sys.path.insert(0, s)
            break

    if "blink" in sys.modules:
        mod = sys.modules["blink"]
        if getattr(mod, "__file__", None) is None:
            for k in list(sys.modules.keys()):
                if k == "blink" or k.startswith("blink."):
                    del sys.modules[k]

    try:
        import blink  # noqa: F401
        import blink.spectral_normalization  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "未检测到 BLINK 库。请确保 D:\\JETF\\blink 存在并包含完整的 BLINK 模块源码。"
        ) from exc


def jetf_peaks_to_blink(
    peaks: SpectrumPeaks,
    use_raw_intensity: bool = False,
) -> NDArray[np.float64]:
    """将 JETF SpectrumPeaks 转换为 BLINK 所需的 2D numpy 数组 (shape: (2, n_peaks))。

    第 0 行: 质荷比 mass (m/z)
    第 1 行: 峰强度 intensity
    """
    n = len(peaks.mass)
    if n == 0:
        return np.empty((2, 0), dtype=np.float64)

    if use_raw_intensity and peaks.norm > 0.0:
        intensities = np.asarray(peaks.intensity * peaks.norm, dtype=np.float64)
    else:
        intensities = np.asarray(peaks.intensity, dtype=np.float64)

    masses = np.asarray(peaks.mass, dtype=np.float64)
    return np.vstack([masses, intensities])


def blink_to_jetf_peaks(
    mzi: NDArray[np.float64],
) -> SpectrumPeaks:
    """将 BLINK 2D 数组 (shape: (2, n_peaks)) 转换回 JETF SpectrumPeaks 对象。"""
    mzi_arr = np.asarray(mzi, dtype=np.float64)
    if (
        mzi_arr.ndim != 2
        or mzi_arr.shape[0] != 2
        or mzi_arr.shape[1] == 0
        or mzi_arr.size == 0
    ):
        return SpectrumPeaks._create_unchecked(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            0.0,
        )

    # 过滤非有限数值 (NaN, Inf) 与负强度
    valid_cols = (
        np.isfinite(mzi_arr[0])
        & np.isfinite(mzi_arr[1])
        & (mzi_arr[0] >= 0.0)
        & (mzi_arr[1] >= 0.0)
    )
    if not np.any(valid_cols):
        return SpectrumPeaks._create_unchecked(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            0.0,
        )

    mass = mzi_arr[0, valid_cols]
    intensity = mzi_arr[1, valid_cols]

    # 保证 mass 升序排列
    if len(mass) > 1 and not np.all(mass[:-1] <= mass[1:]):
        order = np.argsort(mass)
        mass = mass[order]
        intensity = intensity[order]

    norm = float(np.linalg.norm(intensity))
    if norm > 0.0:
        normalized_intensity = intensity / norm
    else:
        normalized_intensity = np.zeros_like(intensity)
    energy = normalized_intensity**2
    peak_id = np.arange(len(mass), dtype=np.int64)
    return SpectrumPeaks._create_unchecked(mass, normalized_intensity, energy, peak_id, norm)


def jetf_library_to_blink(
    library: PreprocessedLibrary,
    indices: Sequence[int] | None = None,
    use_raw_intensity: bool = False,
) -> list[NDArray[np.float64]]:
    """批量将 JETF PreprocessedLibrary 中的谱转换为 BLINK 2D numpy 数组列表。"""
    rows = range(library.n_spectra) if indices is None else indices
    return [
        jetf_peaks_to_blink(
            library.peaks.spectrum_at(int(r)),
            use_raw_intensity=use_raw_intensity,
        )
        for r in rows
    ]


def score_blink_pair(
    p1: SpectrumPeaks | NDArray[np.float64],
    p2: SpectrumPeaks | NDArray[np.float64],
    tolerance: float = 0.01,
    bin_width: float = 0.001,
    intensity_power: float = 0.5,
) -> tuple[float, int]:
    """使用 BLINK 算子计算单对质谱的相似度得分与匹配峰数。

    返回值: (score, n_matched)
    边界保护: 遇到空谱或全零谱时安全返回 (0.0, 0)，绝不崩溃。
    """
    check_blink_available()
    from blink import blink as bk

    m1 = jetf_peaks_to_blink(p1) if isinstance(p1, SpectrumPeaks) else np.asarray(p1, dtype=np.float64)
    m2 = jetf_peaks_to_blink(p2) if isinstance(p2, SpectrumPeaks) else np.asarray(p2, dtype=np.float64)

    # 边界检测：维度与空谱
    if (
        m1.ndim != 2
        or m2.ndim != 2
        or m1.shape[0] != 2
        or m2.shape[0] != 2
        or m1.shape[1] == 0
        or m2.shape[1] == 0
    ):
        return 0.0, 0

    # 边界检测：非有限数值 (NaN, Inf)
    if not (np.all(np.isfinite(m1)) and np.all(np.isfinite(m2))):
        return 0.0, 0

    # 边界检测：全 0 或负强度
    if np.all(m1[1] <= 0) or np.all(m2[1] <= 0):
        return 0.0, 0

    try:
        disc = bk.discretize_spectra(
            [m1],
            [m2],
            [0.0],
            [0.0],
            tolerance=tolerance,
            bin_width=bin_width,
            intensity_power=intensity_power,
            trim_empty=False,
            remove_duplicates=False,
        )
        scores = bk.score_sparse_spectra(disc)
        score_val = float(scores["mzi"].toarray()[0, 0])
        count_val = int(round(float(scores["mzc"].toarray()[0, 0])))
        return max(0.0, min(1.0, score_val)), max(0, count_val)
    except Exception:
        return 0.0, 0


@dataclass(frozen=True)
class BlinkSearchResult:
    """单条 1-to-N 查询的打分与检索结果，包含耗时细粒度拆解。"""

    indices: NDArray[np.int64]
    scores: NDArray[np.float64]
    counts: NDArray[np.int64]
    discretize_time_s: float
    score_time_s: float
    total_time_s: float

    def __iter__(self):
        """支持元组解构: scores, indices = engine.search_single(...)"""
        return iter((self.scores, self.indices))


@dataclass(frozen=True)
class BlinkBatchResult:
    """批量查询检索结果与耗时汇总。"""

    indices: list[NDArray[np.int64]]
    scores: list[NDArray[np.float64]]
    counts: list[NDArray[np.int64]]
    discretize_time_s: float
    score_time_s: float
    total_time_s: float
    n_queries: int


class BlinkBenchmarkEngine:
    """BLINK 基准测试执行引擎。

    核心特性:
    1. 库预离散化 (Pre-discretization): 在初始化时预先构建库参考端稀疏矩阵 (CSC 格式)，
       消除重复预处理，与 JET-Forest 索引检索评测对齐。
    2. 精确耗时拆解: 将单次查询耗时拆解为 query_discretize_time 与 pure_score_time。
    3. 安全鲁棒: 妥善处理空谱、稀疏奇异值与越界质量。
    """

    def __init__(
        self,
        library: PreprocessedLibrary | Sequence[SpectrumPeaks] | Sequence[NDArray[np.float64]],
        tolerance: float = 0.01,
        bin_width: float = 0.001,
        intensity_power: float = 0.5,
        max_mz_daltons: float = 5000.0,
    ) -> None:
        check_blink_available()
        import blink.spectral_normalization as sn

        self.tolerance = float(tolerance)
        self.bin_width = float(bin_width)
        self.intensity_power = float(intensity_power)

        # 统一转为 2D 数组列表
        t0 = time.perf_counter()
        if hasattr(library, "peaks") and hasattr(library, "n_spectra"):
            mzis = jetf_library_to_blink(library)
        else:
            mzis = [
                jetf_peaks_to_blink(item) if isinstance(item, SpectrumPeaks) else np.asarray(item, dtype=np.float64)
                for item in library
            ]
        self.n_library = len(mzis)

        # 动态根据当前库实际最大 m/z 设定 max_bin（与 BLINK 原生 _calc_max_mz 行为一致）
        max_mz_found = 0.0
        for m in mzis:
            if m.ndim == 2 and m.shape[0] == 2 and m.shape[1] > 0 and np.all(np.isfinite(m)):
                max_mz_found = max(max_mz_found, float(np.max(m[0])))
        effective_max_mz = max(max_mz_found + 20.0, 1000.0)
        if max_mz_daltons is not None:
            effective_max_mz = min(effective_max_mz, float(max_mz_daltons))
        self.max_mz_daltons = effective_max_mz
        self.max_bin = int(self.max_mz_daltons / self.bin_width) + int(self.tolerance / self.bin_width) + 100

        # 预离散化库稀疏矩阵 (s2)
        valid_indices = [
            i
            for i, m in enumerate(mzis)
            if m.ndim == 2 and m.shape[0] == 2 and m.shape[1] > 0 and not np.all(m[1] <= 0) and np.all(np.isfinite(m))
        ]
        if valid_indices:
            valid_mzis = [mzis[i] for i in valid_indices]
            norm_res = sn._normalize_spectra(
                valid_mzis,
                bin_width=self.bin_width,
                intensity_power=self.intensity_power,
                trim_empty=False,
                remove_duplicates=False,
            )
            orig_spec_ids = np.array(valid_indices, dtype=np.int64)[norm_res["spec_ids"]]
            mz_bins = norm_res["mz_bins"]
            valid_bins = (mz_bins >= 0) & (mz_bins < self.max_bin)

            self.s2_mzi = sp.csc_matrix(
                (
                    norm_res["normalized_intensities"][valid_bins],
                    (mz_bins[valid_bins], orig_spec_ids[valid_bins]),
                ),
                shape=(self.max_bin, self.n_library),
                dtype=np.float64,
            )
            self.s2_mzc = sp.csc_matrix(
                (
                    norm_res["counts"][valid_bins],
                    (mz_bins[valid_bins], orig_spec_ids[valid_bins]),
                ),
                shape=(self.max_bin, self.n_library),
                dtype=np.float64,
            )
        else:
            self.s2_mzi = sp.csc_matrix((self.max_bin, self.n_library), dtype=np.float64)
            self.s2_mzc = sp.csc_matrix((self.max_bin, self.n_library), dtype=np.float64)

        # 显式清理临时谱列表以节省内存，避免大规模评测中的内存残留
        del mzis
        self.library_discretize_time_s = time.perf_counter() - t0

    def _discretize_single_query(
        self, query_peaks: SpectrumPeaks | NDArray[np.float64]
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """将单条查询谱离散化并按 tolerance 窗口扩展，生成 CSR 稀疏向量。"""
        import blink.data_binning as db
        import blink.spectral_normalization as sn

        mzi = (
            jetf_peaks_to_blink(query_peaks)
            if isinstance(query_peaks, SpectrumPeaks)
            else np.asarray(query_peaks, dtype=np.float64)
        )

        if (
            mzi.ndim != 2
            or mzi.shape[0] != 2
            or mzi.shape[1] == 0
            or mzi.size == 0
            or not np.all(np.isfinite(mzi))
            or np.all(mzi[1] <= 0)
        ):
            empty_vec = sp.csr_matrix((1, self.max_bin), dtype=np.float64)
            return empty_vec, empty_vec

        n_q = sn._normalize_spectra(
            [mzi],
            bin_width=self.bin_width,
            intensity_power=self.intensity_power,
            trim_empty=False,
            remove_duplicates=False,
        )
        db._network_kernel(n_q, self.tolerance, self.bin_width)

        mz_bins = n_q["mz_bins"]
        valid = (mz_bins >= 0) & (mz_bins < self.max_bin)

        q_i = sp.csr_matrix(
            (
                n_q["normalized_intensities"][valid],
                (n_q["spec_ids"][valid], mz_bins[valid]),
            ),
            shape=(1, self.max_bin),
            dtype=np.float64,
        )
        q_c = sp.csr_matrix(
            (
                n_q["counts"][valid],
                (n_q["spec_ids"][valid], mz_bins[valid]),
            ),
            shape=(1, self.max_bin),
            dtype=np.float64,
        )
        return q_i, q_c

    def search_single(
        self,
        query_peaks: SpectrumPeaks | NDArray[np.float64],
        top_k: int = 10,
    ) -> BlinkSearchResult:
        """执行单条查询对全库的 1-to-N 检索与打分，返回 Top-K 命中。"""
        # 1. 查询离散化阶段
        t0 = time.perf_counter()
        q_i, q_c = self._discretize_single_query(query_peaks)
        t_disc = time.perf_counter() - t0

        # 2. 纯稀疏矩阵乘法打分阶段
        t0 = time.perf_counter()
        scores_arr = (q_i @ self.s2_mzi).toarray()[0]
        counts_arr = (q_c @ self.s2_mzc).toarray()[0]
        t_score = time.perf_counter() - t0

        k = min(top_k, self.n_library)
        if k <= 0 or self.n_library == 0:
            return BlinkSearchResult(
                indices=np.empty(0, dtype=np.int64),
                scores=np.empty(0, dtype=np.float64),
                counts=np.empty(0, dtype=np.int64),
                discretize_time_s=t_disc,
                score_time_s=t_score,
                total_time_s=t_disc + t_score,
            )

        if self.n_library <= k:
            sorted_idx = np.argsort(scores_arr)[::-1]
        else:
            part_idx = np.argpartition(scores_arr, -k)[-k:]
            sorted_idx = part_idx[np.argsort(scores_arr[part_idx])[::-1]]

        top_indices = sorted_idx.astype(np.int64)
        top_scores = scores_arr[top_indices]
        top_counts = np.round(counts_arr[top_indices]).astype(np.int64)

        return BlinkSearchResult(
            indices=top_indices,
            scores=top_scores,
            counts=top_counts,
            discretize_time_s=t_disc,
            score_time_s=t_score,
            total_time_s=t_disc + t_score,
        )

    def score_batch(
        self,
        queries: Sequence[SpectrumPeaks] | Sequence[NDArray[np.float64]],
        top_k: int = 10,
    ) -> BlinkBatchResult:
        """执行多条查询批量打分与 Top-K 检索。"""
        n_queries = len(queries)
        if n_queries == 0:
            return BlinkBatchResult(
                indices=[],
                scores=[],
                counts=[],
                discretize_time_s=0.0,
                score_time_s=0.0,
                total_time_s=0.0,
                n_queries=0,
            )

        import blink.data_binning as db
        import blink.spectral_normalization as sn

        t0 = time.perf_counter()
        q_mzis = [
            jetf_peaks_to_blink(q) if isinstance(q, SpectrumPeaks) else np.asarray(q, dtype=np.float64)
            for q in queries
        ]

        valid_indices = [
            i
            for i, m in enumerate(q_mzis)
            if m.ndim == 2
            and m.shape[0] == 2
            and m.shape[1] > 0
            and not np.all(m[1] <= 0)
            and np.all(np.isfinite(m))
        ]

        if valid_indices:
            valid_mzis = [q_mzis[i] for i in valid_indices]
            norm_res = sn._normalize_spectra(
                valid_mzis,
                bin_width=self.bin_width,
                intensity_power=self.intensity_power,
                trim_empty=False,
                remove_duplicates=False,
            )
            db._network_kernel(norm_res, self.tolerance, self.bin_width)
            orig_spec_ids = np.array(valid_indices, dtype=np.int64)[norm_res["spec_ids"]]
            mz_bins = norm_res["mz_bins"]
            valid_bins = (mz_bins >= 0) & (mz_bins < self.max_bin)

            q_mat_i = sp.csr_matrix(
                (
                    norm_res["normalized_intensities"][valid_bins],
                    (orig_spec_ids[valid_bins], mz_bins[valid_bins]),
                ),
                shape=(n_queries, self.max_bin),
                dtype=np.float64,
            )
            q_mat_c = sp.csr_matrix(
                (
                    norm_res["counts"][valid_bins],
                    (orig_spec_ids[valid_bins], mz_bins[valid_bins]),
                ),
                shape=(n_queries, self.max_bin),
                dtype=np.float64,
            )
        else:
            q_mat_i = sp.csr_matrix((n_queries, self.max_bin), dtype=np.float64)
            q_mat_c = sp.csr_matrix((n_queries, self.max_bin), dtype=np.float64)

        t_disc = time.perf_counter() - t0

        t0 = time.perf_counter()
        all_scores = (q_mat_i @ self.s2_mzi).toarray()
        all_counts = (q_mat_c @ self.s2_mzc).toarray()
        t_score = time.perf_counter() - t0

        k = min(top_k, self.n_library)
        all_indices_res = []
        all_scores_res = []
        all_counts_res = []

        for i in range(n_queries):
            row_s = all_scores[i]
            row_c = all_counts[i]
            if k <= 0 or self.n_library == 0:
                all_indices_res.append(np.empty(0, dtype=np.int64))
                all_scores_res.append(np.empty(0, dtype=np.float64))
                all_counts_res.append(np.empty(0, dtype=np.int64))
                continue

            if self.n_library <= k:
                s_idx = np.argsort(row_s)[::-1]
            else:
                p_idx = np.argpartition(row_s, -k)[-k:]
                s_idx = p_idx[np.argsort(row_s[p_idx])[::-1]]

            top_idx = s_idx.astype(np.int64)
            all_indices_res.append(top_idx)
            all_scores_res.append(row_s[top_idx])
            all_counts_res.append(np.round(row_c[top_idx]).astype(np.int64))

        return BlinkBatchResult(
            indices=all_indices_res,
            scores=all_scores_res,
            counts=all_counts_res,
            discretize_time_s=t_disc,
            score_time_s=t_score,
            total_time_s=t_disc + t_score,
            n_queries=n_queries,
        )

    search_batch = score_batch
