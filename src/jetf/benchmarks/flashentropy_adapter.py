"""JET-Forest 与 FlashEntropySearch 间的数据结构适配器与基准评测引擎。

涵盖：
1. FlashEntropySearch 环境探测与动态导入 (check_flashentropy_available)
2. JETF 数据结构 (SpectrumPeaks, SpectrumMeta, PreprocessedLibrary) 与 FlashEntropy (2D float32 peaks: (n, 2)) 互转
3. 逐对谱打分算子 (score_flashentropy_pair)
4. FlashEntropy 基准引擎 (FlashEntropyBenchmarkEngine)，支持离线索引构建、单条 1-to-N 检索 (Open/Identity)、
   批量打分与数据转换/纯检索耗时拆解
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from jetf.types import SpectrumPeaks

if TYPE_CHECKING:
    from jetf.preprocessing import PreprocessedLibrary
    from jetf.types import SpectrumMeta


def check_flashentropy_available() -> None:
    """确保 FlashEntropySearch (ms_entropy) 库可用。

    优先尝试直接导入已安装的 ms_entropy 官方发行包；
    若未安装，则回退查找并添加项目根目录下的 FlashEntropySearch 源码路径。
    """
    try:
        import ms_entropy  # noqa: F401
        from ms_entropy import FlashEntropySearch  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path("D:/JETF/FlashEntropySearch/src"),
        repo_root / "FlashEntropySearch" / "src",
        Path.cwd() / "FlashEntropySearch" / "src",
    ]
    for c in candidates:
        if (c / "mimas").is_dir():
            s = str(c.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            break

    try:
        import ms_entropy  # noqa: F401
        from ms_entropy import FlashEntropySearch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "未检测到 FlashEntropySearch。请通过 `uv pip install ms_entropy` 安装高性能发行版，"
            "或确保 FlashEntropySearch 源码目录完整存在。"
        ) from exc


def jetf_peaks_to_flashentropy(
    peaks: SpectrumPeaks,
    meta: SpectrumMeta | None = None,
    use_raw_intensity: bool = False,
) -> dict:
    """将 JETF SpectrumPeaks 与 SpectrumMeta 转换为 FlashEntropy 所需的字典结构。

    返回格式:
    {
        "precursor_mz": float,
        "peaks": NDArray[np.float32] (shape: (n_peaks, 2)),
        "id": str,
    }
    """
    n = len(peaks.mass)
    if n == 0:
        peaks_2d = np.empty((0, 2), dtype=np.float32)
    else:
        if use_raw_intensity and peaks.norm > 0.0:
            intensities = np.asarray(peaks.intensity * peaks.norm, dtype=np.float32)
        else:
            intensities = np.asarray(peaks.intensity, dtype=np.float32)
        
        masses = np.asarray(peaks.mass, dtype=np.float32)
        valid = np.isfinite(masses) & np.isfinite(intensities) & (masses >= 0.0) & (intensities >= 0.0)
        if not np.any(valid):
            peaks_2d = np.empty((0, 2), dtype=np.float32)
        else:
            masses = masses[valid]
            intensities = intensities[valid]
            # FlashEntropy 要求谱内峰强度之和归一化为 1.0 (Sum == 1.0)
            tot_i = float(np.sum(intensities))
            if tot_i > 0.0 and np.isfinite(tot_i):
                intensities = intensities / tot_i
            else:
                intensities = np.zeros_like(intensities)
            peaks_2d = np.column_stack([masses, intensities])

    precursor_mz = 0.0
    ext_id = ""
    if meta is not None:
        if meta.precursor_mz is not None and np.isfinite(meta.precursor_mz):
            precursor_mz = float(meta.precursor_mz)
        ext_id = str(meta.external_id)

    return {
        "precursor_mz": precursor_mz,
        "peaks": peaks_2d,
        "id": ext_id,
    }


def flashentropy_to_jetf_peaks(
    peaks_2d: NDArray[np.float32] | Sequence[Sequence[float]],
) -> SpectrumPeaks:
    """将 FlashEntropy 2D 数组 (shape: (n_peaks, 2)) 转换回 JETF SpectrumPeaks 对象。"""
    arr = np.asarray(peaks_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 2:
        return SpectrumPeaks._create_unchecked(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            0.0,
        )

    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 0] >= 0.0) & (arr[:, 1] >= 0.0)
    if not np.any(valid):
        return SpectrumPeaks._create_unchecked(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            0.0,
        )

    mass = arr[valid, 0]
    intensity = arr[valid, 1]

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


def jetf_library_to_flashentropy(
    library: PreprocessedLibrary | Sequence[SpectrumPeaks],
    indices: Sequence[int] | None = None,
    use_raw_intensity: bool = False,
) -> list[dict]:
    """批量将 JETF PreprocessedLibrary 或谱列表转换为 FlashEntropy 字典列表。

    每个字典包含 'original_idx' 字段以维护与 JETF 库原始行号的映射。
    """
    if hasattr(library, "n_spectra") and hasattr(library, "peaks"):
        rows = range(library.n_spectra) if indices is None else indices
        result = []
        for r in rows:
            row_idx = int(r)
            meta = library.spectra[row_idx] if hasattr(library, "spectra") else None
            d = jetf_peaks_to_flashentropy(
                library.peaks.spectrum_at(row_idx),
                meta=meta,
                use_raw_intensity=use_raw_intensity,
            )
            d["original_idx"] = row_idx
            result.append(d)
        return result
    else:
        rows = range(len(library)) if indices is None else indices
        result = []
        for r in rows:
            row_idx = int(r)
            d = jetf_peaks_to_flashentropy(
                library[row_idx],
                meta=None,
                use_raw_intensity=use_raw_intensity,
            )
            d["original_idx"] = row_idx
            result.append(d)
        return result


def score_flashentropy_pair(
    p1: SpectrumPeaks | NDArray[np.float32],
    p2: SpectrumPeaks | NDArray[np.float32],
    ms2_tolerance_da: float = 0.02,
) -> float:
    """使用 FlashEntropy 的 calculate_entropy_similarity 计算单对质谱的光谱信息熵相似度。"""
    check_flashentropy_available()
    from ms_entropy import calculate_entropy_similarity

    if isinstance(p1, SpectrumPeaks):
        arr1 = np.column_stack([p1.mass, p1.intensity]).astype(np.float32)
    else:
        arr1 = np.asarray(p1, dtype=np.float32)

    if isinstance(p2, SpectrumPeaks):
        arr2 = np.column_stack([p2.mass, p2.intensity]).astype(np.float32)
    else:
        arr2 = np.asarray(p2, dtype=np.float32)

    if arr1.shape[0] == 0 or arr2.shape[0] == 0:
        return 0.0

    try:
        score = float(calculate_entropy_similarity(arr1, arr2, ms2_tolerance_in_da=ms2_tolerance_da))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


@dataclass(frozen=True)
class FlashEntropySearchResult:
    """单条 1-to-N 查询检索结果与耗时拆解。"""

    indices: NDArray[np.int64]       # 映射回原始参考库的谱索引 (按相似度降序)
    scores: NDArray[np.float64]      # 对应相似度得分
    prep_time_s: float               # 查询谱格式转换与准备耗时 (秒)
    score_time_s: float              # 纯倒排索引搜索与相似度计算耗时 (秒)
    total_time_s: float              # 总时延 (秒)

    def __iter__(self):
        """支持元组解构: scores, indices = engine.search_single(...)"""
        return iter((self.scores, self.indices))


@dataclass(frozen=True)
class FlashEntropyBatchResult:
    """批量查询检索结果与耗时汇总。"""

    indices: list[NDArray[np.int64]]
    scores: list[NDArray[np.float64]]
    prep_time_s: float
    score_time_s: float
    total_time_s: float
    n_queries: int


class FlashEntropyBenchmarkEngine:
    """FlashEntropySearch 基准测试执行引擎。

    核心特性:
    1. 一次性离线建库: 在初始化时预构建倒排索引，并记录构建耗时与常驻内存；
    2. 索引严格对齐: 自动追踪 FlashEntropy 排序后索引与原 JETF 库索引的对应映射；
    3. 场景覆盖: 原生支持全库开放检索 (Open Search) 与窄前体窗口检索 (Identity Search)；
    4. 耗时拆解: 将单次检索延迟精准拆解为 prep_time 与 pure_search_time。
    """

    def __init__(
        self,
        library: PreprocessedLibrary,
        indices: Sequence[int] | None = None,
        max_indexed_mz: float = 1500.00005,
        precursor_ions_removal_da: float | None = None,
        noise_threshold: float = 0.0,
        clean_spectra: bool = True,
    ) -> None:
        check_flashentropy_available()
        from ms_entropy import FlashEntropySearch

        self.library = library
        self.max_indexed_mz = max_indexed_mz
        self.precursor_ions_removal_da = precursor_ions_removal_da
        self.noise_threshold = noise_threshold
        self.clean_spectra = clean_spectra

        t_conv_start = time.perf_counter()
        raw_list = jetf_library_to_flashentropy(library, indices=indices)
        self.convert_time_s = time.perf_counter() - t_conv_start

        # 过滤极端空谱，避免 FlashEntropy 索引构建报错
        valid_list = [spec for spec in raw_list if len(spec["peaks"]) > 0]
        if not valid_list:
            raise ValueError("参考库中无可供 FlashEntropy 索引构建的有效谱图")

        self.engine = FlashEntropySearch()
        t_build_start = time.perf_counter()
        sorted_specs = self.engine.build_index(
            valid_list,
            max_indexed_mz=max_indexed_mz,
            precursor_ions_removal_da=precursor_ions_removal_da,
            noise_threshold=noise_threshold,
            min_ms2_difference_in_da=0.05,
            clean_spectra=clean_spectra,
        )
        self.build_time_s = time.perf_counter() - t_build_start

        # 建立 FlashEntropy 排序后顺序 -> 原库索引映射
        self.orig_indices = np.array(
            [spec.get("original_idx", i) for i, spec in enumerate(sorted_specs)],
            dtype=np.int64,
        )
        self.n_indexed = len(self.orig_indices)

    def search_single(
        self,
        query_peaks: SpectrumPeaks | NDArray[np.float32],
        precursor_mz: float | None = None,
        top_k: int = 10,
        mode: str = "open",
        ms1_tolerance_da: float = 0.02,
        ms2_tolerance_da: float = 0.02,
    ) -> FlashEntropySearchResult:
        """执行单条查询检索，返回 Top-K 结果与耗时拆解。"""
        t0 = time.perf_counter()
        if isinstance(query_peaks, SpectrumPeaks):
            if len(query_peaks.mass) == 0:
                raw_q = np.empty((0, 2), dtype=np.float32)
            else:
                raw_q = np.column_stack([query_peaks.mass, query_peaks.intensity]).astype(np.float32)
        else:
            raw_q = np.asarray(query_peaks, dtype=np.float32)

        if raw_q.shape[0] == 0:
            q_peaks_2d = np.empty((0, 2), dtype=np.float32)
        elif self.clean_spectra:
            q_peaks_2d = self.engine.clean_spectrum_for_search(
                precursor_mz=precursor_mz if precursor_mz is not None else 0.0,
                peaks=raw_q,
                precursor_ions_removal_da=self.precursor_ions_removal_da,
                noise_threshold=self.noise_threshold,
                min_ms2_difference_in_da=0.05,
            )
        else:
            tot = float(np.sum(raw_q[:, 1]))
            if tot > 0.0:
                raw_q = raw_q.copy()
                raw_q[:, 1] /= tot
            q_peaks_2d = raw_q

        t_prep = time.perf_counter() - t0

        t1 = time.perf_counter()
        if q_peaks_2d.shape[0] == 0:
            scores_arr = np.zeros(self.n_indexed, dtype=np.float32)
        else:
            if mode == "identity" and precursor_mz is not None:
                scores_arr = self.engine.identity_search(
                    precursor_mz=float(precursor_mz),
                    peaks=q_peaks_2d,
                    ms1_tolerance_in_da=ms1_tolerance_da,
                    ms2_tolerance_in_da=ms2_tolerance_da,
                )
            else:
                scores_arr = self.engine.open_search(
                    peaks=q_peaks_2d,
                    ms2_tolerance_in_da=ms2_tolerance_da,
                )

        t_score = time.perf_counter() - t1
        t_total = time.perf_counter() - t0

        # 提取 Top-K
        k = min(top_k, len(scores_arr))
        if k <= 0:
            top_indices = np.empty(0, dtype=np.int64)
            top_scores = np.empty(0, dtype=np.float64)
        else:
            if k < len(scores_arr):
                part = np.argpartition(scores_arr, -k)[-k:]
                sorted_sub = part[np.argsort(-scores_arr[part])]
            else:
                sorted_sub = np.argsort(-scores_arr)
            top_indices = self.orig_indices[sorted_sub]
            top_scores = scores_arr[sorted_sub].astype(np.float64)

        return FlashEntropySearchResult(
            indices=top_indices,
            scores=top_scores,
            prep_time_s=t_prep,
            score_time_s=t_score,
            total_time_s=t_total,
        )

    def score_batch(
        self,
        queries: Sequence[tuple[SpectrumPeaks, float | None] | SpectrumPeaks],
        top_k: int = 10,
        mode: str = "open",
        ms1_tolerance_da: float = 0.02,
        ms2_tolerance_da: float = 0.02,
    ) -> FlashEntropyBatchResult:
        """执行批量查询检索。"""
        t0 = time.perf_counter()
        indices_list: list[NDArray[np.int64]] = []
        scores_list: list[NDArray[np.float64]] = []
        tot_prep = 0.0
        tot_score = 0.0

        for q in queries:
            if isinstance(q, tuple):
                p, pmz = q
            else:
                p, pmz = q, None
            res = self.search_single(
                query_peaks=p,
                precursor_mz=pmz,
                top_k=top_k,
                mode=mode,
                ms1_tolerance_da=ms1_tolerance_da,
                ms2_tolerance_da=ms2_tolerance_da,
            )
            indices_list.append(res.indices)
            scores_list.append(res.scores)
            tot_prep += res.prep_time_s
            tot_score += res.score_time_s

        t_total = time.perf_counter() - t0
        return FlashEntropyBatchResult(
            indices=indices_list,
            scores=scores_list,
            prep_time_s=tot_prep,
            score_time_s=tot_score,
            total_time_s=t_total,
            n_queries=len(queries),
        )
