"""检索查询配置与离子模式策略。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from jetf.preprocessing import CORRECTNESS_V1
from jetf.scoring import SCORER_VERSIONED_ID
from jetf.types import DEFAULT_FRAGMENT_TOLERANCE_DA, IonMode, PrecursorWindow, SpectrumMeta


class SearchMode(str, Enum):
    """检索模式。"""

    TOP_K = "top_k"
    THRESHOLD = "threshold"


class IonModePolicy(str, Enum):
    """离子模式过滤匹配策略。"""

    EXACT = "exact"
    INCLUDE_UNKNOWN = "include_unknown"
    ANY = "any"


def ion_mode_passes(
    query_mode: IonMode, policy: IonModePolicy, spectrum_mode: IonMode
) -> bool:
    """判定目标离子模式在给定策略下是否通过。"""
    if policy is IonModePolicy.ANY:
        return True
    if policy is IonModePolicy.EXACT:
        return spectrum_mode == query_mode
    if policy is IonModePolicy.INCLUDE_UNKNOWN:
        return spectrum_mode == query_mode or spectrum_mode is IonMode.UNKNOWN
    raise ValueError(f"未知离子模式政策: {policy!r}")


def is_eligible(query: QueryConfig, spectrum_meta: SpectrumMeta) -> bool:
    """元数据资格判定 F(Q, P)：self-exclusion、离子模式、前体窗口。"""
    if query.exclude_spectrum_id is not None and spectrum_meta.external_id == query.exclude_spectrum_id:
        return False
    if not ion_mode_passes(query.ion_mode, query.ion_mode_policy, spectrum_meta.ion_mode):
        return False
    if query.precursor_window is not None and not query.precursor_window.contains(spectrum_meta.precursor_mz):
        return False
    return True


@dataclass(frozen=True)
class QueryConfig:
    """质谱开放式检索查询配置。"""

    mode: SearchMode
    k: int = 10
    threshold: float | None = None
    ion_mode: IonMode = IonMode.UNKNOWN
    ion_mode_policy: IonModePolicy = IonModePolicy.INCLUDE_UNKNOWN
    fragment_tolerance_da: float = DEFAULT_FRAGMENT_TOLERANCE_DA
    min_matched_peaks: int = 1
    exclude_spectrum_id: str | None = None
    precursor_window: PrecursorWindow | None = None
    preprocess_version: str = CORRECTNESS_V1.versioned_id
    scorer_version: str = SCORER_VERSIONED_ID
    snapshot_id: str = "default_snapshot"

    def __post_init__(self) -> None:
        if self.mode == SearchMode.TOP_K:
            if self.k <= 0:
                raise ValueError(f"Top-K 模式的 k 必须为正整数，得到 {self.k}")
        elif self.mode == SearchMode.THRESHOLD:
            if self.threshold is None:
                raise ValueError("Threshold 模式必须指定 threshold")
            if not math.isfinite(self.threshold):
                raise ValueError(f"门槛必须为有限数，得到 {self.threshold}")
        if self.fragment_tolerance_da <= 0.0 or not math.isfinite(self.fragment_tolerance_da):
            raise ValueError(f"片断容差必须为正有限数，得到 {self.fragment_tolerance_da}")
        if self.min_matched_peaks < 0:
            raise ValueError(f"min_matched_peaks 不能为负，得到 {self.min_matched_peaks}")
        if self.precursor_window is not None and not isinstance(self.precursor_window, PrecursorWindow):
            raise TypeError(
                f"precursor_window 必须为 PrecursorWindow 实例，得到 {type(self.precursor_window).__name__}"
            )
