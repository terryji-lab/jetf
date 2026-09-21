"""JET-Forest (jetf): 前体自适应浅层包络森林与 BVH-SAH 紧致索引质谱检索引擎。"""

from __future__ import annotations

__version__ = "0.1.0"

from jetf.bounds import NodeEnvelope, QueryContext, batch_root_bounds, build_query_context, peak_bound
from jetf.builder import build_forest_index
from jetf.bvh_sah import sah_cost, split_sah_bvh
from jetf.exhaustive import search_exhaustive
from jetf.search import search_forest
from jetf.cleaning import (
    DEFAULT_CLEAN_CONFIG,
    MatchmsCleanConfig,
    clean_parsed_library,
    clean_spectrum_with_matchms,
)
from jetf.mgf import ParsedLibrary, RejectReason, RejectedRecord, parse_mgf
from jetf.preprocessing import (
    CORRECTNESS_V1,
    PreprocessedLibrary,
    PreprocessSpec,
    preprocess_library,
    preprocess_query,
)
from jetf.query import IonModePolicy, QueryConfig, SearchMode, is_eligible
from jetf.results import SearchHit, SearchOutcome, SearchStats
from jetf.scoring import (
    SCORER_VERSIONED_ID,
    GreedyCosineResult,
    score_greedy_cosine,
    single_spectrum_bound,
)
from jetf.serialization import load_forest_snapshot, save_forest_snapshot
from jetf.structure import (
    DEFAULT_FOREST_SPEC,
    ForestEnvelopes,
    ForestIndex,
    ForestNodes,
    ForestPartition,
    ForestPostings,
    ForestSpec,
    ForestTrees,
    ZeroEnergyMembers,
    check_forest_index,
)
from jetf.types import (
    DEFAULT_FRAGMENT_TOLERANCE_DA,
    ENERGY_DTYPE,
    INTENSITY_DTYPE,
    INTERNAL_ID_DTYPE,
    MASS_DTYPE,
    PEAK_ID_DTYPE,
    IonMode,
    PrecursorWindow,
    SourceRef,
    Spectrum,
    SpectrumMeta,
    SpectrumPeaks,
)

__all__ = [
    "__version__",
    "ForestSpec",
    "DEFAULT_FOREST_SPEC",
    "ForestPartition",
    "ForestTrees",
    "ForestNodes",
    "ForestEnvelopes",
    "ForestPostings",
    "ForestIndex",
    "ZeroEnergyMembers",
    "check_forest_index",
    "sah_cost",
    "split_sah_bvh",
    "build_forest_index",
    "search_forest",
    "search_exhaustive",
    "save_forest_snapshot",
    "load_forest_snapshot",
    "parse_mgf",
    "ParsedLibrary",
    "MatchmsCleanConfig",
    "DEFAULT_CLEAN_CONFIG",
    "clean_parsed_library",
    "clean_spectrum_with_matchms",
    "RejectedRecord",
    "RejectReason",
    "preprocess_library",
    "preprocess_query",
    "PreprocessedLibrary",
    "PreprocessSpec",
    "CORRECTNESS_V1",
    "score_greedy_cosine",
    "single_spectrum_bound",
    "GreedyCosineResult",
    "SCORER_VERSIONED_ID",
    "QueryConfig",
    "SearchMode",
    "IonModePolicy",
    "is_eligible",
    "SearchHit",
    "SearchOutcome",
    "SearchStats",
    "NodeEnvelope",
    "QueryContext",
    "build_query_context",
    "peak_bound",
    "batch_root_bounds",
    "IonMode",
    "PrecursorWindow",
    "SourceRef",
    "SpectrumMeta",
    "SpectrumPeaks",
    "Spectrum",
    "DEFAULT_FRAGMENT_TOLERANCE_DA",
    "MASS_DTYPE",
    "INTENSITY_DTYPE",
    "ENERGY_DTYPE",
    "INTERNAL_ID_DTYPE",
    "PEAK_ID_DTYPE",
]
