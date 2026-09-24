"""JET-Forest 基准评测套件 (与 matchms, BLINK, FlashEntropySearch 多算法性能与一致性对比)。"""

from jetf.benchmarks.adapter import (
    check_matchms_available,
    jetf_library_to_matchms,
    jetf_peaks_to_matchms,
)
from jetf.benchmarks.blink_accuracy import (
    ConfusionMatrixResult,
    PairwiseAccuracyResult,
    RetrievalAccuracyResult,
    evaluate_blink_confusion_matrix,
    evaluate_blink_pairwise_accuracy,
    evaluate_blink_retrieval_accuracy,
)
from jetf.benchmarks.blink_adapter import (
    BlinkBatchResult,
    BlinkBenchmarkEngine,
    BlinkSearchResult,
    blink_to_jetf_peaks,
    check_blink_available,
    jetf_library_to_blink,
    jetf_peaks_to_blink,
    score_blink_pair,
)
from jetf.benchmarks.blink_throughput import (
    ExtrapolationResult,
    ScalingBenchmarkResult,
    benchmark_blink_scaling,
    extrapolate_blink_throughput,
    measure_jetf_snapshot_throughput,
)
from jetf.benchmarks.consistency import (
    PairwiseConsistencyResult,
    QueryRetrievalConsistency,
    RetrievalConsistencySummary,
    evaluate_pairwise_consistency,
    evaluate_retrieval_consistency,
)
from jetf.benchmarks.dataset import (
    BenchmarkDataset,
    load_benchmark_dataset,
    sample_query_spectra,
    sample_query_spectra_from_forest,
    slice_parsed_library,
)
from jetf.benchmarks.flashentropy_accuracy import (
    EntropyCosineCorrelationResult,
    EntropyRetrievalAgreementResult,
    evaluate_entropy_cosine_correlation,
    evaluate_entropy_retrieval_agreement,
)
from jetf.benchmarks.flashentropy_adapter import (
    FlashEntropyBatchResult,
    FlashEntropyBenchmarkEngine,
    FlashEntropySearchResult,
    check_flashentropy_available,
    flashentropy_to_jetf_peaks,
    jetf_library_to_flashentropy,
    jetf_peaks_to_flashentropy,
    score_flashentropy_pair,
)
from jetf.benchmarks.reporter import (
    build_benchmark_report_dict,
    format_pairwise_consistency_table,
    format_pairwise_throughput_table,
    format_retrieval_consistency_table,
    format_retrieval_throughput_table,
    generate_json_report,
    save_csv_report,
    save_json_report,
)
from jetf.benchmarks.throughput import (
    PairwiseThroughputResult,
    RetrievalThroughputResult,
    benchmark_pairwise_throughput,
    benchmark_retrieval_throughput,
    benchmark_scalability,
)
from jetf.benchmarks.unified_runner import (
    IndexBuildStats,
    LatencyStats,
    MultiEngineBenchmarkRunner,
    UnifiedBenchmarkReport,
    UnifiedBenchmarkRunner,
    detect_available_engines,
)

__all__ = [
    # matchms 适配
    "check_matchms_available",
    "jetf_peaks_to_matchms",
    "jetf_library_to_matchms",
    # 数据集
    "BenchmarkDataset",
    "load_benchmark_dataset",
    "sample_query_spectra",
    "sample_query_spectra_from_forest",
    "slice_parsed_library",
    # matchms 一致性与吞吐
    "PairwiseConsistencyResult",
    "QueryRetrievalConsistency",
    "RetrievalConsistencySummary",
    "evaluate_pairwise_consistency",
    "evaluate_retrieval_consistency",
    "PairwiseThroughputResult",
    "RetrievalThroughputResult",
    "benchmark_pairwise_throughput",
    "benchmark_retrieval_throughput",
    "benchmark_scalability",
    # 报告与输出
    "format_pairwise_consistency_table",
    "format_retrieval_consistency_table",
    "format_pairwise_throughput_table",
    "format_retrieval_throughput_table",
    "generate_json_report",
    "save_json_report",
    "save_csv_report",
    "build_benchmark_report_dict",
    # BLINK 适配与评测
    "check_blink_available",
    "jetf_peaks_to_blink",
    "blink_to_jetf_peaks",
    "jetf_library_to_blink",
    "score_blink_pair",
    "BlinkBenchmarkEngine",
    "BlinkSearchResult",
    "BlinkBatchResult",
    "PairwiseAccuracyResult",
    "RetrievalAccuracyResult",
    "ConfusionMatrixResult",
    "ScalingBenchmarkResult",
    "ExtrapolationResult",
    "evaluate_blink_pairwise_accuracy",
    "evaluate_blink_retrieval_accuracy",
    "evaluate_blink_confusion_matrix",
    "benchmark_blink_scaling",
    "extrapolate_blink_throughput",
    "measure_jetf_snapshot_throughput",
    # FlashEntropySearch 适配与评测
    "check_flashentropy_available",
    "jetf_peaks_to_flashentropy",
    "flashentropy_to_jetf_peaks",
    "jetf_library_to_flashentropy",
    "score_flashentropy_pair",
    "FlashEntropyBenchmarkEngine",
    "FlashEntropySearchResult",
    "FlashEntropyBatchResult",
    "EntropyCosineCorrelationResult",
    "EntropyRetrievalAgreementResult",
    "evaluate_entropy_cosine_correlation",
    "evaluate_entropy_retrieval_agreement",
    # 统一多引擎调度
    "detect_available_engines",
    "MultiEngineBenchmarkRunner",
    "LatencyStats",
    "IndexBuildStats",
    "UnifiedBenchmarkReport",
]
