"""JET-Forest 基准评测套件 (与 matchms 性能与一致性对比)。"""

from jetf.benchmarks.adapter import (
    check_matchms_available,
    jetf_library_to_matchms,
    jetf_peaks_to_matchms,
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
)
from jetf.benchmarks.reporter import (
    build_benchmark_report_dict,
    format_pairwise_consistency_table,
    format_pairwise_throughput_table,
    format_retrieval_consistency_table,
    format_retrieval_throughput_table,
    generate_full_markdown_report,
    generate_json_report,
    save_json_report,
    save_csv_report,
)
from jetf.benchmarks.throughput import (
    PairwiseThroughputResult,
    RetrievalThroughputResult,
    benchmark_pairwise_throughput,
    benchmark_retrieval_throughput,
    benchmark_scalability,
)

__all__ = [
    "check_matchms_available",
    "jetf_peaks_to_matchms",
    "jetf_library_to_matchms",
    "BenchmarkDataset",
    "load_benchmark_dataset",
    "sample_query_spectra",
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
    "format_pairwise_consistency_table",
    "format_retrieval_consistency_table",
    "format_pairwise_throughput_table",
    "format_retrieval_throughput_table",
    "generate_full_markdown_report",
    "generate_json_report",
    "save_json_report",
    "save_csv_report",
    "build_benchmark_report_dict",
]
