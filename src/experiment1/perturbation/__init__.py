"""実験1: 摂動モジュール.

ランダム摂動およびベンチマーク摂動を提供する.
"""

# 既存のperturbationモジュールを再エクスポート
from src.perturbation.benchmark_perturbator import (
    BenchmarkPerturbator,
    OccurrencePerturbation,
    OriginalBenchmarkData,
    PerturbedBenchmarkData,
    PerturbedExample,
    generate_perturbed_data,
)
from src.perturbation.perturbator import (
    PerturbationOperation,
    PerturbationRecord,
    Perturbator,
    apply_perturbation_to_text,
    find_word_occurrences,
)

__all__ = [
    # perturbator.py
    "PerturbationOperation",
    "PerturbationRecord",
    "Perturbator",
    "apply_perturbation_to_text",
    "find_word_occurrences",
    # benchmark_perturbator.py
    "OccurrencePerturbation",
    "PerturbedExample",
    "PerturbedBenchmarkData",
    "OriginalBenchmarkData",
    "BenchmarkPerturbator",
    "generate_perturbed_data",
]
