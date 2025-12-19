"""ケーススタディ分析モジュール.

少数のサンプルを使って、摂動パターンによる
エントロピー変化を詳細に分析する.
初期分析や手法の検証に使用する.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarks.benchmark_loader import load_bbh, load_gsm8k, load_mmlu
from src.experiment2.entropy_analysis.entropy_calculator import (
    GenerationEntropyResult,
    compare_entropy_trajectories,
    generate_with_entropy,
)
from src.experiment2.pattern_perturbation.pattern_generator import (
    PatternType,
    apply_pattern_perturbation,
    generate_perturbation_mapping_table,
)
from src.experiment2.token_extraction.benchmark_token_extractor import (
    extract_question_text,
)
from src.utils.logger import logger

# パターン名の型定義
BenchmarkName = Literal["gsm8k", "bbh", "mmlu"]


@dataclass
class CaseStudySample:
    """ケーススタディのサンプル.

    Attributes:
        sample_id: サンプルID
        benchmark_name: ベンチマーク名
        original_text: 元のテキスト
        perturbed_texts: パターンごとの摂動テキスト
        entropy_results: パターンごとのエントロピー結果
        perturbation_details: 適用された摂動の詳細
    """

    sample_id: int
    benchmark_name: str
    original_text: str
    perturbed_texts: dict[str, str] = field(default_factory=dict)
    entropy_results: dict[str, GenerationEntropyResult] = field(default_factory=dict)
    perturbation_details: dict[str, list[dict]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "sample_id": self.sample_id,
            "benchmark_name": self.benchmark_name,
            "original_text": (
                self.original_text[:200] + "..."
                if len(self.original_text) > 200
                else self.original_text
            ),
            "perturbed_texts": {
                pattern: (text[:200] + "..." if len(text) > 200 else text)
                for pattern, text in self.perturbed_texts.items()
            },
            "entropy_comparison": compare_entropy_trajectories(self.entropy_results),
            "perturbation_details": self.perturbation_details,
        }


@dataclass
class CaseStudyResult:
    """ケーススタディ全体の結果.

    Attributes:
        benchmark_name: ベンチマーク名
        model_name: モデル名
        num_samples: サンプル数
        samples: 各サンプルの結果
        aggregate_statistics: 集計統計
    """

    benchmark_name: str
    model_name: str
    num_samples: int
    samples: list[CaseStudySample] = field(default_factory=list)
    aggregate_statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "metadata": {
                "benchmark_name": self.benchmark_name,
                "model_name": self.model_name,
                "num_samples": self.num_samples,
            },
            "aggregate_statistics": self.aggregate_statistics,
            "samples": [s.to_dict() for s in self.samples],
        }

    def compute_aggregate_statistics(self) -> None:
        """集計統計を計算."""
        if not self.samples:
            return

        patterns = ["original", "pattern1", "pattern2", "pattern3"]
        stats: dict[str, dict] = {p: {"mean_entropies": [], "max_entropies": []} for p in patterns}

        for sample in self.samples:
            for pattern in patterns:
                if pattern in sample.entropy_results:
                    result = sample.entropy_results[pattern]
                    stats[pattern]["mean_entropies"].append(result.mean_entropy)
                    stats[pattern]["max_entropies"].append(result.max_entropy)

        self.aggregate_statistics = {}
        for pattern, data in stats.items():
            if data["mean_entropies"]:
                self.aggregate_statistics[pattern] = {
                    "avg_mean_entropy": sum(data["mean_entropies"]) / len(data["mean_entropies"]),
                    "avg_max_entropy": sum(data["max_entropies"]) / len(data["max_entropies"]),
                    "num_samples": len(data["mean_entropies"]),
                }

        # パターン間の差分を計算
        if "original" in self.aggregate_statistics:
            original_mean = self.aggregate_statistics["original"]["avg_mean_entropy"]
            self.aggregate_statistics["differences"] = {}
            for pattern in ["pattern1", "pattern2", "pattern3"]:
                if pattern in self.aggregate_statistics:
                    pattern_mean = self.aggregate_statistics[pattern]["avg_mean_entropy"]
                    self.aggregate_statistics["differences"][pattern] = {
                        "avg_entropy_increase": pattern_mean - original_mean,
                        "avg_entropy_ratio": (
                            pattern_mean / original_mean if original_mean > 0 else float("inf")
                        ),
                    }


def load_benchmark_samples(
    benchmark_name: BenchmarkName,
    num_samples: int = 10,
    subset: str | None = None,
) -> list[dict]:
    """ベンチマークからサンプルをロード.

    Args:
        benchmark_name: ベンチマーク名
        num_samples: サンプル数
        subset: サブセット名（BBH/MMLUの場合）

    Returns:
        サンプルのリスト
    """
    if benchmark_name == "gsm8k":
        examples = load_gsm8k(split="test")
    elif benchmark_name == "bbh":
        if subset is None:
            subset = "boolean_expressions"  # デフォルトサブセット
        examples = load_bbh(subset=subset)
    elif benchmark_name == "mmlu":
        if subset is None:
            subset = "abstract_algebra"  # デフォルトサブセット
        examples = load_mmlu(subset=subset, split="test")
    else:
        raise ValueError(f"未対応のベンチマーク: {benchmark_name}")

    # サンプル数を制限
    return examples[:num_samples]


def run_case_study(
    benchmark_name: BenchmarkName,
    model_name: str,
    num_samples: int = 10,
    target_tokens: list[str] | None = None,
    subset: str | None = None,
    max_new_tokens: int = 50,
    output_path: Path | None = None,
    gpu_id: str = "0",
) -> CaseStudyResult:
    """ケーススタディを実行.

    Args:
        benchmark_name: ベンチマーク名
        model_name: モデル名
        num_samples: サンプル数
        target_tokens: 摂動対象トークンリスト（Noneで自動抽出）
        subset: サブセット名
        max_new_tokens: 生成する最大トークン数
        output_path: 出力パス
        gpu_id: 使用するGPU ID

    Returns:
        ケーススタディ結果
    """
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"ケーススタディ開始: {benchmark_name}, {model_name}, {num_samples} samples")
    logger.info(f"デバイス: {device}")

    # モデルとトークナイザーをロード
    from src.models.model_loader import SUPPORTED_MODELS

    if model_name in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_name].get("hf_name", model_name)
    else:
        hf_name = model_name

    logger.info(f"モデルをロード: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # サンプルをロード
    samples = load_benchmark_samples(benchmark_name, num_samples, subset)
    logger.info(f"{len(samples)} サンプルをロード")

    # ターゲットトークンが指定されていない場合、サンプルから抽出
    if target_tokens is None:
        # サンプルテキストから頻出単語を抽出（簡易版）
        import re
        from collections import Counter

        word_counter: Counter = Counter()
        for sample in samples:
            text = extract_question_text(sample, benchmark_name)
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            word_counter.update(words)

        # 上位20単語を使用
        target_tokens = [word for word, _ in word_counter.most_common(20)]
        logger.info(f"ターゲットトークン（自動抽出）: {target_tokens[:10]}...")

    # 摂動マッピングテーブルを生成
    mapping_table = generate_perturbation_mapping_table(
        target_tokens,
        require_all_patterns=False,  # ケーススタディでは部分的でもOK
        seed=42,
    )
    logger.info(f"摂動マッピング生成: {len(mapping_table)} tokens")

    # 結果を格納
    result = CaseStudyResult(
        benchmark_name=benchmark_name,
        model_name=model_name,
        num_samples=len(samples),
    )

    # 各サンプルを処理
    patterns: list[PatternType] = ["pattern1", "pattern2", "pattern3"]

    for idx, sample in enumerate(samples):
        logger.info(f"サンプル {idx + 1}/{len(samples)} を処理中...")

        # 問題テキストを抽出
        original_text = extract_question_text(sample, benchmark_name)

        case_sample = CaseStudySample(
            sample_id=idx,
            benchmark_name=benchmark_name,
            original_text=original_text,
        )

        # オリジナルテキストでエントロピーを計算
        try:
            original_result = generate_with_entropy(
                model=model,
                tokenizer=tokenizer,
                prompt=original_text,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            case_sample.entropy_results["original"] = original_result
        except Exception as e:
            logger.warning(f"オリジナル処理でエラー: {e}")
            continue

        # 各パターンで摂動を適用しエントロピーを計算
        for pattern in patterns:
            perturbed_text, applied = apply_pattern_perturbation(
                original_text, mapping_table, pattern
            )

            if not applied:
                # 摂動が適用されなかった場合はスキップ
                continue

            case_sample.perturbed_texts[pattern] = perturbed_text
            case_sample.perturbation_details[pattern] = applied

            try:
                perturbed_result = generate_with_entropy(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=perturbed_text,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                case_sample.entropy_results[pattern] = perturbed_result
            except Exception as e:
                logger.warning(f"パターン {pattern} 処理でエラー: {e}")

        result.samples.append(case_sample)

    # 集計統計を計算
    result.compute_aggregate_statistics()

    # 結果を出力
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"結果を保存: {output_path}")

    # サマリーを表示
    logger.info("=== ケーススタディ結果サマリー ===")
    for pattern, stats in result.aggregate_statistics.items():
        if pattern != "differences":
            logger.info(f"  {pattern}: 平均エントロピー={stats.get('avg_mean_entropy', 0):.4f}")

    if "differences" in result.aggregate_statistics:
        logger.info("  --- パターン間の差分 ---")
        for pattern, diff in result.aggregate_statistics["differences"].items():
            logger.info(
                f"  {pattern}: エントロピー増加={diff['avg_entropy_increase']:.4f} "
                f"(比率={diff['avg_entropy_ratio']:.4f})"
            )

    return result


def quick_analysis(
    model_name: str = "gemma-3-1b-it",
    num_samples: int = 5,
    gpu_id: str = "0",
) -> None:
    """クイック分析を実行（デバッグ・検証用）.

    Args:
        model_name: モデル名
        num_samples: サンプル数
        gpu_id: GPU ID
    """
    logger.info("=== クイック分析開始 ===")

    # GSM8Kで簡易テスト
    result = run_case_study(
        benchmark_name="gsm8k",
        model_name=model_name,
        num_samples=num_samples,
        max_new_tokens=30,
        gpu_id=gpu_id,
    )

    logger.info("=== クイック分析完了 ===")
    return result
