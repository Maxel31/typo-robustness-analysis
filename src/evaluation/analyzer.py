"""結果分析モジュール.

摂動による性能低下の分析と正規化を行う.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.evaluator import EvaluationResult
from src.utils.logger import logger


@dataclass
class WordImpactResult:
    """単語ごとの影響度結果."""

    target_word: str
    target_word_score: float  # 頻出度スコア
    baseline_accuracy: float  # ベースライン精度
    perturbed_accuracy: float  # 摂動後精度
    accuracy_drop: float  # 精度低下量
    total_occurrences: int  # 総出現回数
    perturbed_occurrences: int  # 摂動された出現回数
    num_examples: int  # サンプル数
    normalized_impact: float  # 正規化された影響度
    per_perturbation_impact: float  # 1回の摂動あたりの影響度


@dataclass
class AnalysisResult:
    """分析結果全体."""

    model_name: str
    benchmark_name: str
    baseline_accuracy: float
    word_impacts: list[WordImpactResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_normalized_impact(
    baseline_accuracy: float,
    perturbed_accuracy: float,
    perturbed_occurrences: int,
    total_occurrences: int,
    num_examples: int,
    normalization_method: str = "per_perturbation",
) -> tuple[float, float]:
    """正規化された影響度を計算.

    摂動回数のばらつきを考慮した正規化を行う.

    Args:
        baseline_accuracy: ベースライン精度
        perturbed_accuracy: 摂動後精度
        perturbed_occurrences: 摂動された出現回数
        total_occurrences: 総出現回数
        num_examples: サンプル数
        normalization_method: 正規化手法
            - "per_perturbation": 1回の摂動あたりの精度低下
            - "per_example": 1サンプルあたりの精度低下
            - "per_occurrence": 1出現あたりの精度低下
            - "log_perturbation": 摂動回数の対数で正規化
            - "log_occurrence": 総出現回数の対数で正規化

    Returns:
        (正規化された影響度, 1回の摂動あたりの影響度)
    """
    # 精度低下量
    accuracy_drop = baseline_accuracy - perturbed_accuracy

    # 摂動がない場合は0
    if perturbed_occurrences == 0:
        return 0.0, 0.0

    # 1回の摂動あたりの影響度
    per_perturbation_impact = accuracy_drop / perturbed_occurrences

    # 正規化手法に応じた計算
    if normalization_method == "per_perturbation":
        # 摂動1回あたりの精度低下
        normalized_impact = per_perturbation_impact
    elif normalization_method == "per_example":
        # サンプル1件あたりの精度低下
        normalized_impact = accuracy_drop / num_examples if num_examples > 0 else 0.0
    elif normalization_method == "per_occurrence":
        # 出現1回あたりの精度低下
        normalized_impact = accuracy_drop / total_occurrences if total_occurrences > 0 else 0.0
    elif normalization_method == "log_perturbation":
        # 摂動回数の対数で正規化 (log(1+x)で0回の場合も安全に処理)
        log_perturbations = math.log1p(perturbed_occurrences)
        normalized_impact = accuracy_drop / log_perturbations if log_perturbations > 0 else 0.0
    elif normalization_method == "log_occurrence":
        # 総出現回数の対数で正規化
        log_occurrences = math.log1p(total_occurrences)
        normalized_impact = accuracy_drop / log_occurrences if log_occurrences > 0 else 0.0
    else:
        # デフォルトは摂動1回あたり
        normalized_impact = per_perturbation_impact

    return normalized_impact, per_perturbation_impact


def analyze_perturbation_impact(
    baseline_result: EvaluationResult,
    perturbed_results: dict[str, tuple[EvaluationResult, dict[str, Any]]],
    model_name: str,
    benchmark_name: str,
    normalization_method: str = "per_perturbation",
) -> AnalysisResult:
    """摂動による影響度を分析.

    Args:
        baseline_result: ベースライン（摂動なし）の評価結果
        perturbed_results: 単語ごとの摂動評価結果
            キー: 単語, 値: (評価結果, メタデータ)
        model_name: モデル名
        benchmark_name: ベンチマーク名
        normalization_method: 正規化手法

    Returns:
        分析結果
    """
    logger.info(f"分析開始: {model_name} / {benchmark_name}")
    logger.info(f"ベースライン精度: {baseline_result.accuracy:.4f}")
    logger.info(f"分析対象単語数: {len(perturbed_results)}")

    word_impacts = []

    for target_word, (eval_result, metadata) in perturbed_results.items():
        # メタデータから情報を取得
        target_word_score = metadata.get("target_word_score", 0.0)
        total_occurrences = metadata.get("total_occurrences", 0)
        perturbed_occurrences = metadata.get("perturbed_occurrences", 0)
        num_examples = metadata.get("num_examples", eval_result.total_samples)

        # 正規化された影響度を計算
        normalized_impact, per_perturbation_impact = calculate_normalized_impact(
            baseline_accuracy=baseline_result.accuracy,
            perturbed_accuracy=eval_result.accuracy,
            perturbed_occurrences=perturbed_occurrences,
            total_occurrences=total_occurrences,
            num_examples=num_examples,
            normalization_method=normalization_method,
        )

        word_impact = WordImpactResult(
            target_word=target_word,
            target_word_score=target_word_score,
            baseline_accuracy=baseline_result.accuracy,
            perturbed_accuracy=eval_result.accuracy,
            accuracy_drop=baseline_result.accuracy - eval_result.accuracy,
            total_occurrences=total_occurrences,
            perturbed_occurrences=perturbed_occurrences,
            num_examples=num_examples,
            normalized_impact=normalized_impact,
            per_perturbation_impact=per_perturbation_impact,
        )
        word_impacts.append(word_impact)

    # 正規化された影響度で降順ソート
    word_impacts.sort(key=lambda x: x.normalized_impact, reverse=True)

    analysis = AnalysisResult(
        model_name=model_name,
        benchmark_name=benchmark_name,
        baseline_accuracy=baseline_result.accuracy,
        word_impacts=word_impacts,
        metadata={
            "normalization_method": normalization_method,
            "total_words_analyzed": len(word_impacts),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )

    logger.info(f"分析完了: {len(word_impacts)}単語")
    if word_impacts:
        logger.info(f"最も影響の大きい単語: {word_impacts[0].target_word}")
        logger.info(f"  - 正規化影響度: {word_impacts[0].normalized_impact:.6f}")
        logger.info(f"  - 精度低下: {word_impacts[0].accuracy_drop:.4f}")

    return analysis


def generate_ranking(
    analysis: AnalysisResult,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """影響度ランキングを生成.

    Args:
        analysis: 分析結果
        top_n: 上位N件（Noneの場合は全件）

    Returns:
        ランキングデータ
    """
    ranking = []

    impacts = analysis.word_impacts[:top_n] if top_n else analysis.word_impacts

    for rank, impact in enumerate(impacts, 1):
        ranking.append(
            {
                "rank": rank,
                "target_word": impact.target_word,
                "target_word_score": impact.target_word_score,
                "baseline_accuracy": impact.baseline_accuracy,
                "perturbed_accuracy": impact.perturbed_accuracy,
                "accuracy_drop": impact.accuracy_drop,
                "total_occurrences": impact.total_occurrences,
                "perturbed_occurrences": impact.perturbed_occurrences,
                "num_examples": impact.num_examples,
                "normalized_impact": impact.normalized_impact,
                "per_perturbation_impact": impact.per_perturbation_impact,
            }
        )

    return ranking


def save_analysis_result(
    analysis: AnalysisResult,
    output_dir: Path,
) -> Path:
    """分析結果を保存.

    Args:
        analysis: 分析結果
        output_dir: 出力ディレクトリ

    Returns:
        保存したファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"

    ranking = generate_ranking(analysis)

    result_data = {
        "metadata": {
            "model_name": analysis.model_name,
            "benchmark_name": analysis.benchmark_name,
            "baseline_accuracy": analysis.baseline_accuracy,
            "total_words_analyzed": len(analysis.word_impacts),
            "normalization_method": analysis.metadata.get("normalization_method"),
            "created_at": analysis.metadata.get("created_at"),
        },
        "ranking": ranking,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    logger.info(f"分析結果を保存: {output_path}")
    return output_path


def load_analysis_result(result_path: Path) -> dict[str, Any]:
    """保存された分析結果を読み込み.

    Args:
        result_path: 結果ファイルのパス

    Returns:
        分析結果データ
    """
    with open(result_path, encoding="utf-8") as f:
        return json.load(f)
