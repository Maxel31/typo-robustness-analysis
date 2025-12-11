"""ベンチマーク評価モジュール.

各ベンチマークに対する評価関数を提供.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from src.models.inference import InferenceResult, MMLUInferenceResult
from src.utils.logger import logger


@dataclass
class EvaluationResult:
    """評価結果を保持するデータクラス."""

    total_samples: int
    correct_samples: int
    accuracy: float
    per_sample_results: list[dict[str, Any]] = field(default_factory=list)
    subtask_results: dict[str, dict[str, float]] | None = None


def extract_number_from_text(text: str) -> str | None:
    """テキストから数値を抽出（GSM8K評価用）.

    lm-eval-harnessのGSM8K評価を参考にした抽出方式.
    参考: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml

    抽出優先順位:
    1. #### <数値> パターン（GSM8K公式回答形式）
    2. The answer is <数値> パターン（strict-match方式）
    3. 最後の数値（flexible-extract方式、フォールバック）

    Args:
        text: 抽出対象のテキスト

    Returns:
        抽出された数値文字列（抽出できない場合はNone）
    """
    # 1. #### <数値> パターンを優先（GSM8K公式形式）
    # ####の直後に空白（改行以外）があり、その後に数値が続くパターン
    # 改行後の数値は別の文脈の可能性があるため除外
    hash_pattern = r"####[ \t]*(-?[\d,]+\.?\d*)"
    hash_match = re.search(hash_pattern, text)
    if hash_match and hash_match.group(1):  # 数値が空でないことを確認
        extracted = hash_match.group(1)
        normalized = extracted.replace(",", "")
        if normalized.endswith("."):
            normalized = normalized[:-1]
        return normalized if normalized else None

    # 2. The answer is <数値> パターン（strict-match方式）
    answer_pattern = r"[Tt]he answer is\s*\$?(-?[\d,]+\.?\d*)"
    answer_match = re.search(answer_pattern, text)
    if answer_match:
        extracted = answer_match.group(1)
        normalized = extracted.replace(",", "")
        if normalized.endswith("."):
            normalized = normalized[:-1]
        return normalized if normalized else None

    # 3. flexible-extractパターン（フォールバック）
    # (-?[$0-9.,]{2,}): $や数字、カンマ、ピリオドを含む2文字以上の文字列
    # |(-?[0-9]+): または負の整数
    flex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(flex_pattern, text)
    if not matches:
        return None

    # 最後のマッチを選択（group_select: -1）
    last_match = matches[-1]
    extracted = last_match[0] if last_match[0] else last_match[1]

    if not extracted:
        return None

    # 正規化: $, カンマ, ピリオド（末尾）を除去
    normalized = extracted.replace("$", "").replace(",", "")
    if normalized.endswith("."):
        normalized = normalized[:-1]

    return normalized if normalized else None


def normalize_number(num_str: str) -> float | None:
    """数値文字列を正規化.

    Args:
        num_str: 数値文字列

    Returns:
        正規化された数値（変換できない場合はNone）
    """
    try:
        # カンマを除去
        cleaned = num_str.replace(",", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def evaluate_gsm8k(results: list[InferenceResult]) -> EvaluationResult:
    """GSM8Kベンチマークの評価.

    数値回答のExact Matchで評価.

    Args:
        results: 推論結果のリスト

    Returns:
        評価結果
    """
    correct = 0
    per_sample_results = []

    for result in results:
        # 期待される回答を正規化
        expected = normalize_number(result.expected_answer)

        # 生成されたテキストから数値を抽出
        extracted = extract_number_from_text(result.generated_text)
        predicted = normalize_number(extracted) if extracted else None

        # 比較（小数点以下を考慮）
        is_correct = False
        if expected is not None and predicted is not None:
            # 整数比較の場合
            if expected == int(expected) and predicted == int(predicted):
                is_correct = int(expected) == int(predicted)
            else:
                # 小数比較の場合は許容誤差を設定
                is_correct = abs(expected - predicted) < 1e-6

        if is_correct:
            correct += 1

        per_sample_results.append(
            {
                "id": result.example_id,
                "expected": result.expected_answer,
                "predicted": extracted,
                "is_correct": is_correct,
                "generated_text": result.generated_text[:200],  # 省略
            }
        )

    accuracy = correct / len(results) if results else 0.0

    return EvaluationResult(
        total_samples=len(results),
        correct_samples=correct,
        accuracy=accuracy,
        per_sample_results=per_sample_results,
    )


def extract_answer_bbh(text: str) -> str:
    """BBHの回答を抽出（lm-eval-harness方式）.

    lm-eval-harnessのBBH評価と同様の方式で回答を抽出.
    正規表現: (?<=the answer is )(.*)(?=.)
    参考: lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml

    Args:
        text: 抽出対象のテキスト

    Returns:
        抽出された回答
    """
    text = text.strip()

    # lm-eval-harness方式: "the answer is X." からXを抽出
    # 正規表現: (?<=the answer is )(.*)(?=.)
    # より柔軟なパターンで対応
    answer_match = re.search(r"[Tt]he answer is\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # フォールバック: 生成テキスト全体をそのまま返す
    return text


def extract_answer_mmlu(text: str) -> str:
    """MMLUの回答を抽出.

    多肢選択問題の回答（A, B, C, D）を抽出.

    Args:
        text: 抽出対象のテキスト

    Returns:
        抽出された回答（A, B, C, D）
    """
    text = text.strip()

    # "the answer is" パターン
    answer_match = re.search(r"[Tt]he answer is\s*[:\s]*\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # 括弧付きの形式 (A), (B), etc.
    match = re.search(r"\(([A-Da-d])\)", text)
    if match:
        return match.group(1).upper()

    # 単独の選択肢 A, B, C, D
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    # フォールバック
    return text


def extract_answer_from_text(text: str, benchmark_name: str) -> str:
    """テキストから回答を抽出.

    Args:
        text: 抽出対象のテキスト
        benchmark_name: ベンチマーク名

    Returns:
        抽出された回答
    """
    text = text.strip()

    if benchmark_name == "bbh":
        return extract_answer_bbh(text)

    if benchmark_name == "mmlu":
        return extract_answer_mmlu(text)

    # 日本語ベンチマーク
    if benchmark_name in ["jamp", "jnli", "niilc", "jsquad", "jcommonsenseqa"]:
        # 最初の行または最初の文を抽出
        lines = text.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            # 句点で区切る
            sentences = first_line.split("。")
            if sentences:
                return sentences[0].strip()

    return text.strip()


def normalize_bbh_answer(text: str) -> str:
    """BBHの回答を正規化.

    Args:
        text: 正規化対象のテキスト

    Returns:
        正規化された回答
    """
    text = text.strip()

    # True/False形式
    lower_text = text.lower()
    if lower_text in ["true", "false"]:
        return lower_text.capitalize()

    # 括弧付きの選択肢 (A), (B), etc. → A, B
    match = re.search(r"\(([A-Za-z])\)", text)
    if match:
        return match.group(1).upper()

    # 単独の選択肢 A, B, etc.
    match = re.match(r"^([A-Za-z])$", text)
    if match:
        return match.group(1).upper()

    return text


def evaluate_bbh(
    results: list[InferenceResult],
    examples: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    """BBHベンチマークの評価（lm-eval-harness方式）.

    lm-eval-harnessと同様の評価方式:
    1. "the answer is X." からXを抽出
    2. exact_matchでターゲットと比較

    Args:
        results: 推論結果のリスト
        examples: 元のサンプルデータ（subtask情報を含む）

    Returns:
        評価結果
    """
    correct = 0
    per_sample_results = []
    subtask_counts: dict[str, dict[str, int]] = {}

    for i, result in enumerate(results):
        # expected_answer: データセットのtarget（正解）
        expected = result.expected_answer.strip()
        # predicted: モデル出力から抽出した回答
        predicted = extract_answer_bbh(result.generated_text)

        # exact_match: 大文字小文字を無視して比較
        is_correct = expected.lower() == predicted.lower()

        if is_correct:
            correct += 1

        # サブタスク別の集計
        subtask = "unknown"
        if examples and i < len(examples):
            subtask = examples[i].get("subtask", "unknown")

        if subtask not in subtask_counts:
            subtask_counts[subtask] = {"correct": 0, "total": 0}
        subtask_counts[subtask]["total"] += 1
        if is_correct:
            subtask_counts[subtask]["correct"] += 1

        per_sample_results.append(
            {
                "id": result.example_id,
                "subtask": subtask,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "generated_text": result.generated_text[:200],
            }
        )

    accuracy = correct / len(results) if results else 0.0

    # サブタスク別の正解率を計算
    subtask_results = {}
    for subtask, counts in subtask_counts.items():
        subtask_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        subtask_results[subtask] = {
            "accuracy": subtask_accuracy,
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return EvaluationResult(
        total_samples=len(results),
        correct_samples=correct,
        accuracy=accuracy,
        per_sample_results=per_sample_results,
        subtask_results=subtask_results,
    )


def normalize_choice_answer(text: str) -> str:
    """多肢選択の回答を正規化（MMLU、BBH共通）.

    Args:
        text: 正規化対象のテキスト

    Returns:
        正規化された回答（A, B, C, D等）
    """
    text = text.strip().upper()

    # 括弧付きの選択肢 (A), (B), etc. → A, B
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)

    # 単独の選択肢 A, B, etc.
    match = re.match(r"^([A-Z])$", text)
    if match:
        return match.group(1)

    return text


def evaluate_mmlu(
    results: list[InferenceResult],
    examples: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    """MMLUベンチマークの評価.

    多肢選択問題のExact Matchで評価.

    Args:
        results: 推論結果のリスト
        examples: 元のサンプルデータ（subject情報を含む）

    Returns:
        評価結果
    """
    correct = 0
    per_sample_results = []
    subject_counts: dict[str, dict[str, int]] = {}

    for i, result in enumerate(results):
        # expected_answerを正規化（(A) → A）
        expected = normalize_choice_answer(result.expected_answer)
        predicted = extract_answer_from_text(result.generated_text, "mmlu")

        is_correct = expected == predicted

        if is_correct:
            correct += 1

        # 科目別の集計
        subject = "unknown"
        if examples and i < len(examples):
            subject = examples[i].get("subject", "unknown")

        if subject not in subject_counts:
            subject_counts[subject] = {"correct": 0, "total": 0}
        subject_counts[subject]["total"] += 1
        if is_correct:
            subject_counts[subject]["correct"] += 1

        per_sample_results.append(
            {
                "id": result.example_id,
                "subject": subject,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "generated_text": result.generated_text[:200],
            }
        )

    accuracy = correct / len(results) if results else 0.0

    # 科目別の正解率を計算
    subtask_results = {}
    for subject, counts in subject_counts.items():
        subject_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        subtask_results[subject] = {
            "accuracy": subject_accuracy,
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return EvaluationResult(
        total_samples=len(results),
        correct_samples=correct,
        accuracy=accuracy,
        per_sample_results=per_sample_results,
        subtask_results=subtask_results,
    )


def evaluate_mmlu_logprobs(
    results: list[MMLUInferenceResult],
    examples: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    """MMLUベンチマークの評価（ログ確率ベース）.

    lm-eval-harness方式: output_type=multiple_choice（選択肢のログ確率比較）
    最大ログ確率の選択肢を予測とし、期待回答と比較.

    Args:
        results: MMLU推論結果のリスト（MMLUInferenceResult）
        examples: 元のサンプルデータ（subject情報を含む）

    Returns:
        評価結果
    """
    correct = 0
    per_sample_results = []
    subject_counts: dict[str, dict[str, int]] = {}

    for i, result in enumerate(results):
        # expected_answerを正規化（(A) → A）
        expected = normalize_choice_answer(result.expected_answer)
        # predicted_answerはログ確率最大の選択肢
        predicted = result.predicted_answer

        is_correct = expected == predicted

        if is_correct:
            correct += 1

        # 科目別の集計
        subject = "unknown"
        if examples and i < len(examples):
            subject = examples[i].get("subject", "unknown")

        if subject not in subject_counts:
            subject_counts[subject] = {"correct": 0, "total": 0}
        subject_counts[subject]["total"] += 1
        if is_correct:
            subject_counts[subject]["correct"] += 1

        # ログ確率情報を含めた結果
        per_sample_results.append(
            {
                "id": result.example_id,
                "subject": subject,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "choice_logprobs": dict(zip(result.choices, result.choice_logprobs, strict=True)),
                "original_text": result.original_text[:200],
            }
        )

    accuracy = correct / len(results) if results else 0.0

    # 科目別の正解率を計算
    subtask_results = {}
    for subject, counts in subject_counts.items():
        subject_accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        subtask_results[subject] = {
            "accuracy": subject_accuracy,
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return EvaluationResult(
        total_samples=len(results),
        correct_samples=correct,
        accuracy=accuracy,
        per_sample_results=per_sample_results,
        subtask_results=subtask_results,
    )


def evaluate_japanese_benchmark(
    results: list[InferenceResult],
    benchmark_name: str,
) -> EvaluationResult:
    """日本語ベンチマークの評価.

    Args:
        results: 推論結果のリスト
        benchmark_name: ベンチマーク名

    Returns:
        評価結果
    """
    correct = 0
    per_sample_results = []

    for result in results:
        expected = result.expected_answer.strip()
        predicted = extract_answer_from_text(result.generated_text, benchmark_name)

        # 完全一致または部分一致で評価
        is_correct = expected == predicted or expected in predicted or predicted in expected

        if is_correct:
            correct += 1

        per_sample_results.append(
            {
                "id": result.example_id,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "generated_text": result.generated_text[:200],
            }
        )

    accuracy = correct / len(results) if results else 0.0

    return EvaluationResult(
        total_samples=len(results),
        correct_samples=correct,
        accuracy=accuracy,
        per_sample_results=per_sample_results,
    )


class BenchmarkEvaluator:
    """ベンチマーク評価クラス.

    各ベンチマークに応じた評価を統一的に行う.
    """

    def __init__(self, benchmark_name: str) -> None:
        """初期化.

        Args:
            benchmark_name: ベンチマーク名
        """
        self.benchmark_name = benchmark_name

    def evaluate(
        self,
        results: list[InferenceResult],
        examples: list[dict[str, Any]] | None = None,
    ) -> EvaluationResult:
        """推論結果を評価.

        Args:
            results: 推論結果のリスト
            examples: 元のサンプルデータ

        Returns:
            評価結果
        """
        logger.info(f"評価開始: {self.benchmark_name}, {len(results)}件")

        if self.benchmark_name == "gsm8k":
            evaluation = evaluate_gsm8k(results)
        elif self.benchmark_name == "bbh":
            evaluation = evaluate_bbh(results, examples)
        elif self.benchmark_name == "mmlu":
            evaluation = evaluate_mmlu(results, examples)
        elif self.benchmark_name in ["jamp", "jnli", "niilc", "jsquad", "jcommonsenseqa"]:
            evaluation = evaluate_japanese_benchmark(results, self.benchmark_name)
        else:
            logger.warning(f"未知のベンチマーク: {self.benchmark_name}, デフォルト評価を使用")
            evaluation = evaluate_japanese_benchmark(results, self.benchmark_name)

        correct = evaluation.correct_samples
        total = evaluation.total_samples
        logger.info(
            f"評価完了: {self.benchmark_name}, 正解率={evaluation.accuracy:.4f} ({correct}/{total})"
        )

        return evaluation
