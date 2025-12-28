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
    """テキストから数値を抽出（GSM8K評価用、lm-eval-harness公式方式）.

    lm-eval-harness公式のGSM8K評価方式に完全準拠.
    参考: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml

    公式フィルタ:
    - strict-match: regex "The answer is (\\-?[0-9\\.\\,]+)."
    - flexible-extract: regex "(-?[$0-9.,]{2,})|(-?[0-9]+)" with group_select: -1

    Args:
        text: 抽出対象のテキスト

    Returns:
        抽出された数値文字列（抽出できない場合はNone）
    """
    # 公式strict-match: "The answer is (\-?[0-9\.\,]+)."
    # ピリオドで終わる形式を期待
    strict_pattern = r"[Tt]he answer is\s*\$?(-?[0-9.,]+)\."
    strict_match = re.search(strict_pattern, text)
    if strict_match:
        extracted = strict_match.group(1)
        # 正規化: カンマを除去
        normalized = extracted.replace(",", "")
        if normalized:
            return normalized

    # フォールバック: 公式flexible-extract
    # regex: "(-?[$0-9.,]{2,})|(-?[0-9]+)" with group_select: -1
    flex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(flex_pattern, text)
    if not matches:
        return None

    # 最後のマッチを選択（group_select: -1）
    last_match = matches[-1]
    extracted = last_match[0] if last_match[0] else last_match[1]

    if not extracted:
        return None

    # 正規化: $, カンマを除去
    normalized = extracted.replace("$", "").replace(",", "")

    return normalized if normalized else None


def normalize_gsm8k_answer(text: str) -> str:
    """GSM8K回答を正規化（lm-eval-harness公式方式）.

    lm-eval-harness公式のexact_match設定に準拠:
    - regexes_to_ignore: ["#### ", ",", "\\$", "\\."]
    - ignore_case: true

    Args:
        text: 正規化対象のテキスト

    Returns:
        正規化された文字列
    """
    if text is None:
        return ""
    # 公式のregexes_to_ignoreに従って除去
    normalized = text
    normalized = re.sub(r"#### ", "", normalized)  # #### を除去
    normalized = normalized.replace(",", "")  # カンマを除去
    normalized = normalized.replace("$", "")  # $を除去
    normalized = re.sub(r"\.$", "", normalized)  # 末尾のピリオドを除去
    normalized = normalized.strip().lower()  # 小文字化
    return normalized


def evaluate_gsm8k(results: list[InferenceResult]) -> EvaluationResult:
    """GSM8Kベンチマークの評価（lm-eval-harness公式方式）.

    lm-eval-harness公式のexact_match評価に完全準拠.
    参考: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml

    公式設定:
    - metric: exact_match
    - ignore_case: true
    - regexes_to_ignore: ["#### ", ",", "\\$", "\\."]

    Args:
        results: 推論結果のリスト

    Returns:
        評価結果
    """
    correct = 0
    per_sample_results = []

    for result in results:
        # 期待される回答を正規化（####後の数値）
        expected_raw = result.expected_answer
        expected = normalize_gsm8k_answer(expected_raw)

        # 生成されたテキストから数値を抽出して正規化
        extracted = extract_number_from_text(result.generated_text)
        predicted = normalize_gsm8k_answer(extracted) if extracted else ""

        # 公式のexact_match: 文字列比較
        is_correct = expected == predicted and expected != ""

        if is_correct:
            correct += 1

        per_sample_results.append(
            {
                "id": result.example_id,
                "expected": expected_raw,
                "expected_normalized": expected,
                "predicted": extracted,
                "predicted_normalized": predicted,
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
    """BBHの回答を抽出（lm-eval-harness公式方式）.

    lm-eval-harness公式のBBH評価方式に完全準拠.
    参考: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml

    公式フィルタ:
    - regex_pattern: "(?<=the answer is )(.*)(?=.)"
    - group_select: 0 (最初のマッチを使用)
    - fallback: "[invalid]"

    Args:
        text: 抽出対象のテキスト

    Returns:
        抽出された回答（マッチしない場合は"[invalid]"）
    """
    text = text.strip()

    # 公式正規表現: "(?<=the answer is )(.*)(?=.)"
    # Python re.findallで group_select=0 相当の動作を再現
    regex = re.compile(r"(?<=the answer is )(.*)(?=\.)", re.IGNORECASE)
    matches = regex.findall(text)

    if matches:
        # group_select=0: 最初のマッチを使用
        match = matches[0]
        if isinstance(match, tuple):
            match = [m for m in match if m]
            if match:
                match = match[0]
            else:
                return "[invalid]"
        return match.strip()

    # フォールバック: 公式では "[invalid]" を返す
    return "[invalid]"


def extract_answer_mmlu(text: str) -> str:
    """MMLUの回答を抽出（テキスト生成評価用）.

    MMLU-Pro (TIGER-AI-Lab) と openai/simple-evals のベストプラクティスに基づく
    多段階抽出戦略を使用。複数マッチがある場合は最後のマッチを採用。

    参考:
    - https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
    - https://github.com/openai/simple-evals/pull/34

    抽出順序:
    1. "answer is X" パターン（MMLU-Pro Stage 1）
    2. "Answer: X" パターン（MMLU-Pro Stage 2）
    3. 行頭の "X." パターン（選択肢直接引用）
    4. 最後の単独選択肢文字（MMLU-Pro Stage 3 フォールバック）

    Args:
        text: 抽出対象のテキスト

    Returns:
        抽出された回答（A-J、マッチしない場合は"[invalid]"）
    """
    # 前処理: マークダウン記法を除去（openai/simple-evals準拠）
    text = text.strip()
    text = re.sub(r"\*\*", "", text)  # **bold** を除去
    text = re.sub(r"\$\\boxed\{([^}]*)\}\$", r"\1", text)  # $\boxed{X}$ を X に
    text = re.sub(r"\$([^$]*)\$", r"\1", text)  # $X$ を X に

    # Stage 1: "answer is X" パターン（MMLU-Pro公式）
    # re.findallで全マッチを取得し、最後のマッチを採用
    matches = re.findall(
        r"[Aa]nswer\s+is[:\s]*\(?([A-Ja-j])\)?",
        text,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()

    # Stage 2: "Answer: X" パターン（MMLU-Pro公式）
    matches = re.findall(
        r"[Aa]nswer[:\s]+\(?([A-Ja-j])\)?",
        text,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()

    # Stage 3: 行頭の "X." パターン（例: "E. I and III"）
    # 選択肢を直接引用するモデル出力に対応
    matches = re.findall(
        r"(?:^|\n)\s*([A-Ja-j])\.\s",
        text,
        re.MULTILINE,
    )
    if matches:
        return matches[-1].upper()

    # Stage 4: 括弧付き選択肢 または 単一文字のみのテキスト（保守的フォールバック）
    # "I don't know" の "I" を誤抽出しないよう、より厳格な条件を使用

    # 4a: 括弧付きパターン: (A), (B) など
    matches = re.findall(r"\(([A-Ja-j])\)", text)
    if matches:
        return matches[-1].upper()

    # 4b: テキスト全体が単一選択肢文字の場合のみ
    stripped = text.strip()
    if len(stripped) == 1 and stripped.upper() in "ABCDEFGHIJ":
        return stripped.upper()

    # マッチしない場合は "[invalid]" を返す
    return "[invalid]"


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

    if benchmark_name == "truthfulqa":
        # TruthfulQAは生成テキストをそのまま正規化して返す
        return " ".join(text.lower().split())

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


def evaluate_bbh(
    results: list[InferenceResult],
    examples: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    """BBHベンチマークの評価（lm-eval-harness公式方式）.

    lm-eval-harness公式の評価方式に完全準拠:
    1. regex "(?<=the answer is )(.*)(?=.)" でフィルタリング
    2. exact_match（ignore_case, ignore_punctuation は無効）
    3. マッチしない場合は "[invalid]" → 不正解

    参考: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml

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
        # BBHのtargetは "So the answer is X." 形式なので、最終回答部分を抽出
        expected_raw = result.expected_answer.strip()
        expected = extract_answer_bbh(expected_raw)
        if expected == "[invalid]":
            # targetから抽出できない場合は、target全体を使用
            expected = expected_raw

        # predicted: モデル出力から抽出した回答
        predicted = extract_answer_bbh(result.generated_text)

        # exact_match: 公式設定では ignore_case, ignore_punctuation はコメントアウト
        # 厳密な文字列比較を行う
        is_correct = expected == predicted

        # "[invalid]" の場合は常に不正解
        if predicted == "[invalid]":
            is_correct = False

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
                "expected_raw": expected_raw,
                "expected": expected,
                "predicted": predicted,
                "is_correct": is_correct,
                "generated_text": result.generated_text[:500],
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
        """推論結果を評価（テキスト生成方式）.

        注意: MMLUの公式評価はログ確率方式（evaluate_mmlu_with_logprobs）を推奨.
        テキスト生成方式は非推奨だが、互換性のために残す.

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
            # 非推奨: テキスト生成方式。公式はログ確率方式を使用。
            logger.warning(
                "MMLU: テキスト生成方式を使用中。"
                "lm-eval-harness公式はログ確率方式（evaluate_mmlu_with_logprobs）を推奨。"
            )
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

    def evaluate_mmlu_with_logprobs(
        self,
        results: list[MMLUInferenceResult],
        examples: list[dict[str, Any]] | None = None,
    ) -> EvaluationResult:
        """MMLUをログ確率方式で評価（lm-eval-harness公式方式）.

        lm-eval-harness公式設定:
        - output_type: multiple_choice
        - metric: acc (選択肢のログ確率比較)

        参考: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml

        Args:
            results: MMLU推論結果のリスト（MMLUInferenceResult）
            examples: 元のサンプルデータ（subject情報を含む）

        Returns:
            評価結果
        """
        if self.benchmark_name != "mmlu":
            raise ValueError(
                f"evaluate_mmlu_with_logprobs はMMLU専用です。"
                f"現在のベンチマーク: {self.benchmark_name}"
            )

        logger.info(f"評価開始（ログ確率方式）: {self.benchmark_name}, {len(results)}件")

        evaluation = evaluate_mmlu_logprobs(results, examples)

        correct = evaluation.correct_samples
        total = evaluation.total_samples
        logger.info(
            f"評価完了: {self.benchmark_name}, 正解率={evaluation.accuracy:.4f} ({correct}/{total})"
        )

        return evaluation
