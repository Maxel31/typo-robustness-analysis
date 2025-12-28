"""ケーススタディ分析モジュール.

少数のサンプルを使って、摂動パターンによる
エントロピー変化を詳細に分析する.
初期分析や手法の検証に使用する.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarks.benchmark_loader import load_benchmark
from src.evaluation.evaluator import (
    extract_answer_bbh,
    extract_answer_mmlu,
    extract_number_from_text,
    normalize_gsm8k_answer,
)
from src.experiment2.entropy_analysis.entropy_calculator import (
    GenerationEntropyResult,
    compare_entropy_trajectories,
    generate_with_entropy,
    generate_with_entropy_batch,
)
from src.experiment2.pattern_perturbation.pattern_generator import (
    PatternType,
    apply_aligned_perturbations,
    load_mapping_table,
)
from src.experiment2.token_extraction.benchmark_token_extractor import (
    extract_question_text,
)
from src.utils.logger import logger

# パターン名の型定義
BenchmarkName = Literal["gsm8k", "bbh", "mmlu", "truthfulqa"]


def extract_answer(generated_text: str, benchmark_name: str) -> str:
    """ベンチマークに応じて生成テキストから回答を抽出.

    Args:
        generated_text: モデルの生成テキスト
        benchmark_name: ベンチマーク名

    Returns:
        抽出された回答
    """
    if benchmark_name == "gsm8k":
        # GSM8Kは数値を抽出して正規化
        extracted = extract_number_from_text(generated_text)
        return normalize_gsm8k_answer(extracted) if extracted else ""
    elif benchmark_name == "bbh":
        return extract_answer_bbh(generated_text)
    elif benchmark_name == "mmlu":
        # MMLUは生成テキストから回答を抽出（A/B/C/D）
        return extract_answer_mmlu(generated_text)
    elif benchmark_name == "truthfulqa":
        # TruthfulQAは生成テキストをそのまま使用（小文字化・空白正規化）
        return " ".join(generated_text.lower().split())
    else:
        # 不明なベンチマークはそのまま返す
        return generated_text.strip()


def evaluate_answer(
    predicted: str,
    expected: str,
    benchmark_name: str,
) -> bool:
    """予測回答と期待回答を比較して正解判定.

    Args:
        predicted: 抽出された予測回答
        expected: 期待される正解
        benchmark_name: ベンチマーク名

    Returns:
        正解かどうか
    """
    if benchmark_name == "gsm8k":
        # GSM8K: 正規化後の文字列比較
        expected_norm = normalize_gsm8k_answer(expected)
        return predicted == expected_norm and predicted != ""
    elif benchmark_name == "bbh":
        # BBH: 厳密な文字列比較、[invalid]は不正解
        if predicted == "[invalid]":
            return False
        # 期待回答から抽出
        expected_extracted = extract_answer_bbh(expected)
        if expected_extracted == "[invalid]":
            expected_extracted = expected.strip()
        return predicted == expected_extracted
    elif benchmark_name == "mmlu":
        # MMLU: A/B/C/D の比較
        if predicted == "[invalid]":
            return False
        # 期待回答を正規化
        expected_norm = expected.strip().upper()
        # 括弧を除去 (A) → A
        if expected_norm.startswith("(") and expected_norm.endswith(")"):
            expected_norm = expected_norm[1:-1]
        return predicted == expected_norm
    elif benchmark_name == "truthfulqa":
        # TruthfulQA: 正規化後の部分一致
        # best_answerに予測回答が含まれているかチェック
        expected_norm = " ".join(expected.lower().split())
        if not predicted or not expected_norm:
            return False
        # 予測回答が期待回答に含まれる、または期待回答が予測回答に含まれる
        return predicted in expected_norm or expected_norm in predicted
    else:
        # 不明なベンチマークは完全一致
        return predicted == expected.strip()


def check_generation_changed(
    original_text: str,
    perturbed_text: str,
) -> bool:
    """生成テキストがオリジナルから変化したかどうかを判定.

    生成確率は無視し、生成された文字列が同一かどうかのみで判定する.

    Args:
        original_text: オリジナル（摂動なし）の生成テキスト
        perturbed_text: 摂動後の生成テキスト

    Returns:
        変化した場合True（文字列が異なる場合）
    """
    # 文字列を直接比較（生成確率は無視）
    # 空白の正規化のみ行う
    original_normalized = " ".join(original_text.split())
    perturbed_normalized = " ".join(perturbed_text.split())
    return original_normalized != perturbed_normalized


def compute_mmlu_choice_logprobs(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompt: str,
    choices: list[str] | None = None,
) -> tuple[str, list[float]]:
    """MMLUの選択肢ログ確率を計算し、予測回答を返す.

    lm-eval-harness準拠のmultiple_choice方式.

    Args:
        model: 言語モデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト（質問と選択肢を含む）
        choices: 選択肢リスト（デフォルト: ["A", "B", "C", "D"]）

    Returns:
        (予測回答, 各選択肢のログ確率リスト)
    """
    if choices is None:
        choices = ["A", "B", "C", "D"]

    from src.experiment2.entropy_analysis.entropy_calculator import get_model_device

    device = get_model_device(model)
    logprobs_list = []

    with torch.no_grad():
        for choice in choices:
            # プロンプト + 選択肢を連結
            full_text = prompt + choice

            # トークナイズ
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
            ).to(device)

            # 選択肢トークンのIDを取得
            choice_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
            if not choice_ids:
                logprobs_list.append(float("-inf"))
                continue
            choice_token_id = choice_ids[0]

            # モデル出力を取得
            outputs = model(**inputs)
            logits = outputs.logits

            # プロンプトの最後の位置での選択肢トークンのログ確率
            last_logits = logits[0, -len(choice_ids) - 1, :]
            log_probs = torch.log_softmax(last_logits.float(), dim=-1)
            logprob = log_probs[choice_token_id].item()
            logprobs_list.append(logprob)

    # 最もログ確率が高い選択肢を予測として返す
    predicted_idx = logprobs_list.index(max(logprobs_list))
    predicted = choices[predicted_idx]

    return predicted, logprobs_list


@dataclass
class PatternEvaluationResult:
    """各パターンの評価結果.

    Attributes:
        predicted_answer: 抽出された予測回答
        expected_answer: 期待される正解
        is_correct: 正解かどうか
        generation_changed: オリジナルから生成文が変化したか（originalは常にFalse）
        original_is_correct: オリジナル（摂動なし）が正解だったか（originalパターン自体ではNone）
        choice_logprobs: MMLU用の選択肢ログ確率（MMLU以外ではNone）
    """

    predicted_answer: str
    expected_answer: str
    is_correct: bool
    generation_changed: bool
    original_is_correct: bool | None = None  # オリジナルの正誤（patternの場合のみ設定）
    choice_logprobs: list[float] | None = None  # MMLU用

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        result = {
            "predicted_answer": self.predicted_answer,
            "expected_answer": self.expected_answer,
            "is_correct": self.is_correct,
            "generation_changed": self.generation_changed,
        }
        if self.original_is_correct is not None:
            result["original_is_correct"] = self.original_is_correct
        if self.choice_logprobs is not None:
            result["choice_logprobs"] = self.choice_logprobs
        return result


@dataclass
class CaseStudySample:
    """ケーススタディのサンプル.

    Attributes:
        sample_id: サンプルID
        benchmark_name: ベンチマーク名
        original_text: 元のテキスト
        expected_answer: 期待される正解（ベンチマークから取得）
        perturbed_texts: パターンごとの摂動テキスト
        entropy_results: パターンごとのエントロピー結果
        perturbation_details: 適用された摂動の詳細
        evaluation_results: パターンごとの評価結果
        mmlu_logprobs: MMLU用のログ確率評価結果（パターン→(予測, ログ確率リスト)）
    """

    sample_id: int
    benchmark_name: str
    original_text: str
    expected_answer: str = ""
    perturbed_texts: dict[str, str | None] = field(default_factory=dict)
    entropy_results: dict[str, GenerationEntropyResult] = field(default_factory=dict)
    perturbation_details: dict[str, list[dict]] = field(default_factory=dict)
    evaluation_results: dict[str, PatternEvaluationResult] = field(default_factory=dict)
    mmlu_logprobs: dict[str, tuple[str, list[float]]] = field(default_factory=dict)

    def compute_evaluation_results(
        self,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """各パターンの評価結果を計算.

        entropy_resultsに存在するパターンに対して評価結果を計算する.
        MMLUの場合、モデルとトークナイザーが必要（ログ確率ベース評価）.

        Args:
            model: 言語モデル（MMLU評価用、MMLUでない場合は不要）
            tokenizer: トークナイザー（MMLU評価用、MMLUでない場合は不要）
        """
        if not self.entropy_results:
            return

        # オリジナルの生成テキストを取得（変化判定の基準）
        original_generated = ""
        if "original" in self.entropy_results:
            original_generated = self.entropy_results["original"].generated_text

        # オリジナルの正誤を先に計算
        original_is_correct: bool | None = None

        # MMLUの場合、ログ確率ベースで評価
        if self.benchmark_name == "mmlu" and model is not None and tokenizer is not None:
            # 各パターンのログ確率評価を実行
            for pattern in self.entropy_results.keys():
                if pattern == "original":
                    prompt = self.original_text
                else:
                    prompt = self.perturbed_texts.get(pattern)
                    if prompt is None:
                        continue

                predicted, logprobs = compute_mmlu_choice_logprobs(model, tokenizer, prompt)
                self.mmlu_logprobs[pattern] = (predicted, logprobs)

        # 評価結果を計算
        for pattern, entropy_result in self.entropy_results.items():
            generated_text = entropy_result.generated_text

            # 生成テキストから回答を抽出（全ベンチマーク共通）
            predicted = extract_answer(generated_text, self.benchmark_name)

            # MMLUの場合、ログ確率は参考情報として保持
            if self.benchmark_name == "mmlu" and pattern in self.mmlu_logprobs:
                _, logprobs = self.mmlu_logprobs[pattern]
                choice_logprobs = logprobs
            else:
                choice_logprobs = None

            # 正解判定
            is_correct = evaluate_answer(predicted, self.expected_answer, self.benchmark_name)

            # オリジナルの正誤を記録
            if pattern == "original":
                original_is_correct = is_correct
                generation_changed = False
                orig_correct_for_pattern = None  # originalパターン自体にはセットしない
            else:
                # 生成文変化判定（生成確率は無視、文字列のみで判定）
                generation_changed = check_generation_changed(original_generated, generated_text)
                orig_correct_for_pattern = original_is_correct

            self.evaluation_results[pattern] = PatternEvaluationResult(
                predicted_answer=predicted,
                expected_answer=self.expected_answer,
                is_correct=is_correct,
                generation_changed=generation_changed,
                original_is_correct=orig_correct_for_pattern,
                choice_logprobs=choice_logprobs,
            )

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        # 各パターンの生成文を抽出（entropy_resultsに存在するもののみ）
        generated_texts = {}
        for pattern, result in self.entropy_results.items():
            generated_texts[pattern] = result.generated_text

        # 摂動テキストを抽出（実際に処理されたもののみ）
        perturbed_texts_output = {}
        for pattern, text in self.perturbed_texts.items():
            if text is not None and pattern in self.entropy_results:
                perturbed_texts_output[pattern] = text

        # 評価結果を抽出
        evaluation_output = {}
        for pattern, eval_result in self.evaluation_results.items():
            if pattern in self.entropy_results:
                evaluation_output[pattern] = eval_result.to_dict()

        return {
            "sample_id": self.sample_id,
            "benchmark_name": self.benchmark_name,
            "original_text": self.original_text,
            "expected_answer": self.expected_answer,
            "perturbed_texts": perturbed_texts_output,
            "generated_texts": generated_texts,
            "evaluation_results": evaluation_output,
            "entropy_comparison": compare_entropy_trajectories(self.entropy_results),
            "perturbation_details": {
                k: v for k, v in self.perturbation_details.items() if k in self.entropy_results
            },
        }

    def has_all_patterns(self) -> bool:
        """全パターン（original + pattern1-3）が存在するか確認."""
        required_patterns = {"original", "pattern1", "pattern2", "pattern3"}
        return required_patterns.issubset(self.entropy_results.keys())


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

        # 評価結果の集計用
        eval_stats: dict[str, dict[str, int]] = {
            p: {"correct": 0, "total": 0, "generation_changed": 0} for p in patterns
        }

        for sample in self.samples:
            for pattern in patterns:
                if pattern in sample.entropy_results:
                    result = sample.entropy_results[pattern]
                    stats[pattern]["mean_entropies"].append(result.mean_entropy)
                    stats[pattern]["max_entropies"].append(result.max_entropy)

                # 評価結果を集計
                if pattern in sample.evaluation_results:
                    eval_result = sample.evaluation_results[pattern]
                    eval_stats[pattern]["total"] += 1
                    if eval_result.is_correct:
                        eval_stats[pattern]["correct"] += 1
                    if eval_result.generation_changed:
                        eval_stats[pattern]["generation_changed"] += 1

        self.aggregate_statistics = {}
        for pattern, data in stats.items():
            if data["mean_entropies"]:
                self.aggregate_statistics[pattern] = {
                    "avg_mean_entropy": sum(data["mean_entropies"]) / len(data["mean_entropies"]),
                    "avg_max_entropy": sum(data["max_entropies"]) / len(data["max_entropies"]),
                    "num_samples": len(data["mean_entropies"]),
                }

            # 評価結果を追加
            if eval_stats[pattern]["total"] > 0:
                total = eval_stats[pattern]["total"]
                correct = eval_stats[pattern]["correct"]
                changed = eval_stats[pattern]["generation_changed"]
                if pattern not in self.aggregate_statistics:
                    self.aggregate_statistics[pattern] = {}
                self.aggregate_statistics[pattern].update(
                    {
                        "accuracy": correct / total,
                        "correct_count": correct,
                        "generation_changed_count": changed,
                        "generation_changed_rate": changed / total,
                    }
                )

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

        # 分析カテゴリ別の集計（8カテゴリ）
        # オリジナル正誤 × 摂動後正誤 × 生成変化 = 2 × 2 × 2 = 8カテゴリ
        # 略称: orig=オリジナル, pert=摂動後, corr=正答, incorr=誤答
        #       unchanged=生成変化なし, changed=生成変化あり
        self.aggregate_statistics["analysis_categories"] = {}
        for pattern in ["pattern1", "pattern2", "pattern3"]:
            categories = {
                # オリジナル正解のケース
                "orig_correct_pert_correct_unchanged": 0,  # 元正解→変化なし→正解維持
                "orig_correct_pert_correct_changed": 0,  # 元正解→変化あり→正解維持
                "orig_correct_pert_incorrect_unchanged": 0,  # 元正解→変化なし→誤答化
                "orig_correct_pert_incorrect_changed": 0,  # 元正解→変化あり→誤答化
                # オリジナル不正解のケース
                "orig_incorrect_pert_correct_unchanged": 0,  # 元誤答→変化なし→正解化
                "orig_incorrect_pert_correct_changed": 0,  # 元誤答→変化あり→正解化
                "orig_incorrect_pert_incorrect_unchanged": 0,  # 元誤答→変化なし→誤答維持
                "orig_incorrect_pert_incorrect_changed": 0,  # 元誤答→変化あり→誤答維持
            }
            for sample in self.samples:
                if pattern not in sample.evaluation_results:
                    continue
                eval_result = sample.evaluation_results[pattern]

                # オリジナルの正誤（None の場合はスキップ）
                if eval_result.original_is_correct is None:
                    continue

                orig_correct = eval_result.original_is_correct
                pert_correct = eval_result.is_correct
                gen_changed = eval_result.generation_changed

                # カテゴリキーを構築
                orig_part = "orig_correct" if orig_correct else "orig_incorrect"
                pert_part = "pert_correct" if pert_correct else "pert_incorrect"
                change_part = "changed" if gen_changed else "unchanged"
                key = f"{orig_part}_{pert_part}_{change_part}"
                categories[key] += 1

            self.aggregate_statistics["analysis_categories"][pattern] = categories


def load_benchmark_samples(
    benchmark_name: BenchmarkName,
    num_samples: int | None = 10,
    subset: str | None = None,
    seed: int = 42,
    stratified: bool = True,
) -> list[dict]:
    """ベンチマークからサンプルをロード.

    Args:
        benchmark_name: ベンチマーク名
        num_samples: サンプル数（Noneの場合は全件使用）
                     MMLUでstratified=Trueの場合は各トピックからのサンプル数
        subset: サブセット名（BBH/MMLUの場合、現在は未使用）
        seed: 乱数シード（再現性のため）
        stratified: Trueの場合、MMLUでは各トピックからnum_samples件ずつサンプリング

    Returns:
        選択されたサンプルのリスト
    """
    import random
    from collections import defaultdict

    # load_benchmark関数を使用してロード
    # 注: 現在のload_benchmarkは全サブセットをロードする
    benchmark_data = load_benchmark(name=benchmark_name, max_samples=None)
    examples = benchmark_data.examples

    # num_samplesがNoneの場合は全件を返す
    if num_samples is None:
        logger.info(f"全件使用: {len(examples)}件")
        return examples

    # ランダムにサンプルを選択（再現性のためシードを設定）
    random.seed(seed)

    # MMLUの場合、各トピックから層別サンプリング
    if benchmark_name == "mmlu" and stratified:
        # トピック別にサンプルをグループ化
        samples_by_topic: dict[str, list[dict]] = defaultdict(list)
        for example in examples:
            # subjectフィールドにトピック名が格納されている（MMLULoader参照）
            topic = example.get("subject", "unknown")
            samples_by_topic[topic].append(example)

        # 各トピックからnum_samples件ずつサンプリング
        stratified_samples = []
        topics = sorted(samples_by_topic.keys())
        logger.info(f"MMLU層別サンプリング: {len(topics)}トピックから各{num_samples}件")

        for topic in topics:
            topic_samples = samples_by_topic[topic]
            if len(topic_samples) <= num_samples:
                stratified_samples.extend(topic_samples)
            else:
                stratified_samples.extend(random.sample(topic_samples, num_samples))

        logger.info(f"  合計サンプル数: {len(stratified_samples)}")
        return stratified_samples

    # 他のベンチマークは従来通りランダムサンプリング
    if len(examples) <= num_samples:
        return examples
    return random.sample(examples, num_samples)


def run_case_study(
    benchmark_name: BenchmarkName,
    model_name: str,
    num_samples: int | None = 10,
    mapping_file: Path | None = None,
    subset: str | None = None,
    max_new_tokens: int = 50,
    output_path: Path | None = None,
    gpu_id: str = "0",
    seed: int = 42,
    batch_size: int = 1,
    top_k_sampling: int = 1,
    require_all_patterns: bool = True,
    num_perturbations: int = 1,
    stratified: bool = True,
) -> CaseStudyResult:
    """ケーススタディを実行.

    Args:
        benchmark_name: ベンチマーク名
        model_name: モデル名
        num_samples: サンプル数（Noneの場合は全件使用）
                     MMLUでstratified=Trueの場合は各トピックからのサンプル数
        mapping_file: 事前生成された摂動マッピングファイル（必須）
        subset: サブセット名
        max_new_tokens: 生成する最大トークン数
        output_path: 出力パス
        gpu_id: 使用するGPU ID
        seed: 乱数シード（サンプル選択の再現性のため）
        batch_size: 推論時のバッチサイズ（デフォルト: 1）
        top_k_sampling: 生成時のtop-kサンプリング（1=greedy, 2以上=top-kからサンプリング）
        require_all_patterns: 全パターン（original+pattern1-3）が揃ったサンプルのみ出力
        num_perturbations: 1文あたりの摂動箇所数（全パターンで同じ箇所に摂動を適用）
        stratified: Trueの場合、MMLUでは各トピックからnum_samples件ずつサンプリング

    Returns:
        ケーススタディ結果
    """
    # 注: GPU設定(CUDA_VISIBLE_DEVICES)はスクリプト側でtorchインポート前に行う必要あり
    # ここでは設定済みのデバイスを確認するのみ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"ケーススタディ開始: {benchmark_name}, {model_name}, {num_samples} samples")
    logger.info(f"デバイス: {device} (GPU ID: {gpu_id})")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"Top-kサンプリング: {top_k_sampling}")

    # 摂動マッピングファイルの確認
    if mapping_file is None:
        raise ValueError(
            "摂動マッピングファイルが必要です。"
            "scripts/generate_perturbation_mapping.py で事前に生成してください。"
        )

    if not mapping_file.exists():
        raise FileNotFoundError(f"摂動マッピングファイルが見つかりません: {mapping_file}")

    # 摂動マッピングをロード
    mapping_table = load_mapping_table(mapping_file)
    logger.info(f"摂動マッピングをロード: {len(mapping_table)} tokens from {mapping_file}")

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
        torch_dtype=torch.bfloat16,  # Gemma 3等はbfloat16が必須（float16ではNaNが発生）
        device_map="auto",
        trust_remote_code=True,
    )

    # サンプルをロード
    samples = load_benchmark_samples(
        benchmark_name, num_samples, subset, seed=seed, stratified=stratified
    )
    logger.info(f"{len(samples)} サンプルをロード")

    # 結果を格納
    result = CaseStudyResult(
        benchmark_name=benchmark_name,
        model_name=model_name,
        num_samples=len(samples),
    )

    # 各サンプルを処理
    patterns: list[PatternType] = ["pattern1", "pattern2", "pattern3"]

    # 全プロンプトを収集してバッチ処理
    # 各サンプルに対して: original + pattern1 + pattern2 + pattern3 = 最大4プロンプト
    all_prompts: list[tuple[int, str, str]] = []  # (sample_idx, pattern_name, prompt)
    case_samples: list[CaseStudySample] = []
    perturbation_details_map: dict[tuple[int, str], list[dict]] = {}

    for idx, sample in enumerate(samples):
        original_text = extract_question_text(sample, benchmark_name)

        # 期待される回答を取得（ベンチマークごとにフィールド名が異なる可能性があるため柔軟に対応）
        expected_answer = sample.get("answer", sample.get("target", ""))
        if expected_answer is None:
            expected_answer = ""

        case_sample = CaseStudySample(
            sample_id=idx,
            benchmark_name=benchmark_name,
            original_text=original_text,
            expected_answer=str(expected_answer),
        )
        case_samples.append(case_sample)

        # オリジナルプロンプトを追加
        all_prompts.append((idx, "original", original_text))

        # 検索範囲の終了位置を取得（MMLUなどで選択肢を除外するため）
        search_end_position = sample.get("question_end_position")

        # 全パターンで同じ箇所に摂動を適用
        # num_perturbations=1の場合は1単語、2以上の場合は複数単語に摂動
        aligned_result = apply_aligned_perturbations(
            original_text,
            mapping_table,
            num_perturbations,
            search_end_position=search_end_position,
        )

        if aligned_result is not None:
            # 各パターンのプロンプトを追加
            for pattern in patterns:
                perturbed_text = aligned_result.perturbed_texts[pattern]
                all_prompts.append((idx, pattern, perturbed_text))
                case_sample.perturbed_texts[pattern] = perturbed_text
                perturbation_details_map[(idx, pattern)] = aligned_result.applied_perturbations[
                    pattern
                ]
        else:
            # 全パターンで摂動可能な単語が不足
            for pattern in patterns:
                case_sample.perturbed_texts[pattern] = None
                perturbation_details_map[(idx, pattern)] = []

    logger.info(f"合計 {len(all_prompts)} プロンプトを処理")

    # バッチ処理でエントロピーを計算
    if batch_size == 1:
        # バッチサイズ1の場合は従来の逐次処理
        for i, (sample_idx, pattern_name, prompt) in enumerate(all_prompts):
            logger.info(f"プロンプト {i + 1}/{len(all_prompts)} を処理中...")
            try:
                entropy_result = generate_with_entropy(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    device=None,  # 自動検出
                    top_k_sampling=top_k_sampling,
                    seed=seed,
                )
                case_samples[sample_idx].entropy_results[pattern_name] = entropy_result
                if pattern_name != "original":
                    case_samples[sample_idx].perturbation_details[pattern_name] = (
                        perturbation_details_map.get((sample_idx, pattern_name), [])
                    )
            except Exception as e:
                logger.warning(f"サンプル{sample_idx} {pattern_name} 処理でエラー: {e}")
    else:
        # バッチ処理
        total_batches = (len(all_prompts) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_prompts))
            batch_items = all_prompts[start_idx:end_idx]
            batch_prompts = [item[2] for item in batch_items]

            logger.info(
                f"バッチ {batch_idx + 1}/{total_batches} を処理中 "
                f"({len(batch_prompts)} プロンプト)..."
            )

            try:
                entropy_results = generate_with_entropy_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    max_new_tokens=max_new_tokens,
                    device=None,  # 自動検出
                    top_k_sampling=top_k_sampling,
                    seed=seed,
                )

                # 結果を対応するサンプルに割り当て
                for (sample_idx, pattern_name, _), entropy_result in zip(
                    batch_items, entropy_results, strict=True
                ):
                    case_samples[sample_idx].entropy_results[pattern_name] = entropy_result
                    if pattern_name != "original":
                        case_samples[sample_idx].perturbation_details[pattern_name] = (
                            perturbation_details_map.get((sample_idx, pattern_name), [])
                        )
            except Exception as e:
                logger.warning(f"バッチ {batch_idx + 1} 処理でエラー: {e}")
                # エラー時は逐次処理にフォールバック
                for sample_idx, pattern_name, prompt in batch_items:
                    try:
                        entropy_result = generate_with_entropy(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            device=None,
                            top_k_sampling=top_k_sampling,
                            seed=seed,
                        )
                        case_samples[sample_idx].entropy_results[pattern_name] = entropy_result
                        if pattern_name != "original":
                            case_samples[sample_idx].perturbation_details[pattern_name] = (
                                perturbation_details_map.get((sample_idx, pattern_name), [])
                            )
                    except Exception as e2:
                        logger.warning(f"サンプル{sample_idx} {pattern_name} 処理でエラー: {e2}")

    # 各サンプルの評価結果を計算
    # MMLUの場合、ログ確率ベースで評価するためモデルとトークナイザーが必要
    for case_sample in case_samples:
        case_sample.compute_evaluation_results(model=model, tokenizer=tokenizer)

    # サンプルをフィルタリング
    for case_sample in case_samples:
        if "original" not in case_sample.entropy_results:
            continue
        if require_all_patterns and not case_sample.has_all_patterns():
            logger.info(
                f"サンプル{case_sample.sample_id}をスキップ: "
                f"全パターンが揃っていません (存在: {list(case_sample.entropy_results.keys())})"
            )
            continue
        result.samples.append(case_sample)

    logger.info(f"有効サンプル数: {len(result.samples)} / {len(case_samples)}")

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
        if pattern not in ["differences", "analysis_categories"]:
            accuracy = stats.get("accuracy", 0)
            changed_rate = stats.get("generation_changed_rate", 0)
            logger.info(
                f"  {pattern}: 平均エントロピー={stats.get('avg_mean_entropy', 0):.4f}, "
                f"正解率={accuracy:.4f}, 生成変化率={changed_rate:.4f}"
            )

    if "differences" in result.aggregate_statistics:
        logger.info("  --- パターン間の差分 ---")
        for pattern, diff in result.aggregate_statistics["differences"].items():
            logger.info(
                f"  {pattern}: エントロピー増加={diff['avg_entropy_increase']:.4f} "
                f"(比率={diff['avg_entropy_ratio']:.4f})"
            )

    if "analysis_categories" in result.aggregate_statistics:
        logger.info("  --- 分析カテゴリ別集計（8カテゴリ） ---")
        for pattern, categories in result.aggregate_statistics["analysis_categories"].items():
            # オリジナル正解のケース
            orig_corr_unchanged = categories.get("orig_correct_pert_correct_unchanged", 0)
            orig_corr_changed = categories.get("orig_correct_pert_correct_changed", 0)
            orig_to_incorr_unchanged = categories.get("orig_correct_pert_incorrect_unchanged", 0)
            orig_to_incorr_changed = categories.get("orig_correct_pert_incorrect_changed", 0)
            # オリジナル不正解のケース
            orig_incorr_to_corr_unchanged = categories.get(
                "orig_incorrect_pert_correct_unchanged", 0
            )
            orig_incorr_to_corr_changed = categories.get("orig_incorrect_pert_correct_changed", 0)
            orig_incorr_unchanged = categories.get("orig_incorrect_pert_incorrect_unchanged", 0)
            orig_incorr_changed = categories.get("orig_incorrect_pert_incorrect_changed", 0)

            logger.info(f"  {pattern}:")
            logger.info(
                f"    元正解→正解維持: no_chg={orig_corr_unchanged}, chg={orig_corr_changed}"
            )
            logger.info(
                f"    元正解→誤答化: no_chg={orig_to_incorr_unchanged}, "
                f"chg={orig_to_incorr_changed}"
            )
            logger.info(
                f"    元誤答→正解化: no_chg={orig_incorr_to_corr_unchanged}, "
                f"chg={orig_incorr_to_corr_changed}"
            )
            logger.info(
                f"    元誤答→誤答維持: no_chg={orig_incorr_unchanged}, chg={orig_incorr_changed}"
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
