"""ベンチマーク内トークン頻度抽出モジュール.

各ベンチマークの問題文からトークン頻度を集計する.
サブワード単位/単語単位での集計に対応.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer, PreTrainedTokenizer

from src.benchmarks.benchmark_loader import (
    load_bbh,
    load_gsm8k,
    load_mmlu,
)
from src.utils.logger import logger


@dataclass
class TokenFrequency:
    """トークン頻度情報.

    Attributes:
        token: トークン文字列
        frequency: 出現頻度
        rank: 頻度ランク（1から開始）
        token_id: トークナイザーでのID（サブワード単位の場合のみ）
    """

    token: str
    frequency: int
    rank: int
    token_id: int | None = None


@dataclass
class TokenExtractionResult:
    """トークン抽出結果.

    Attributes:
        benchmark_name: ベンチマーク名
        model_name: 使用したモデル名（トークナイザー用）
        unit: 集計単位（subword/word）
        total_tokens: 総トークン数
        unique_tokens: ユニークトークン数
        tokens: 頻度順のトークンリスト
    """

    benchmark_name: str
    model_name: str
    unit: str
    total_tokens: int
    unique_tokens: int
    tokens: list[TokenFrequency] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "metadata": {
                "benchmark_name": self.benchmark_name,
                "model_name": self.model_name,
                "unit": self.unit,
                "total_tokens": self.total_tokens,
                "unique_tokens": self.unique_tokens,
            },
            "tokens": [
                {
                    "token": t.token,
                    "frequency": t.frequency,
                    "rank": t.rank,
                    "token_id": t.token_id,
                }
                for t in self.tokens
            ],
        }


def extract_question_text(example: dict, benchmark_name: str) -> str:
    """ベンチマーク例から問題文を抽出（選択肢を除外）.

    Args:
        example: ベンチマーク例
        benchmark_name: ベンチマーク名

    Returns:
        問題文（選択肢を除外）
    """
    if benchmark_name == "gsm8k":
        return example.get("question", "")

    elif benchmark_name == "bbh":
        # BBHはinputフィールドに問題文がある
        # 選択肢部分（Options:以降）を除外
        text = example.get("input", "")
        # Options: 以降を除外
        if "Options:" in text:
            text = text.split("Options:")[0]
        return text.strip()

    elif benchmark_name == "mmlu":
        # MMLUはquestionフィールドに問題文がある
        # choicesは別フィールドなので自動的に除外される
        return example.get("question", "")

    else:
        # デフォルト: questionまたはinputフィールドを使用
        return example.get("question", example.get("input", ""))


def tokenize_text_subword(
    text: str,
    tokenizer: PreTrainedTokenizer,
) -> list[tuple[str, int]]:
    """テキストをサブワード単位でトークナイズ.

    Args:
        text: 入力テキスト
        tokenizer: トークナイザー

    Returns:
        (トークン文字列, トークンID)のリスト
    """
    # トークナイズ
    encoding = tokenizer(text, add_special_tokens=False)
    token_ids = encoding["input_ids"]

    # トークンIDを文字列に変換
    tokens = []
    for token_id in token_ids:
        token_str = tokenizer.decode([token_id])
        tokens.append((token_str, token_id))

    return tokens


def tokenize_text_word(text: str) -> list[str]:
    """テキストを単語単位でトークナイズ.

    Args:
        text: 入力テキスト

    Returns:
        単語のリスト
    """
    # 英数字とアポストロフィのみを含む単語を抽出
    # 数字のみの単語は除外
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9']*\b", text.lower())
    return words


def extract_tokens_from_benchmark(
    benchmark_name: str,
    tokenizer: PreTrainedTokenizer | None = None,
    unit: Literal["subword", "word"] = "subword",
    subset: str | None = None,
) -> Counter:
    """ベンチマークからトークン頻度を抽出.

    Args:
        benchmark_name: ベンチマーク名（gsm8k, bbh, mmlu）
        tokenizer: トークナイザー（subword単位の場合必須）
        unit: 集計単位
        subset: サブセット名（BBH/MMLUの場合）

    Returns:
        トークン頻度のCounter
    """
    # ベンチマークをロード
    if benchmark_name == "gsm8k":
        examples = load_gsm8k(split="test")
    elif benchmark_name == "bbh":
        if subset is None:
            raise ValueError("BBHにはsubsetの指定が必要です")
        examples = load_bbh(subset=subset)
    elif benchmark_name == "mmlu":
        if subset is None:
            raise ValueError("MMLUにはsubsetの指定が必要です")
        examples = load_mmlu(subset=subset, split="test")
    else:
        raise ValueError(f"未対応のベンチマーク: {benchmark_name}")

    logger.info(f"ベンチマーク '{benchmark_name}' から {len(examples)} 件をロード")

    # トークン頻度をカウント
    token_counter: Counter = Counter()

    for example in examples:
        # 問題文を抽出
        question_text = extract_question_text(example, benchmark_name)

        if unit == "subword":
            if tokenizer is None:
                raise ValueError("subword単位の集計にはtokenizerが必要です")
            tokens = tokenize_text_subword(question_text, tokenizer)
            for token_str, _ in tokens:
                # 空白のみ/特殊文字のみのトークンは除外
                if token_str.strip() and re.search(r"[a-zA-Z0-9]", token_str):
                    token_counter[token_str] += 1
        else:
            words = tokenize_text_word(question_text)
            token_counter.update(words)

    return token_counter


def extract_frequent_tokens(
    benchmark_name: str,
    model_name: str,
    top_n: int = 300,
    unit: Literal["subword", "word"] = "subword",
    subsets: list[str] | None = None,
    output_path: Path | None = None,
) -> TokenExtractionResult:
    """ベンチマークから頻出トークンを抽出.

    Args:
        benchmark_name: ベンチマーク名
        model_name: モデル名（トークナイザー取得用）
        top_n: 上位N件を抽出
        unit: 集計単位
        subsets: サブセット名のリスト（BBH/MMLUの場合、Noneで全サブセット）
        output_path: 出力ファイルパス

    Returns:
        トークン抽出結果
    """
    # トークナイザーをロード（subword単位の場合）
    tokenizer = None
    if unit == "subword":
        logger.info(f"トークナイザーをロード: {model_name}")
        # SUPPORTED_MODELSからHuggingFace名を取得
        from src.models.model_loader import SUPPORTED_MODELS

        if model_name in SUPPORTED_MODELS:
            hf_name = SUPPORTED_MODELS[model_name].get("hf_name", model_name)
        else:
            hf_name = model_name

        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)

    # サブセットの決定
    if benchmark_name == "bbh":
        if subsets is None:
            # BBHの全サブセット
            subsets = [
                "boolean_expressions",
                "causal_judgement",
                "date_understanding",
                "disambiguation_qa",
                "dyck_languages",
                "formal_fallacies",
                "geometric_shapes",
                "hyperbaton",
                "logical_deduction_five_objects",
                "logical_deduction_seven_objects",
                "logical_deduction_three_objects",
                "movie_recommendation",
                "multistep_arithmetic_two",
                "navigate",
                "object_counting",
                "penguins_in_a_table",
                "reasoning_about_colored_objects",
                "ruin_names",
                "salient_translation_error_detection",
                "snarks",
                "sports_understanding",
                "temporal_sequences",
                "tracking_shuffled_objects_five_objects",
                "tracking_shuffled_objects_seven_objects",
                "tracking_shuffled_objects_three_objects",
                "web_of_lies",
                "word_sorting",
            ]
    elif benchmark_name == "mmlu":
        if subsets is None:
            # MMLUの主要サブセット（全57カテゴリは多いので代表的なものを選択）
            subsets = [
                "abstract_algebra",
                "anatomy",
                "astronomy",
                "business_ethics",
                "clinical_knowledge",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_medicine",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "econometrics",
                "electrical_engineering",
                "elementary_mathematics",
                "formal_logic",
                "global_facts",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_european_history",
                "high_school_geography",
                "high_school_government_and_politics",
                "high_school_macroeconomics",
                "high_school_mathematics",
                "high_school_microeconomics",
                "high_school_physics",
                "high_school_psychology",
                "high_school_statistics",
                "high_school_us_history",
                "high_school_world_history",
                "human_aging",
                "human_sexuality",
                "international_law",
                "jurisprudence",
                "logical_fallacies",
                "machine_learning",
                "management",
                "marketing",
                "medical_genetics",
                "miscellaneous",
                "moral_disputes",
                "moral_scenarios",
                "nutrition",
                "philosophy",
                "prehistory",
                "professional_accounting",
                "professional_law",
                "professional_medicine",
                "professional_psychology",
                "public_relations",
                "security_studies",
                "sociology",
                "us_foreign_policy",
                "virology",
                "world_religions",
            ]
    else:
        subsets = [None]  # GSM8Kなどサブセットがない場合

    # トークン頻度を集計
    total_counter: Counter = Counter()

    for subset in subsets:
        try:
            counter = extract_tokens_from_benchmark(
                benchmark_name=benchmark_name,
                tokenizer=tokenizer,
                unit=unit,
                subset=subset,
            )
            total_counter.update(counter)
            if subset:
                logger.info(f"  {subset}: {sum(counter.values())} tokens")
        except Exception as e:
            logger.warning(f"サブセット '{subset}' の処理でエラー: {e}")
            continue

    # 頻度順にソート
    sorted_tokens = total_counter.most_common(top_n)

    # 結果を構築
    token_frequencies = []
    for rank, (token, freq) in enumerate(sorted_tokens, start=1):
        token_id = None
        if unit == "subword" and tokenizer:
            # トークンIDを取得
            ids = tokenizer.encode(token, add_special_tokens=False)
            if ids:
                token_id = ids[0]

        token_frequencies.append(
            TokenFrequency(
                token=token,
                frequency=freq,
                rank=rank,
                token_id=token_id,
            )
        )

    result = TokenExtractionResult(
        benchmark_name=benchmark_name,
        model_name=model_name,
        unit=unit,
        total_tokens=sum(total_counter.values()),
        unique_tokens=len(total_counter),
        tokens=token_frequencies,
    )

    logger.info(
        f"抽出完了: 総トークン数={result.total_tokens}, "
        f"ユニーク数={result.unique_tokens}, "
        f"上位{len(token_frequencies)}件を返却"
    )

    # 出力ファイルに保存
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"出力ファイルを保存: {output_path}")

    return result


def find_common_tokens(
    results: list[TokenExtractionResult],
    min_benchmarks: int | None = None,
) -> list[str]:
    """複数ベンチマーク間で共通のトークンを検索.

    Args:
        results: 各ベンチマークのトークン抽出結果
        min_benchmarks: 最低出現ベンチマーク数（Noneで全ベンチマーク）

    Returns:
        共通トークンのリスト（頻度順）
    """
    if not results:
        return []

    if min_benchmarks is None:
        min_benchmarks = len(results)

    # 各ベンチマークのトークンセットを取得
    token_sets = []
    for result in results:
        tokens = {t.token for t in result.tokens}
        token_sets.append(tokens)

    # 共通トークンをカウント
    token_count: Counter = Counter()
    for tokens in token_sets:
        token_count.update(tokens)

    # 指定数以上のベンチマークに出現するトークンを抽出
    common_tokens = [token for token, count in token_count.items() if count >= min_benchmarks]

    # 総頻度順にソート
    token_total_freq: dict[str, int] = {}
    for result in results:
        for t in result.tokens:
            if t.token in common_tokens:
                token_total_freq[t.token] = token_total_freq.get(t.token, 0) + t.frequency

    common_tokens.sort(key=lambda x: token_total_freq.get(x, 0), reverse=True)

    logger.info(
        f"共通トークン数: {len(common_tokens)} (min_benchmarks={min_benchmarks}/{len(results)})"
    )

    return common_tokens
