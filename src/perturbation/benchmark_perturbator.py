"""ベンチマークデータに対する摂動生成モジュール.

ベンチマークデータに対して、対象単語ごとに摂動を適用し、
推論用のデータを生成・保存する.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.perturbation.perturbator import (
    PerturbationOperation,
    Perturbator,
)
from src.utils.logger import logger


@dataclass
class OccurrencePerturbation:
    """単一の出現に対する摂動情報."""

    occurrence_index: int
    start_position: int
    end_position: int
    original_word: str
    perturbed_word: str
    operations: list[PerturbationOperation]

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        return {
            "occurrence_index": self.occurrence_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "original_word": self.original_word,
            "perturbed_word": self.perturbed_word,
            "operations": [op.to_dict() for op in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OccurrencePerturbation":
        """辞書から復元."""
        return cls(
            occurrence_index=data["occurrence_index"],
            start_position=data["start_position"],
            end_position=data["end_position"],
            original_word=data["original_word"],
            perturbed_word=data["perturbed_word"],
            operations=[PerturbationOperation.from_dict(op) for op in data["operations"]],
        )


@dataclass
class PerturbedExample:
    """摂動が適用された単一のサンプル."""

    example_id: int
    seed: int
    original_text: str
    perturbed_text: str
    perturbations: list[OccurrencePerturbation] = field(default_factory=list)
    # 元のベンチマークデータの追加フィールド（answer等）
    extra_fields: dict[str, Any] = field(default_factory=dict)
    # このサンプル内での対象単語の総出現数
    total_occurrences_in_example: int = 0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        result = {
            "id": self.example_id,
            "seed": self.seed,
            "original_text": self.original_text,
            "perturbed_text": self.perturbed_text,
            "perturbations": [p.to_dict() for p in self.perturbations],
            "total_occurrences_in_example": self.total_occurrences_in_example,
            "perturbed_count_in_example": len(self.perturbations),
        }
        result.update(self.extra_fields)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerturbedExample":
        """辞書から復元."""
        known_keys = {
            "id",
            "seed",
            "original_text",
            "perturbed_text",
            "perturbations",
            "total_occurrences_in_example",
            "perturbed_count_in_example",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            example_id=data["id"],
            seed=data["seed"],
            original_text=data["original_text"],
            perturbed_text=data["perturbed_text"],
            perturbations=[OccurrencePerturbation.from_dict(p) for p in data["perturbations"]],
            extra_fields=extra,
            total_occurrences_in_example=data.get("total_occurrences_in_example", 0),
        )


@dataclass
class PerturbedBenchmarkData:
    """対象単語に対する摂動済みベンチマークデータ."""

    benchmark_name: str
    target_word: str
    language: str
    replace_prob: float
    insert_prob: float
    delete_prob: float
    base_seed: int
    examples: list[PerturbedExample] = field(default_factory=list)
    total_occurrences: int = 0  # データセット内の総出現数
    perturbed_occurrences: int = 0  # 摂動が適用された出現数
    target_word_score: float | None = None  # 頻出単語リストにおけるスコア

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        metadata = {
            "benchmark_name": self.benchmark_name,
            "target_word": self.target_word,
            "language": self.language,
            "replace_prob": self.replace_prob,
            "insert_prob": self.insert_prob,
            "delete_prob": self.delete_prob,
            "base_seed": self.base_seed,
            "num_examples": len(self.examples),
            "total_occurrences": self.total_occurrences,
            "perturbed_occurrences": self.perturbed_occurrences,
        }
        # スコアが設定されている場合のみ追加
        if self.target_word_score is not None:
            metadata["target_word_score"] = self.target_word_score
        return {
            "metadata": metadata,
            "examples": [ex.to_dict() for ex in self.examples],
        }

    def save(self, file_path: Path) -> None:
        """JSONファイルに保存."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"摂動データを保存: {file_path} ({len(self.examples)}件)")

    @classmethod
    def load(cls, file_path: Path) -> "PerturbedBenchmarkData":
        """JSONファイルから読み込み."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        metadata = data["metadata"]
        # 後方互換性: 古い形式ではperturbation_probを使用
        if "perturbation_prob" in metadata:
            replace_prob = metadata["perturbation_prob"]
            insert_prob = metadata["perturbation_prob"]
            delete_prob = metadata["perturbation_prob"]
        else:
            replace_prob = metadata["replace_prob"]
            insert_prob = metadata["insert_prob"]
            delete_prob = metadata["delete_prob"]
        return cls(
            benchmark_name=metadata["benchmark_name"],
            target_word=metadata["target_word"],
            language=metadata["language"],
            replace_prob=replace_prob,
            insert_prob=insert_prob,
            delete_prob=delete_prob,
            base_seed=metadata["base_seed"],
            examples=[PerturbedExample.from_dict(ex) for ex in data["examples"]],
            total_occurrences=metadata.get("total_occurrences", 0),
            perturbed_occurrences=metadata.get("perturbed_occurrences", 0),
            target_word_score=metadata.get("target_word_score"),
        )


@dataclass
class OriginalBenchmarkData:
    """元のベンチマークデータ."""

    benchmark_name: str
    language: str
    text_field: str  # テキストが含まれるフィールド名 (例: "question", "input")
    examples: list[dict[str, Any]] = field(default_factory=list)
    search_end_field: str | None = None  # 検索終了位置のフィールド名（選択肢を除外するため）

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        return {
            "metadata": {
                "benchmark_name": self.benchmark_name,
                "language": self.language,
                "text_field": self.text_field,
                "search_end_field": self.search_end_field,
                "total_examples": len(self.examples),
            },
            "examples": self.examples,
        }

    def save(self, file_path: Path) -> None:
        """JSONファイルに保存."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"元データを保存: {file_path} ({len(self.examples)}件)")

    @classmethod
    def load(cls, file_path: Path) -> "OriginalBenchmarkData":
        """JSONファイルから読み込み."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        metadata = data["metadata"]
        return cls(
            benchmark_name=metadata["benchmark_name"],
            language=metadata["language"],
            text_field=metadata["text_field"],
            examples=data["examples"],
            search_end_field=metadata.get("search_end_field"),
        )


class BenchmarkPerturbator:
    """ベンチマークデータに対する摂動を生成するクラス."""

    def __init__(
        self,
        benchmark_name: str,
        language: str = "english",
        replace_prob: float = 0.2,
        insert_prob: float = 0.2,
        delete_prob: float = 0.2,
        base_seed: int = 42,
    ) -> None:
        """初期化.

        Args:
            benchmark_name: ベンチマーク名
            language: 言語 ("english" or "japanese")
            replace_prob: 各文字に対する置換確率 (デフォルト: 0.2)
            insert_prob: 各文字に対する挿入確率 (デフォルト: 0.2)
            delete_prob: 各文字に対する削除確率 (デフォルト: 0.2)
            base_seed: 基本シード（各サンプルのシードはこれを基に生成）
        """
        self.benchmark_name = benchmark_name
        self.language = language
        self.replace_prob = replace_prob
        self.insert_prob = insert_prob
        self.delete_prob = delete_prob
        self.base_seed = base_seed

        logger.info(
            f"BenchmarkPerturbator初期化: benchmark={benchmark_name}, "
            f"language={language}, replace={replace_prob}, insert={insert_prob}, "
            f"delete={delete_prob}, base_seed={base_seed}"
        )

    def _find_word_positions(
        self,
        text: str,
        target_word: str,
        case_sensitive: bool = False,
        search_end_position: int | None = None,
    ) -> list[tuple[int, int, str]]:
        """テキスト内の対象単語の位置を検索.

        Args:
            text: 検索対象テキスト
            target_word: 対象単語
            case_sensitive: 大文字小文字を区別するか
            search_end_position: 検索終了位置（この位置より前のみ検索）

        Returns:
            (開始位置, 終了位置, マッチした文字列) のリスト
        """
        # 検索範囲を制限
        search_text = text if search_end_position is None else text[:search_end_position]

        flags = 0 if case_sensitive else re.IGNORECASE

        if self.language == "japanese":
            # 日本語の場合は単語境界なしで単純マッチング
            pattern = re.compile(re.escape(target_word), flags)
        else:
            # 英語の場合は単語境界を使用
            pattern = re.compile(r"\b" + re.escape(target_word) + r"\b", flags)

        positions = []
        for match in pattern.finditer(search_text):
            positions.append((match.start(), match.end(), match.group()))

        return positions

    def _generate_example_seed(self, example_id: int, word_index: int) -> int:
        """サンプルごとの一意なシードを生成.

        Args:
            example_id: サンプルID
            word_index: 対象単語のインデックス（頻出単語リスト内）

        Returns:
            一意なシード値
        """
        # 基本シード + サンプルID + 単語インデックスの組み合わせで一意なシードを生成
        return (self.base_seed * 10000 + example_id * 100 + word_index) % (2**32)

    def perturb_example(
        self,
        example_id: int,
        text: str,
        target_word: str,
        word_index: int = 0,
        case_sensitive: bool = False,
        extra_fields: dict[str, Any] | None = None,
        search_end_position: int | None = None,
    ) -> PerturbedExample | None:
        """単一サンプルに摂動を適用.

        Args:
            example_id: サンプルID
            text: 元のテキスト
            target_word: 対象単語
            word_index: 単語インデックス（シード生成用）
            case_sensitive: 大文字小文字を区別するか
            extra_fields: 追加フィールド（answer等）
            search_end_position: 検索終了位置（選択肢を除外するため）

        Returns:
            摂動済みサンプル（対象単語が含まれない場合はNone）
        """
        # 対象単語の位置を検索（search_end_positionより前のみ）
        positions = self._find_word_positions(
            text, target_word, case_sensitive, search_end_position
        )

        if not positions:
            return None

        # このサンプル用のシードを生成
        example_seed = self._generate_example_seed(example_id, word_index)

        # 摂動器を作成
        perturbator = Perturbator(
            replace_prob=self.replace_prob,
            insert_prob=self.insert_prob,
            delete_prob=self.delete_prob,
            seed=example_seed,
            language=self.language,
        )

        # 各出現に対して摂動を適用
        perturbations: list[OccurrencePerturbation] = []
        # 後ろから処理して位置がずれないようにする
        perturbed_text = text
        offset = 0

        for occ_idx, (start, end, matched_text) in enumerate(positions):
            # 各出現に対して独立した摂動を生成
            record = perturbator.perturb_word(matched_text.lower())

            # 摂動が発生しなかった場合はスキップ
            if not record.operations:
                continue

            # 大文字小文字パターンを保持
            if matched_text.isupper():
                perturbed_word = record.perturbed_word.upper()
            elif matched_text.istitle():
                perturbed_word = record.perturbed_word.capitalize()
            else:
                perturbed_word = record.perturbed_word

            # テキストを置換
            adjusted_start = start + offset
            adjusted_end = end + offset
            perturbed_text = (
                perturbed_text[:adjusted_start] + perturbed_word + perturbed_text[adjusted_end:]
            )

            # オフセットを更新
            offset += len(perturbed_word) - len(matched_text)

            # 摂動情報を記録
            perturbations.append(
                OccurrencePerturbation(
                    occurrence_index=occ_idx,
                    start_position=start,
                    end_position=end,
                    original_word=matched_text,
                    perturbed_word=perturbed_word,
                    operations=record.operations,
                )
            )

        # 摂動が1つも発生しなかった場合はNoneを返す
        if not perturbations:
            return None

        return PerturbedExample(
            example_id=example_id,
            seed=example_seed,
            original_text=text,
            perturbed_text=perturbed_text,
            perturbations=perturbations,
            extra_fields=extra_fields or {},
            total_occurrences_in_example=len(positions),
        )

    def perturb_benchmark(
        self,
        examples: list[dict[str, Any]],
        target_word: str,
        text_field: str,
        word_index: int = 0,
        case_sensitive: bool = False,
        search_end_field: str | None = None,
    ) -> PerturbedBenchmarkData:
        """ベンチマークデータ全体に摂動を適用.

        Args:
            examples: ベンチマークのサンプルリスト
            target_word: 対象単語
            text_field: テキストが含まれるフィールド名
            word_index: 単語インデックス
            case_sensitive: 大文字小文字を区別するか
            search_end_field: 検索終了位置を示すフィールド名（選択肢を除外するため）

        Returns:
            摂動済みベンチマークデータ
        """
        perturbed_examples: list[PerturbedExample] = []

        for idx, example in enumerate(examples):
            if text_field not in example:
                logger.warning(f"サンプル {idx}: フィールド '{text_field}' が存在しません")
                continue

            text = example[text_field]
            # text_field以外のフィールドをextra_fieldsとして保持
            extra_fields = {k: v for k, v in example.items() if k != text_field}

            # 検索終了位置を取得（選択肢を除外するため）
            search_end_position = None
            if search_end_field and search_end_field in example:
                search_end_position = example[search_end_field]

            perturbed = self.perturb_example(
                example_id=idx,
                text=text,
                target_word=target_word,
                word_index=word_index,
                case_sensitive=case_sensitive,
                extra_fields=extra_fields,
                search_end_position=search_end_position,
            )

            if perturbed is not None:
                perturbed_examples.append(perturbed)

        # 全体の統計を計算
        total_occurrences = sum(ex.total_occurrences_in_example for ex in perturbed_examples)
        perturbed_occurrences = sum(len(ex.perturbations) for ex in perturbed_examples)

        logger.info(
            f"摂動完了: {target_word} - "
            f"{len(perturbed_examples)}/{len(examples)}件に対象単語を含む, "
            f"総出現数: {total_occurrences}, 摂動数: {perturbed_occurrences}"
        )

        return PerturbedBenchmarkData(
            benchmark_name=self.benchmark_name,
            target_word=target_word,
            language=self.language,
            replace_prob=self.replace_prob,
            insert_prob=self.insert_prob,
            delete_prob=self.delete_prob,
            base_seed=self.base_seed,
            examples=perturbed_examples,
            total_occurrences=total_occurrences,
            perturbed_occurrences=perturbed_occurrences,
        )


def generate_perturbed_data(
    benchmark_data: OriginalBenchmarkData,
    frequent_words: list[str],
    word_scores: dict[str, float] | None,
    output_dir: Path,
    replace_prob: float = 0.2,
    insert_prob: float = 0.2,
    delete_prob: float = 0.2,
    base_seed: int = 42,
    case_sensitive: bool = False,
) -> dict[str, Path]:
    """頻出単語リストに基づいて摂動データを生成.

    Args:
        benchmark_data: 元のベンチマークデータ
        frequent_words: 頻出単語リスト
        word_scores: 単語→スコアの辞書（頻出度合いを示す）
        output_dir: 出力ディレクトリ
        replace_prob: 置換確率 (デフォルト: 0.2)
        insert_prob: 挿入確率 (デフォルト: 0.2)
        delete_prob: 削除確率 (デフォルト: 0.2)
        base_seed: 基本シード
        case_sensitive: 大文字小文字を区別するか

    Returns:
        {単語: 保存先パス} の辞書
    """
    perturbator = BenchmarkPerturbator(
        benchmark_name=benchmark_data.benchmark_name,
        language=benchmark_data.language,
        replace_prob=replace_prob,
        insert_prob=insert_prob,
        delete_prob=delete_prob,
        base_seed=base_seed,
    )

    # 元データを保存
    original_dir = output_dir / benchmark_data.benchmark_name / "original"
    original_path = original_dir / "examples.json"
    benchmark_data.save(original_path)

    # 各単語に対して摂動を生成
    saved_paths: dict[str, Path] = {}
    perturbed_dir = output_dir / benchmark_data.benchmark_name / "perturbed"

    for word_idx, word in enumerate(frequent_words):
        perturbed_data = perturbator.perturb_benchmark(
            examples=benchmark_data.examples,
            target_word=word,
            text_field=benchmark_data.text_field,
            word_index=word_idx,
            case_sensitive=case_sensitive,
            search_end_field=benchmark_data.search_end_field,
        )

        # スコアを設定（word_scoresが提供されている場合）
        if word_scores is not None and word in word_scores:
            perturbed_data.target_word_score = word_scores[word]

        # 対象単語を含むサンプルがある場合のみ保存
        if perturbed_data.examples:
            word_dir = perturbed_dir / word
            word_path = word_dir / "examples.json"
            perturbed_data.save(word_path)
            saved_paths[word] = word_path
        else:
            logger.debug(f"単語 '{word}' を含むサンプルがないためスキップ")

    logger.info(
        f"摂動データ生成完了: {len(saved_paths)}/{len(frequent_words)}単語に対してデータを生成"
    )

    return saved_paths
