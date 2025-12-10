"""文字レベル摂動生成モジュール.

Chai et al.の手法に基づき、単語に対して文字レベルの摂動（typo）を適用する.
各文字に対して、置換・挿入・削除がそれぞれ独立して10%の確率で発生する.
"""

import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import logger


@dataclass
class PerturbationOperation:
    """単一の摂動操作を表すデータクラス."""

    position: int
    operation: str  # "replace", "insert", "delete"
    original_char: str | None
    new_char: str | None

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        return {
            "position": self.position,
            "operation": self.operation,
            "original_char": self.original_char,
            "new_char": self.new_char,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerturbationOperation":
        """辞書から復元."""
        return cls(
            position=data["position"],
            operation=data["operation"],
            original_char=data["original_char"],
            new_char=data["new_char"],
        )


@dataclass
class PerturbationRecord:
    """単語に対する摂動記録を表すデータクラス."""

    original_word: str
    perturbed_word: str
    seed: int
    operations: list[PerturbationOperation] = field(default_factory=list)
    occurrences: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換."""
        return {
            "original_word": self.original_word,
            "perturbed_word": self.perturbed_word,
            "seed": self.seed,
            "operations": [op.to_dict() for op in self.operations],
            "occurrences": self.occurrences,
            "num_perturbations": len(self.operations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerturbationRecord":
        """辞書から復元."""
        return cls(
            original_word=data["original_word"],
            perturbed_word=data["perturbed_word"],
            seed=data["seed"],
            operations=[PerturbationOperation.from_dict(op) for op in data["operations"]],
            occurrences=data.get("occurrences", []),
        )

    def save(self, file_path: Path) -> None:
        """JSONファイルに保存."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "PerturbationRecord":
        """JSONファイルから読み込み."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class Perturbator:
    """文字レベル摂動を生成するクラス.

    各文字に対して、置換・挿入・削除がそれぞれ独立した確率で発生する.
    デフォルトでは各操作10%の確率（合計約27%の確率で何らかの摂動が発生）.
    挿入・置換時は元の文字と同じ文字種（ひらがな、カタカナ、英字）の文字を使用する.
    """

    # 英語用の文字セット
    ENGLISH_CHARS = string.ascii_lowercase
    # 日本語用のひらがな文字セット
    HIRAGANA_CHARS = "".join(chr(c) for c in range(0x3041, 0x3097))
    # 日本語用のカタカナ文字セット
    KATAKANA_CHARS = "".join(chr(c) for c in range(0x30A1, 0x30F7))

    @staticmethod
    def _get_char_type(char: str) -> str:
        """文字の種類を判定する.

        Args:
            char: 判定する文字

        Returns:
            文字種 ("hiragana", "katakana", "english", "other")
        """
        code = ord(char)
        if 0x3041 <= code <= 0x3096:
            return "hiragana"
        elif 0x30A1 <= code <= 0x30F6:
            return "katakana"
        elif char.isalpha() and char.isascii():
            return "english"
        else:
            return "other"

    def _get_char_set_for_type(self, char_type: str) -> str:
        """文字種に応じた文字セットを取得する.

        Args:
            char_type: 文字種

        Returns:
            文字セット
        """
        if char_type == "hiragana":
            return self.HIRAGANA_CHARS
        elif char_type == "katakana":
            return self.KATAKANA_CHARS
        elif char_type == "english":
            return self.ENGLISH_CHARS
        else:
            # その他の場合はデフォルトの文字セットを使用
            return self.char_set

    def __init__(
        self,
        replace_prob: float = 0.2,
        insert_prob: float = 0.2,
        delete_prob: float = 0.2,
        seed: int | None = None,
        language: str = "english",
        *,
        perturbation_prob: float | None = None,  # 後方互換性のため
    ) -> None:
        """初期化.

        Args:
            replace_prob: 各文字に対する置換確率 (デフォルト: 0.2)
            insert_prob: 各文字に対する挿入確率 (デフォルト: 0.2)
            delete_prob: 各文字に対する削除確率 (デフォルト: 0.2)
            seed: 乱数シード (再現性のため)
            language: 言語 ("english" or "japanese")
            perturbation_prob: 後方互換性のため。指定時は全操作に同じ確率を適用
        """
        # 後方互換性: perturbation_probが指定された場合は全操作に適用
        if perturbation_prob is not None:
            replace_prob = perturbation_prob
            insert_prob = perturbation_prob
            delete_prob = perturbation_prob

        self.replace_prob = replace_prob
        self.insert_prob = insert_prob
        self.delete_prob = delete_prob
        self.perturbation_prob = replace_prob  # 後方互換性のため保持
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.language = language
        self._rng = random.Random(self.seed)

        # 言語に応じた文字セットを設定
        if language == "english":
            self.char_set = self.ENGLISH_CHARS
        elif language == "japanese":
            self.char_set = self.HIRAGANA_CHARS + self.KATAKANA_CHARS
        else:
            raise ValueError(f"サポートされていない言語: {language}")

        logger.debug(
            f"Perturbator初期化: seed={self.seed}, "
            f"replace={replace_prob}, insert={insert_prob}, delete={delete_prob}"
        )

    def reset_seed(self, seed: int | None = None) -> None:
        """乱数シードをリセット.

        Args:
            seed: 新しいシード (Noneの場合は現在のシードを再設定)
        """
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)

    def _get_random_char(
        self, exclude: str | None = None, reference_char: str | None = None
    ) -> str:
        """ランダムな文字を取得.

        Args:
            exclude: 除外する文字
            reference_char: 参照文字（この文字と同じ種類の文字を選択する）

        Returns:
            ランダムな文字
        """
        # 参照文字が指定された場合、その文字種に応じた文字セットを使用
        if reference_char is not None:
            char_type = self._get_char_type(reference_char)
            available_chars = self._get_char_set_for_type(char_type)
        else:
            available_chars = self.char_set

        if exclude and exclude in available_chars:
            available_chars = available_chars.replace(exclude, "")

        return self._rng.choice(available_chars)

    def perturb_word(self, word: str) -> PerturbationRecord:
        """単語に摂動を適用.

        単語全体に対して最大1回の摂動操作（置換・挿入・削除のいずれか）が発生する.
        摂動を行う場合、ランダムに1つの文字位置を選択し、その位置に対して操作を適用する.

        Args:
            word: 元の単語

        Returns:
            摂動記録
        """
        operations: list[PerturbationOperation] = []

        # 元のシード状態を記録
        current_seed = self.seed

        # 合計確率（単語全体で摂動を行うかどうかの確率）
        total_prob = self.replace_prob + self.insert_prob + self.delete_prob

        # 摂動を行うかどうかを判定（単語全体で1回だけ判定）
        if total_prob > 0 and self._rng.random() < total_prob:
            # 摂動を行う場合、ランダムに1つの文字位置を選択
            target_position = self._rng.randint(0, len(word) - 1)

            # どの操作を行うかを重み付きで選択
            rand_val = self._rng.random() * total_prob

            if rand_val < self.delete_prob:
                # 削除: 対象位置の文字を削除
                operations.append(
                    PerturbationOperation(
                        position=target_position,
                        operation="delete",
                        original_char=word[target_position],
                        new_char=None,
                    )
                )
                perturbed_word = word[:target_position] + word[target_position + 1 :]

            elif rand_val < self.delete_prob + self.replace_prob:
                # 置換: 対象位置の文字を同じ文字種の別の文字に置き換え
                original_char = word[target_position]
                new_char = self._get_random_char(
                    exclude=original_char, reference_char=original_char
                )
                operations.append(
                    PerturbationOperation(
                        position=target_position,
                        operation="replace",
                        original_char=original_char,
                        new_char=new_char,
                    )
                )
                perturbed_word = word[:target_position] + new_char + word[target_position + 1 :]

            else:
                # 挿入: 対象位置の前に同じ文字種のランダムな文字を挿入
                reference_char = word[target_position]
                new_char = self._get_random_char(reference_char=reference_char)
                operations.append(
                    PerturbationOperation(
                        position=target_position,
                        operation="insert",
                        original_char=None,
                        new_char=new_char,
                    )
                )
                perturbed_word = word[:target_position] + new_char + word[target_position:]
        else:
            # 摂動なし: そのまま
            perturbed_word = word

        return PerturbationRecord(
            original_word=word,
            perturbed_word=perturbed_word,
            seed=current_seed,
            operations=operations,
        )

    def reconstruct_perturbation(self, record: PerturbationRecord) -> str:
        """摂動記録から摂動後の単語を再構築.

        Args:
            record: 摂動記録

        Returns:
            摂動後の単語
        """
        # シードを設定して同じ乱数列を再現
        temp_rng = random.Random(record.seed)
        result_chars: list[str] = []

        for _i, char in enumerate(record.original_word):
            if temp_rng.random() < self.perturbation_prob:
                operation_type = temp_rng.choice(["replace", "insert", "delete"])

                if operation_type == "replace":
                    new_char = self._get_random_char_with_rng(temp_rng, exclude=char)
                    result_chars.append(new_char)
                elif operation_type == "insert":
                    new_char = self._get_random_char_with_rng(temp_rng)
                    result_chars.append(new_char)
                    result_chars.append(char)
                elif operation_type == "delete":
                    pass  # 削除
            else:
                result_chars.append(char)

        return "".join(result_chars)

    def _get_random_char_with_rng(self, rng: random.Random, exclude: str | None = None) -> str:
        """指定したRNGを使用してランダムな文字を取得."""
        available_chars = self.char_set
        if exclude and exclude in available_chars:
            available_chars = available_chars.replace(exclude, "")
        return rng.choice(available_chars)


def apply_perturbation_to_text(
    text: str,
    target_word: str,
    perturbator: Perturbator,
    case_sensitive: bool = False,
) -> tuple[str, PerturbationRecord, int]:
    """テキスト内の対象単語すべてに摂動を適用.

    Args:
        text: 元のテキスト
        target_word: 摂動を適用する対象単語
        perturbator: 摂動生成器
        case_sensitive: 大文字小文字を区別するか

    Returns:
        (摂動後のテキスト, 摂動記録, 置換回数)
    """
    # 摂動を生成（1回だけ）
    perturbator.reset_seed()  # シードをリセットして一貫性を保つ
    record = perturbator.perturb_word(target_word.lower() if not case_sensitive else target_word)

    # 置換パターンを構築
    if case_sensitive:
        pattern = re.compile(re.escape(target_word))
    else:
        pattern = re.compile(re.escape(target_word), re.IGNORECASE)

    # 出現位置を記録
    occurrences = []
    for match in pattern.finditer(text):
        occurrences.append(
            {
                "start": match.start(),
                "end": match.end(),
                "matched_text": match.group(),
            }
        )

    record.occurrences = occurrences

    # 置換を実行
    def replace_match(match: re.Match) -> str:
        """マッチした文字列を置換."""
        original = match.group()
        # 元の大文字小文字パターンを保持
        if original.isupper():
            return record.perturbed_word.upper()
        elif original.istitle():
            return record.perturbed_word.capitalize()
        else:
            return record.perturbed_word

    perturbed_text, count = pattern.subn(replace_match, text)

    return perturbed_text, record, count


def find_word_occurrences(
    texts: list[str],
    target_word: str,
    case_sensitive: bool = False,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """複数テキスト内で対象単語の出現位置を検索.

    Args:
        texts: テキストのリスト
        target_word: 検索する単語
        case_sensitive: 大文字小文字を区別するか

    Returns:
        (テキストインデックス, [(開始位置, 終了位置), ...]) のリスト
    """
    if case_sensitive:
        pattern = re.compile(r"\b" + re.escape(target_word) + r"\b")
    else:
        pattern = re.compile(r"\b" + re.escape(target_word) + r"\b", re.IGNORECASE)

    results = []
    for idx, text in enumerate(texts):
        matches = [(m.start(), m.end()) for m in pattern.finditer(text)]
        if matches:
            results.append((idx, matches))

    return results
