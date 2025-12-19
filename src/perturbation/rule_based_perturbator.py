"""規則ベースの摂動生成モジュール.

ランダムではなく、規則に則って摂動（typo）を生成する.
特に、1文字置換のtypoで変化しうる単語のうち、
パターン.1（置換後の単語が存在し、かつ置換前の単語と品詞が一致する）に該当するものを列挙する.
"""

import json
import string
from dataclasses import dataclass, field
from pathlib import Path

from nltk.corpus import wordnet as wn

from src.utils.logger import logger


@dataclass
class RuleBasedPerturbationResult:
    """規則ベース摂動の結果.

    Attributes:
        rank: 元の単語のランク
        original_word: 元の単語
        perturbed_words: パターン.1に該当する摂動後の単語リスト（なければNone）
    """

    rank: int
    original_word: str
    perturbed_words: list[str] | None = field(default=None)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "rank": self.rank,
            "original_word": self.original_word,
            "perturbed_words": self.perturbed_words,
        }


def get_wordnet_pos(word: str) -> set[str]:
    """WordNetから単語の品詞を取得.

    Args:
        word: 対象の単語

    Returns:
        品詞の集合（n=名詞, v=動詞, a=形容詞, r=副詞, s=衛星形容詞）
    """
    synsets = wn.synsets(word)
    if not synsets:
        return set()

    # 全てのsynsetから品詞を収集
    pos_set = set()
    for synset in synsets:
        pos = synset.pos()
        pos_set.add(pos)
        # 's'（衛星形容詞）は'a'（形容詞）と同等として扱う
        if pos == "s":
            pos_set.add("a")
        elif pos == "a":
            pos_set.add("s")

    return pos_set


def word_exists_in_wordnet(word: str) -> bool:
    """単語がWordNetに存在するかを確認.

    Args:
        word: 対象の単語

    Returns:
        存在する場合True
    """
    return len(wn.synsets(word)) > 0


def generate_single_char_replacements(word: str) -> list[str]:
    """1文字置換の全候補を生成.

    Args:
        word: 元の単語

    Returns:
        1文字置換した全ての候補リスト（元の単語と同じものは除外）
    """
    candidates = []
    word_lower = word.lower()

    for i, char in enumerate(word_lower):
        for replacement in string.ascii_lowercase:
            if replacement != char:
                # i番目の文字をreplacementに置換
                new_word = word_lower[:i] + replacement + word_lower[i + 1 :]
                candidates.append(new_word)

    return candidates


def find_pattern1_perturbations(word: str) -> list[str]:
    """パターン.1に該当する摂動候補を検索.

    パターン.1: 置換後の単語が存在し、かつ置換前の単語と品詞が一致する場合.

    Args:
        word: 元の単語

    Returns:
        パターン.1に該当する摂動後の単語リスト
    """
    word_lower = word.lower()

    # 元の単語の品詞を取得
    original_pos = get_wordnet_pos(word_lower)
    if not original_pos:
        logger.debug(f"単語 '{word}' の品詞がWordNetで見つかりません")
        return []

    # 1文字置換の全候補を生成
    candidates = generate_single_char_replacements(word_lower)

    # パターン.1に該当する候補をフィルタリング
    pattern1_words = []
    for candidate in candidates:
        # 候補がWordNetに存在するか確認
        if not word_exists_in_wordnet(candidate):
            continue

        # 候補の品詞を取得
        candidate_pos = get_wordnet_pos(candidate)

        # 元の単語と品詞が一致するか確認（共通の品詞があるか）
        if original_pos & candidate_pos:
            pattern1_words.append(candidate)

    return pattern1_words


def process_frequent_words(
    input_path: Path,
    output_path: Path | None = None,
    top_n: int | None = None,
) -> list[RuleBasedPerturbationResult]:
    """頻出単語リストを処理し、パターン.1の摂動候補を生成.

    Args:
        input_path: 入力JSONファイルのパス（frequent_words_top{n}.json）
        output_path: 出力JSONファイルのパス（Noneの場合は保存しない）
        top_n: 処理する上位N件（Noneの場合は全件処理）

    Returns:
        処理結果のリスト
    """
    # 入力ファイルを読み込み
    logger.info(f"入力ファイルを読み込み: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    words_data = data["words"]
    if top_n:
        words_data = words_data[:top_n]

    logger.info(f"処理対象単語数: {len(words_data)}")

    # 各単語を処理
    results: list[RuleBasedPerturbationResult] = []
    for i, word_entry in enumerate(words_data):
        rank = word_entry["rank"]
        word = word_entry["word"]

        if (i + 1) % 100 == 0:
            logger.info(f"処理中: {i + 1}/{len(words_data)}")

        # パターン.1に該当する摂動候補を検索
        perturbed_words = find_pattern1_perturbations(word)

        # 結果を作成（候補がない場合はNone）
        result = RuleBasedPerturbationResult(
            rank=rank,
            original_word=word,
            perturbed_words=perturbed_words if perturbed_words else None,
        )
        results.append(result)

    logger.info(f"処理完了: {len(results)}件")

    # 統計情報を表示
    words_with_perturbations = sum(1 for r in results if r.perturbed_words)
    total_perturbations = sum(
        len(r.perturbed_words) for r in results if r.perturbed_words
    )
    logger.info(f"パターン.1該当単語数: {words_with_perturbations}/{len(results)}")
    logger.info(f"パターン.1摂動候補総数: {total_perturbations}")

    # 出力ファイルに保存
    if output_path:
        output_data = {
            "metadata": {
                "source": data.get("metadata", {}).get("source", "unknown"),
                "language": data.get("metadata", {}).get("language", "english"),
                "perturbation_type": "single_char_replacement",
                "pattern": "pattern1_same_pos",
                "pattern_description": "置換後の単語が存在し、かつ置換前の単語と品詞が一致",
                "total_words": len(results),
                "words_with_perturbations": words_with_perturbations,
                "total_perturbations": total_perturbations,
            },
            "results": [r.to_dict() for r in results],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"出力ファイルを保存: {output_path}")

    return results


if __name__ == "__main__":
    # テスト実行
    import argparse

    parser = argparse.ArgumentParser(
        description="規則ベースの摂動生成（パターン.1: 品詞一致）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/english/frequent_words_top500.json"),
        help="入力JSONファイルのパス",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/english/pattern1_perturbations.json"),
        help="出力JSONファイルのパス",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="処理する上位N件（指定しない場合は全件）",
    )

    args = parser.parse_args()

    process_frequent_words(
        input_path=args.input,
        output_path=args.output,
        top_n=args.top_n,
    )
