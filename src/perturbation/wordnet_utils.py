"""WordNet関連のユーティリティモジュール.

単語の存在確認、品詞判定などの共通機能を提供する.
実験1（ルールベース摂動）と実験2（パターン摂動）で共有される.
"""

import string
from dataclasses import dataclass
from enum import Enum

from nltk.corpus import wordnet as wn

from src.utils.logger import logger


class POSType(Enum):
    """品詞タイプ."""

    NOUN = "n"  # 名詞
    VERB = "v"  # 動詞
    ADJECTIVE = "a"  # 形容詞
    ADVERB = "r"  # 副詞
    ADJECTIVE_SATELLITE = "s"  # 衛星形容詞


@dataclass
class WordInfo:
    """単語情報.

    Attributes:
        word: 単語
        exists: WordNetに存在するか
        pos_set: 品詞の集合
    """

    word: str
    exists: bool
    pos_set: set[str]


def get_wordnet_pos(word: str) -> set[str]:
    """WordNetから単語の品詞を取得.

    Args:
        word: 対象の単語

    Returns:
        品詞の集合（n=名詞, v=動詞, a=形容詞, r=副詞, s=衛星形容詞）
    """
    synsets = wn.synsets(word.lower())
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
    return len(wn.synsets(word.lower())) > 0


def get_word_info(word: str) -> WordInfo:
    """単語の情報を取得.

    Args:
        word: 対象の単語

    Returns:
        単語情報（存在確認、品詞）
    """
    word_lower = word.lower()
    synsets = wn.synsets(word_lower)

    if not synsets:
        return WordInfo(word=word, exists=False, pos_set=set())

    pos_set = set()
    for synset in synsets:
        pos = synset.pos()
        pos_set.add(pos)
        if pos == "s":
            pos_set.add("a")
        elif pos == "a":
            pos_set.add("s")

    return WordInfo(word=word, exists=True, pos_set=pos_set)


def has_same_pos(word1: str, word2: str) -> bool:
    """2つの単語が同じ品詞を持つかを確認.

    Args:
        word1: 単語1
        word2: 単語2

    Returns:
        共通の品詞がある場合True
    """
    pos1 = get_wordnet_pos(word1)
    pos2 = get_wordnet_pos(word2)

    if not pos1 or not pos2:
        return False

    return bool(pos1 & pos2)


def has_different_pos_only(word1: str, word2: str) -> bool:
    """2つの単語が異なる品詞のみを持つかを確認.

    両方の単語がWordNetに存在し、共通の品詞がない場合にTrue.

    Args:
        word1: 単語1
        word2: 単語2

    Returns:
        異なる品詞のみの場合True
    """
    pos1 = get_wordnet_pos(word1)
    pos2 = get_wordnet_pos(word2)

    if not pos1 or not pos2:
        return False

    return not bool(pos1 & pos2)


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


@dataclass
class PerturbationCandidate:
    """摂動候補.

    Attributes:
        original: 元の単語
        perturbed: 摂動後の単語
        pattern: 摂動パターン（pattern1/pattern2/pattern3）
        original_pos: 元の単語の品詞
        perturbed_pos: 摂動後の単語の品詞（pattern3の場合はNone）
    """

    original: str
    perturbed: str
    pattern: str
    original_pos: set[str]
    perturbed_pos: set[str] | None


def classify_perturbation(original: str, perturbed: str) -> str | None:
    """摂動をパターンに分類.

    Args:
        original: 元の単語
        perturbed: 摂動後の単語

    Returns:
        パターン名（pattern1/pattern2/pattern3）、分類不能の場合None
    """
    original_info = get_word_info(original)
    perturbed_info = get_word_info(perturbed)

    # 元の単語がWordNetに存在しない場合は分類不能
    if not original_info.exists:
        logger.debug(f"元の単語 '{original}' がWordNetに存在しません")
        return None

    # Pattern 3: 摂動後の単語が存在しない
    if not perturbed_info.exists:
        return "pattern3"

    # Pattern 1: 摂動後の単語が存在し、同じ品詞を持つ
    if original_info.pos_set & perturbed_info.pos_set:
        return "pattern1"

    # Pattern 2: 摂動後の単語が存在するが、異なる品詞のみ
    return "pattern2"


def find_perturbation_candidates(
    word: str,
) -> dict[str, list[PerturbationCandidate]]:
    """単語に対する全ての摂動候補をパターン別に検索.

    Args:
        word: 元の単語

    Returns:
        パターン別の摂動候補リスト
    """
    word_lower = word.lower()
    original_info = get_word_info(word_lower)

    if not original_info.exists:
        logger.debug(f"単語 '{word}' の品詞がWordNetで見つかりません")
        return {"pattern1": [], "pattern2": [], "pattern3": []}

    # 1文字置換の全候補を生成
    candidates = generate_single_char_replacements(word_lower)

    results: dict[str, list[PerturbationCandidate]] = {
        "pattern1": [],
        "pattern2": [],
        "pattern3": [],
    }

    for candidate in candidates:
        pattern = classify_perturbation(word_lower, candidate)
        if pattern is None:
            continue

        perturbed_info = get_word_info(candidate)

        perturbation = PerturbationCandidate(
            original=word_lower,
            perturbed=candidate,
            pattern=pattern,
            original_pos=original_info.pos_set,
            perturbed_pos=perturbed_info.pos_set if perturbed_info.exists else None,
        )

        results[pattern].append(perturbation)

    return results


def download_wordnet_if_needed() -> None:
    """WordNetデータが存在しない場合にダウンロード."""
    try:
        wn.synsets("test")
    except LookupError:
        import nltk

        logger.info("WordNetデータをダウンロード中...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        logger.info("WordNetデータのダウンロード完了")
