"""品詞判定ユーティリティモジュール.

WordNet、NLTK、トークナイザーを組み合わせた品詞判定・単語存在確認を提供する.

単語存在確認（厳格モード）:
1. トークナイザーで単一トークンとして存在する
2. かつ、NLTKまたはWordNetで品詞情報が取得できる
※ NLTKのwords corpusは古語や珍しい単語を含むため使用しない

品詞判定:
NLTKのpos_tagを使用して単語の品詞を取得する.
Penn Treebankタグを使用し、同品詞/異品詞の判定には大分類を使用する.

これにより、for/the/whoなどの機能語も正しく処理される.
"""

import string
from dataclasses import dataclass, field
from functools import lru_cache

from nltk import pos_tag
from nltk.corpus import wordnet as wn

# Penn Treebank品詞タグの大分類へのマッピング
# 同品詞/異品詞の判定に使用
PTB_TO_CATEGORY = {
    # 名詞
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "NOUN",
    "NNPS": "NOUN",
    # 動詞
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    # 形容詞
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    # 副詞
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    # 機能語
    "DT": "DET",
    "PDT": "DET",
    "WDT": "DET",
    "IN": "ADP",
    "TO": "PART",
    "RP": "PART",
    "PRP": "PRON",
    "PRP$": "PRON",
    "WP": "PRON",
    "WP$": "PRON",
    "CC": "CCONJ",
    "MD": "AUX",
    "UH": "INTJ",
    "CD": "NUM",
    "EX": "PRON",
    "WRB": "ADV",
    # 記号等
    ".": "PUNCT",
    ",": "PUNCT",
    ":": "PUNCT",
    "``": "PUNCT",
    "''": "PUNCT",
    "-LRB-": "PUNCT",
    "-RRB-": "PUNCT",
    "$": "SYM",
    "#": "SYM",
    "POS": "PART",
    "FW": "X",
    "LS": "X",
    "SYM": "SYM",
}

# 機能語の品詞カテゴリ（WordNetにない品詞）
FUNCTION_WORD_CATEGORIES = {"DET", "ADP", "PART", "PRON", "CCONJ", "AUX", "INTJ", "NUM", "PUNCT"}

# 機能語のPenn Treebankタグ
FUNCTION_WORD_PTB_TAGS = {
    "DT", "PDT", "WDT",  # 限定詞
    "IN", "TO",  # 前置詞・不定詞マーカー
    "RP",  # 助詞
    "PRP", "PRP$", "WP", "WP$", "EX",  # 代名詞
    "CC",  # 等位接続詞
    "MD",  # 助動詞
    "UH",  # 間投詞
    "CD",  # 数詞
}


@dataclass
class WordPOSInfo:
    """単語の品詞情報.

    Attributes:
        word: 単語
        exists_in_wordnet: WordNetに存在するか（内容語のみ）
        exists_in_dictionary: 英語辞書で認識されるか（機能語を含む）
        wordnet_pos_set: WordNetの品詞セット（n, v, a, r）
        nltk_pos: NLTKによる品詞タグ (Penn Treebank)
        pos_category: 品詞カテゴリ（比較に使用）
    """

    word: str
    exists_in_wordnet: bool
    exists_in_dictionary: bool = True  # デフォルトTrue（後方互換性）
    wordnet_pos_set: set[str] = field(default_factory=set)
    nltk_pos: str | None = None
    pos_category: str | None = None


@lru_cache(maxsize=10000)
def get_nltk_pos(word: str) -> str | None:
    """NLTKを使用して単語の品詞を取得.

    Args:
        word: 対象の単語

    Returns:
        Penn Treebank品詞タグ
    """
    tagged = pos_tag([word.lower()])
    if not tagged:
        return None
    return tagged[0][1]


@lru_cache(maxsize=10000)
def get_pos_category(word: str) -> str | None:
    """単語の品詞カテゴリを取得.

    Penn Treebankタグを大分類（NOUN, VERB, ADJ, ADV等）に変換する.

    Args:
        word: 対象の単語

    Returns:
        品詞カテゴリ（NOUN, VERB, ADJ, ADV, DET, ADP等）
    """
    ptb_tag = get_nltk_pos(word)
    if ptb_tag is None:
        return None
    return PTB_TO_CATEGORY.get(ptb_tag)


# 後方互換性のためのエイリアス
def get_spacy_pos(word: str) -> str | None:
    """NLTKを使用して単語の品詞カテゴリを取得（後方互換性用）.

    Args:
        word: 対象の単語

    Returns:
        品詞カテゴリ
    """
    return get_pos_category(word)


@lru_cache(maxsize=10000)
def get_wordnet_pos_set(word: str) -> set[str]:
    """WordNetから単語の品詞セットを取得.

    Args:
        word: 対象の単語

    Returns:
        品詞の集合（n=名詞, v=動詞, a=形容詞, r=副詞）
    """
    synsets = wn.synsets(word.lower())
    if not synsets:
        return set()

    pos_set = set()
    for synset in synsets:
        pos = synset.pos()
        pos_set.add(pos)
        # 's'（衛星形容詞）は'a'（形容詞）と同等
        if pos == "s":
            pos_set.add("a")
        elif pos == "a":
            pos_set.add("s")

    return pos_set


@lru_cache(maxsize=10000)
def word_exists_in_wordnet(word: str) -> bool:
    """単語がWordNetに存在するかを確認.

    注意: WordNetは機能語（for, the, who等）を含まない.
    機能語も含めた確認には word_exists_in_dictionary() を使用.

    Args:
        word: 対象の単語

    Returns:
        存在する場合True
    """
    return len(wn.synsets(word.lower())) > 0


def is_single_token(word: str, tokenizer) -> bool:
    """単語がトークナイザーで単一トークンとして存在するかを確認.

    Args:
        word: 対象の単語
        tokenizer: Transformersのトークナイザー

    Returns:
        単一トークンの場合True
    """
    if tokenizer is None:
        return True  # トークナイザーがない場合はチェックをスキップ

    tokens = tokenizer.encode(word.lower(), add_special_tokens=False)
    return len(tokens) == 1


def has_pos_info(word: str) -> bool:
    """単語が実在語として品詞情報を持つかを確認.

    判定基準（厳格モード）:
    1. WordNetに存在する場合: OK（内容語として確実に存在）
    2. NLTKで機能語として認識される場合: OK（機能語はWordNetにない）
    3. それ以外: NG（NLTKは未知語にも品詞を割り当てるため信頼できない）

    Args:
        word: 対象の単語

    Returns:
        実在語として品詞情報がある場合True
    """
    word_lower = word.lower()

    # 条件1: WordNetに存在するか（内容語）
    if word_exists_in_wordnet(word_lower):
        return True

    # 条件2: NLTKで機能語として認識されるか
    # 機能語はWordNetに含まれないため、NLTKの判定を使用
    nltk_pos = get_nltk_pos(word_lower)
    if nltk_pos and nltk_pos in FUNCTION_WORD_PTB_TAGS:
        return True

    return False


def word_exists_strict(word: str, tokenizer) -> bool:
    """単語が厳格な基準で実在するかを確認.

    以下の両方を満たす場合にTrueを返す:
    1. トークナイザーで単一トークンとして存在する
    2. WordNetに存在する（内容語）、またはNLTKで機能語として認識される

    これにより、トークナイザーには登録されているが実在しない
    サブワード断片（例: ollowing）を除外できる.

    Args:
        word: 対象の単語
        tokenizer: Transformersのトークナイザー

    Returns:
        実在語と判定される場合True
    """
    word_lower = word.lower()

    # 条件1: トークナイザーで単一トークン
    if not is_single_token(word_lower, tokenizer):
        return False

    # 条件2: 品詞情報がある
    if not has_pos_info(word_lower):
        return False

    return True


@lru_cache(maxsize=10000)
def word_exists_in_dictionary(word: str) -> bool:
    """単語がNLTKまたはWordNetで認識されるかを確認（後方互換性用）.

    注意: この関数はトークナイザーを使用しない簡易版です.
    より厳格な判定には word_exists_strict() を使用してください.

    Args:
        word: 対象の単語

    Returns:
        存在する場合True
    """
    return has_pos_info(word.lower())


def get_word_pos_info(word: str) -> WordPOSInfo:
    """単語の品詞情報を取得.

    Args:
        word: 対象の単語

    Returns:
        品詞情報
    """
    word_lower = word.lower()
    exists_wn = word_exists_in_wordnet(word_lower)
    exists_dict = word_exists_in_dictionary(word_lower)
    wn_pos_set = get_wordnet_pos_set(word_lower)
    nltk_pos = get_nltk_pos(word_lower)
    pos_category = get_pos_category(word_lower)

    return WordPOSInfo(
        word=word,
        exists_in_wordnet=exists_wn,
        exists_in_dictionary=exists_dict,
        wordnet_pos_set=wn_pos_set,
        nltk_pos=nltk_pos,
        pos_category=pos_category,
    )


def has_common_pos(word1: str, word2: str) -> bool:
    """2つの単語の主要品詞が一致するかを判定.

    NLTKの品詞カテゴリを使用して判定する。WordNetは複数の品詞を返すため、
    一般的な用法を反映するNLTKの品詞を使用する。

    Args:
        word1: 単語1
        word2: 単語2

    Returns:
        主要品詞が一致する場合True
    """
    pos1 = get_pos_category(word1.lower())
    pos2 = get_pos_category(word2.lower())

    if not pos1 or not pos2:
        return False

    return pos1 == pos2


def get_primary_pos(word: str) -> str | None:
    """単語の主要な品詞を取得.

    NLTKの品詞カテゴリを返す。NLTKは一般的な用法を反映した品詞を返すため、
    同品詞/異品詞の判定に適している。

    Args:
        word: 対象の単語

    Returns:
        主要な品詞カテゴリ（NOUN, VERB, ADJ, ADV等）
    """
    return get_pos_category(word.lower())


# 編集操作のタイプ
EditType = str  # "replace", "insert", "delete"


@dataclass
class EditOperation:
    """編集操作.

    Attributes:
        edit_type: 編集タイプ（replace/insert/delete）
        position: 編集位置
        original_char: 元の文字（replaceとdeleteの場合）
        new_char: 新しい文字（replaceとinsertの場合）
        description: 操作の説明
    """

    edit_type: EditType
    position: int
    original_char: str | None = None
    new_char: str | None = None
    description: str = ""


def generate_single_char_edits(word: str) -> list[tuple[str, EditOperation]]:
    """1文字編集（置換・挿入・削除）の全候補を生成.

    Args:
        word: 元の単語

    Returns:
        (編集後の単語, 編集操作) のリスト
    """
    candidates = []
    word_lower = word.lower()
    lowercase_letters = string.ascii_lowercase

    # 1. 置換（Replace）
    for i, char in enumerate(word_lower):
        for replacement in lowercase_letters:
            if replacement != char:
                new_word = word_lower[:i] + replacement + word_lower[i + 1 :]
                op = EditOperation(
                    edit_type="replace",
                    position=i,
                    original_char=char,
                    new_char=replacement,
                    description=f"replace '{char}' with '{replacement}' at pos {i}",
                )
                candidates.append((new_word, op))

    # 2. 挿入（Insert）
    for i in range(len(word_lower) + 1):
        for char in lowercase_letters:
            new_word = word_lower[:i] + char + word_lower[i:]
            if i == 0:
                desc = f"insert '{char}' at beginning"
            elif i == len(word_lower):
                desc = f"insert '{char}' at end"
            else:
                desc = f"insert '{char}' at pos {i}"
            op = EditOperation(
                edit_type="insert",
                position=i,
                new_char=char,
                description=desc,
            )
            candidates.append((new_word, op))

    # 3. 削除（Delete）
    if len(word_lower) > 1:  # 1文字の単語は削除不可
        for i, char in enumerate(word_lower):
            new_word = word_lower[:i] + word_lower[i + 1 :]
            op = EditOperation(
                edit_type="delete",
                position=i,
                original_char=char,
                description=f"delete '{char}' at pos {i}",
            )
            candidates.append((new_word, op))

    return candidates


def check_subword_preservation(
    original: str,
    perturbed: str,
    tokenizer=None,
) -> bool:
    """サブワードトークン分割が変わらないかを確認.

    Args:
        original: 元の単語
        perturbed: 摂動後の単語
        tokenizer: Transformersのトークナイザー（Noneの場合はチェックしない）

    Returns:
        トークン分割が変わらない場合True
    """
    if tokenizer is None:
        return True

    # トークナイズ
    original_tokens = tokenizer.encode(original, add_special_tokens=False)
    perturbed_tokens = tokenizer.encode(perturbed, add_special_tokens=False)

    # トークン数が同じかどうかを確認
    return len(original_tokens) == len(perturbed_tokens)
