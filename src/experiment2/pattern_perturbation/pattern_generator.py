"""パターン摂動生成モジュール.

決定論的なパターン摂動（Pattern 1/2/3）を生成する.
ベンチマーク間での比較を可能にするため、
同じトークンに対して同じ摂動を適用する.

摂動パターン:
- Pattern 1: 摂動後のトークンが実在 + 同品詞
- Pattern 2: 摂動後のトークンが実在 + 異品詞
- Pattern 3: 摂動後のトークンが非実在（UNK）

単語存在確認（厳格モード）:
1. トークナイザーで単一トークンとして存在する
2. かつ、WordNetに存在する（内容語）、またはspaCyで機能語として認識される
※ NLTKのwords corpusは古語や珍しい単語を含むため使用しない
※ spaCyは未知語にも品詞を割り当てるため、内容語のWordNet確認が必要

品詞判定ロジック:
spaCyの主要品詞を使用して同品詞/異品詞を判定する.
WordNetは複数の品詞を返すため（例: "come"がNOUN+VERB）、
一般的な用法を反映するspaCyの品詞を使用する.

編集操作:
- 置換（Replace）: 1文字を別の文字に置換
- 挿入（Insert）: 1文字を追加
- 削除（Delete）: 1文字を削除

サブワードトークン分割が変わる場合は除外（オプション）.
"""

import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.perturbation.pos_utils import (
    EditOperation,
    check_subword_preservation,
    generate_single_char_edits,
    get_spacy_pos,
    has_common_pos,
    word_exists_strict,
)
from src.utils.logger import logger

# パターン名の型定義
PatternType = Literal["pattern1", "pattern2", "pattern3"]


@dataclass
class PerturbationMapping:
    """トークンと摂動のマッピング.

    Attributes:
        original: 元のトークン
        perturbed: 摂動後のトークン
        pattern: 摂動パターン（pattern1/pattern2/pattern3）
        edit_operation: 編集操作の説明
        edit_type: 編集タイプ（replace/insert/delete）
        original_pos: 元のトークンの品詞（spaCy）
        perturbed_pos: 摂動後のトークンの品詞（spaCy）
    """

    original: str
    perturbed: str
    pattern: PatternType
    edit_operation: str
    edit_type: str = "replace"
    original_pos: str | None = None
    perturbed_pos: str | None = None

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "pattern": self.pattern,
            "edit_operation": self.edit_operation,
            "edit_type": self.edit_type,
            "original_pos": self.original_pos,
            "perturbed_pos": self.perturbed_pos,
        }


@dataclass
class AlignedPerturbationSet:
    """同一編集操作での摂動セット.

    全パターン（Pattern 1/2/3）で同じ位置・同じ編集タイプを使用する.

    Attributes:
        edit_type: 編集タイプ（replace/insert/delete）
        position: 編集位置
        pattern1: Pattern 1の摂動（同品詞実在語）
        pattern2: Pattern 2の摂動（異品詞実在語）
        pattern3: Pattern 3の摂動（非実在語）
    """

    edit_type: str
    position: int
    pattern1: PerturbationMapping | None = None
    pattern2: PerturbationMapping | None = None
    pattern3: PerturbationMapping | None = None

    def is_complete(self) -> bool:
        """全パターンが揃っているか."""
        return self.pattern1 is not None and self.pattern2 is not None and self.pattern3 is not None

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "edit_type": self.edit_type,
            "position": self.position,
            "pattern1": self.pattern1.to_dict() if self.pattern1 else None,
            "pattern2": self.pattern2.to_dict() if self.pattern2 else None,
            "pattern3": self.pattern3.to_dict() if self.pattern3 else None,
        }


@dataclass
class PatternPerturbationResult:
    """パターン摂動生成結果.

    Attributes:
        token: 対象トークン
        mappings: 各パターンへのマッピング辞書（後方互換性用、最初の摂動セット）
        aligned_sets: 同一編集操作での摂動セットリスト（複数回摂動用）
        all_candidates: 全候補リスト（デバッグ用）
    """

    token: str
    mappings: dict[PatternType, PerturbationMapping | None] = field(default_factory=dict)
    aligned_sets: list[AlignedPerturbationSet] = field(default_factory=list)
    all_candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "token": self.token,
            "mappings": {
                pattern: mapping.to_dict() if mapping else None
                for pattern, mapping in self.mappings.items()
            },
            "aligned_sets": [s.to_dict() for s in self.aligned_sets],
            "aligned_sets_count": len(self.aligned_sets),
            "all_candidates_count": len(self.all_candidates),
        }

    def has_all_patterns(self) -> bool:
        """全パターンが生成できたかどうか."""
        return all(self.mappings.get(p) is not None for p in ["pattern1", "pattern2", "pattern3"])

    def has_n_aligned_sets(self, n: int) -> bool:
        """N個以上の完全な摂動セットがあるか."""
        return len([s for s in self.aligned_sets if s.is_complete()]) >= n


@dataclass
class PerturbationCandidate:
    """摂動候補.

    Attributes:
        perturbed: 摂動後の単語
        pattern: パターン（pattern1/pattern2/pattern3）
        edit_op: 編集操作
        perturbed_pos: 摂動後の品詞（spaCy）
    """

    perturbed: str
    pattern: PatternType
    edit_op: EditOperation
    perturbed_pos: str | None = None


def classify_perturbation(
    original: str,
    perturbed: str,
    tokenizer=None,
) -> tuple[PatternType | None, str | None]:
    """摂動をパターンに分類.

    単語存在確認（厳格モード）:
    1. トークナイザーで単一トークンとして存在する
    2. かつ、spaCyまたはWordNetで品詞情報が取得できる

    品詞判定ロジック:
    spaCyの品詞を使用して同品詞/異品詞を判定する.
    WordNetは複数の品詞を返すため、一般的な用法を反映するspaCyの品詞を使用.

    Args:
        original: 元の単語
        perturbed: 摂動後の単語
        tokenizer: Transformersのトークナイザー（厳格な単語存在確認に使用）

    Returns:
        (パターン名, 摂動後の品詞) のタプル。分類不能の場合(None, None)
    """
    # 元の単語が実在語でない場合は分類不能
    if not word_exists_strict(original.lower(), tokenizer):
        return None, None

    # Pattern 3: 摂動後の単語が実在語でない（UNK）
    if not word_exists_strict(perturbed.lower(), tokenizer):
        return "pattern3", None

    # spaCyの品詞を取得
    perturbed_pos = get_spacy_pos(perturbed.lower())

    # Pattern 1 or 2: 品詞の比較（spaCyの主要品詞で判定）
    if has_common_pos(original, perturbed):
        return "pattern1", perturbed_pos
    else:
        return "pattern2", perturbed_pos


def find_perturbation_candidates(
    word: str,
    tokenizer=None,
    check_subword: bool = True,
) -> dict[PatternType, list[PerturbationCandidate]]:
    """単語に対する全ての摂動候補をパターン別に検索.

    置換・挿入・削除の全操作を考慮する.
    サブワードトークン分割が変わる場合は除外（オプション）.

    Args:
        word: 元の単語
        tokenizer: Transformersのトークナイザー（単語存在確認・サブワードチェック用）
        check_subword: サブワードトークン分割をチェックするか

    Returns:
        パターン別の摂動候補リスト
    """
    word_lower = word.lower()

    # 元の単語が実在語かチェック（厳格モード）
    if not word_exists_strict(word_lower, tokenizer):
        logger.debug(f"単語 '{word}' は実在語として認識されません")
        return {"pattern1": [], "pattern2": [], "pattern3": []}

    # 1文字編集の全候補を生成
    edit_candidates = generate_single_char_edits(word_lower)

    results: dict[PatternType, list[PerturbationCandidate]] = {
        "pattern1": [],
        "pattern2": [],
        "pattern3": [],
    }

    for candidate, edit_op in edit_candidates:
        # サブワードトークン分割をチェック
        if check_subword and tokenizer is not None:
            if not check_subword_preservation(word_lower, candidate, tokenizer):
                continue

        # パターンを分類（トークナイザーを使用した厳格な判定）
        pattern, perturbed_pos = classify_perturbation(word_lower, candidate, tokenizer)

        if pattern is None:
            continue

        perturbation = PerturbationCandidate(
            perturbed=candidate,
            pattern=pattern,
            edit_op=edit_op,
            perturbed_pos=perturbed_pos,
        )

        results[pattern].append(perturbation)

    return results


def generate_pattern3_candidate(word: str, seed: int | None = None) -> tuple[str, EditOperation]:
    """Pattern 3（非実在語）の候補を生成.

    1文字をランダムに編集（置換・挿入・削除）して非実在語を生成.

    Args:
        word: 元の単語
        seed: 乱数シード（再現性のため）

    Returns:
        (非実在語候補, 編集操作) のタプル
    """
    if seed is not None:
        random.seed(seed)

    word_lower = word.lower()

    # 編集タイプをランダムに選択
    edit_types = ["replace", "insert"]
    if len(word_lower) > 1:
        edit_types.append("delete")

    edit_type = random.choice(edit_types)

    if edit_type == "replace":
        # ランダムな位置を選択
        pos = random.randint(0, len(word_lower) - 1)
        original_char = word_lower[pos]
        # 元の文字と異なるランダムな文字を選択
        available_chars = string.ascii_lowercase.replace(original_char, "")
        new_char = random.choice(available_chars)
        result = word_lower[:pos] + new_char + word_lower[pos + 1 :]
        op = EditOperation(
            edit_type="replace",
            position=pos,
            original_char=original_char,
            new_char=new_char,
            description=f"replace '{original_char}' with '{new_char}' at pos {pos}",
        )
    elif edit_type == "insert":
        # ランダムな位置に挿入
        pos = random.randint(0, len(word_lower))
        new_char = random.choice(string.ascii_lowercase)
        result = word_lower[:pos] + new_char + word_lower[pos:]
        if pos == 0:
            desc = f"insert '{new_char}' at beginning"
        elif pos == len(word_lower):
            desc = f"insert '{new_char}' at end"
        else:
            desc = f"insert '{new_char}' at pos {pos}"
        op = EditOperation(
            edit_type="insert",
            position=pos,
            new_char=new_char,
            description=desc,
        )
    else:  # delete
        pos = random.randint(0, len(word_lower) - 1)
        original_char = word_lower[pos]
        result = word_lower[:pos] + word_lower[pos + 1 :]
        op = EditOperation(
            edit_type="delete",
            position=pos,
            original_char=original_char,
            description=f"delete '{original_char}' at pos {pos}",
        )

    return result, op


def _build_aligned_sets(
    token: str,
    candidates_by_pattern: dict[PatternType, list[PerturbationCandidate]],
    original_pos: str | None,
) -> list[AlignedPerturbationSet]:
    """同一編集位置での摂動セットを構築.

    各パターンの候補を(edit_type, position)でグループ化し、
    3パターン全てが揃う組み合わせを見つける.

    Args:
        token: 対象トークン
        candidates_by_pattern: パターン別の摂動候補
        original_pos: 元のトークンの品詞

    Returns:
        完全な摂動セットのリスト
    """
    # (edit_type, position) -> {pattern: candidate} のマッピングを構築
    position_groups: dict[tuple[str, int], dict[PatternType, PerturbationCandidate]] = {}

    for pattern_name, candidates in candidates_by_pattern.items():
        for c in candidates:
            key = (c.edit_op.edit_type, c.edit_op.position)
            if key not in position_groups:
                position_groups[key] = {}
            # 各位置で最初の候補のみ使用（決定論的）
            if pattern_name not in position_groups[key]:
                position_groups[key][pattern_name] = c

    # 3パターン全てが揃う位置のみ抽出
    aligned_sets: list[AlignedPerturbationSet] = []

    for (edit_type, position), pattern_candidates in position_groups.items():
        if len(pattern_candidates) == 3:  # 全パターンが存在
            aligned_set = AlignedPerturbationSet(
                edit_type=edit_type,
                position=position,
            )
            for pattern_name, c in pattern_candidates.items():
                mapping = PerturbationMapping(
                    original=token,
                    perturbed=c.perturbed,
                    pattern=pattern_name,
                    edit_operation=c.edit_op.description,
                    edit_type=c.edit_op.edit_type,
                    original_pos=original_pos,
                    perturbed_pos=c.perturbed_pos if pattern_name != "pattern3" else None,
                )
                if pattern_name == "pattern1":
                    aligned_set.pattern1 = mapping
                elif pattern_name == "pattern2":
                    aligned_set.pattern2 = mapping
                else:
                    aligned_set.pattern3 = mapping

            aligned_sets.append(aligned_set)

    # 編集位置でソート（決定論的な順序）
    aligned_sets.sort(key=lambda s: (s.edit_type, s.position))

    return aligned_sets


def generate_pattern_perturbations(
    token: str,
    max_candidates: int = 100,
    seed: int | None = None,
    tokenizer=None,
    check_subword: bool = True,
    num_perturbations: int = 1,
) -> PatternPerturbationResult:
    """トークンに対してパターン摂動を生成.

    Args:
        token: 対象トークン
        max_candidates: Pattern 3生成時の最大試行回数
        seed: 乱数シード（再現性のため）
        tokenizer: Transformersのトークナイザー（単語存在確認・サブワードチェック用）
        check_subword: サブワードトークン分割をチェックするか
        num_perturbations: 生成する摂動セット数（同一編集位置で全パターン揃う組み合わせ）

    Returns:
        パターン摂動結果
    """
    result = PatternPerturbationResult(token=token)

    # 元のトークンが実在語かチェック（厳格モード）
    if not word_exists_strict(token.lower(), tokenizer):
        logger.warning(f"トークン '{token}' は実在語として認識されません")
        return result

    # 品詞情報を取得（spaCyの主要品詞）
    original_pos = get_spacy_pos(token.lower())

    # 全編集候補を探索
    candidates_by_pattern = find_perturbation_candidates(
        token, tokenizer=tokenizer, check_subword=check_subword
    )

    # 全候補をフラット化してall_candidatesに格納
    all_candidates_flat = []
    for _pattern_name, candidates in candidates_by_pattern.items():
        for c in candidates:
            all_candidates_flat.append(
                {
                    "perturbed": c.perturbed,
                    "pattern": c.pattern,
                    "edit_type": c.edit_op.edit_type,
                    "edit_pos": c.edit_op.position,
                    "perturbed_pos": c.perturbed_pos,
                }
            )
    result.all_candidates = all_candidates_flat

    # 同一編集位置での摂動セットを構築
    aligned_sets = _build_aligned_sets(token, candidates_by_pattern, original_pos)
    result.aligned_sets = aligned_sets

    # 後方互換性のため、最初のセットをmappingsに格納
    if aligned_sets:
        first_set = aligned_sets[0]
        if first_set.pattern1:
            result.mappings["pattern1"] = first_set.pattern1
        if first_set.pattern2:
            result.mappings["pattern2"] = first_set.pattern2
        if first_set.pattern3:
            result.mappings["pattern3"] = first_set.pattern3
    else:
        # aligned_setsが空の場合、従来のロジックでmappingsを設定
        # Pattern 1: 同品詞の実在語
        pattern1_candidates = candidates_by_pattern.get("pattern1", [])
        if pattern1_candidates:
            c = pattern1_candidates[0]
            result.mappings["pattern1"] = PerturbationMapping(
                original=token,
                perturbed=c.perturbed,
                pattern="pattern1",
                edit_operation=c.edit_op.description,
                edit_type=c.edit_op.edit_type,
                original_pos=original_pos,
                perturbed_pos=c.perturbed_pos,
            )

        # Pattern 2: 異品詞の実在語
        pattern2_candidates = candidates_by_pattern.get("pattern2", [])
        if pattern2_candidates:
            c = pattern2_candidates[0]
            result.mappings["pattern2"] = PerturbationMapping(
                original=token,
                perturbed=c.perturbed,
                pattern="pattern2",
                edit_operation=c.edit_op.description,
                edit_type=c.edit_op.edit_type,
                original_pos=original_pos,
                perturbed_pos=c.perturbed_pos,
            )

        # Pattern 3: 非実在語
        pattern3_candidates = candidates_by_pattern.get("pattern3", [])
        if pattern3_candidates:
            c = pattern3_candidates[0]
            result.mappings["pattern3"] = PerturbationMapping(
                original=token,
                perturbed=c.perturbed,
                pattern="pattern3",
                edit_operation=c.edit_op.description,
                edit_type=c.edit_op.edit_type,
                original_pos=original_pos,
                perturbed_pos=None,
            )
        else:
            # 候補がない場合はランダム生成
            if seed is not None:
                random.seed(seed)

            for _attempt in range(max_candidates):
                candidate, edit_op = generate_pattern3_candidate(token, seed=None)

                # サブワードチェック
                if check_subword and tokenizer is not None:
                    if not check_subword_preservation(token.lower(), candidate, tokenizer):
                        continue

                # 実在語でないことを確認（UNK）- 厳格モード
                if not word_exists_strict(candidate, tokenizer):
                    result.mappings["pattern3"] = PerturbationMapping(
                        original=token,
                        perturbed=candidate,
                        pattern="pattern3",
                        edit_operation=edit_op.description,
                        edit_type=edit_op.edit_type,
                        original_pos=original_pos,
                        perturbed_pos=None,
                    )
                    break

    return result


def generate_perturbation_mapping_table(
    tokens: list[str],
    output_path: Path | None = None,
    require_all_patterns: bool = True,
    seed: int = 42,
    tokenizer=None,
    check_subword: bool = True,
    target_count: int | None = None,
    num_perturbations: int = 1,
) -> dict[str, PatternPerturbationResult]:
    """トークンリストに対してパターン摂動マッピングテーブルを生成.

    Args:
        tokens: トークンリスト
        output_path: 出力ファイルパス（JSON形式）
        require_all_patterns: 全パターンが必要な場合True
        seed: 乱数シード
        tokenizer: Transformersのトークナイザー（サブワードチェック用）
        check_subword: サブワードトークン分割をチェックするか
        target_count: フィルタ後の目標件数（Noneの場合は全件処理）
        num_perturbations: 必要な摂動セット数（同一編集位置で全パターン揃う組み合わせ）

    Returns:
        トークンからPatternPerturbationResultへのマッピング
    """
    logger.info(f"パターン摂動マッピングを生成: {len(tokens)} tokens")
    if check_subword and tokenizer is not None:
        logger.info("サブワードトークン分割チェック: 有効")
    else:
        logger.info("サブワードトークン分割チェック: 無効")
    if target_count is not None:
        logger.info(f"目標件数: {target_count}")
    if num_perturbations > 1:
        logger.info(f"必要な摂動セット数: {num_perturbations}")

    results: dict[str, PatternPerturbationResult] = {}
    success_count = 0
    partial_count = 0
    processed_count = 0

    for i, token in enumerate(tokens):
        # 目標件数に達したら終了（require_all_patternsの場合のみ有効）
        if target_count is not None and require_all_patterns and success_count >= target_count:
            logger.info(f"目標件数 {target_count} に達したため処理を終了")
            break

        # 各トークンに対して決定論的なシードを使用
        token_seed = seed + hash(token) % 10000
        result = generate_pattern_perturbations(
            token,
            seed=token_seed,
            tokenizer=tokenizer,
            check_subword=check_subword,
            num_perturbations=num_perturbations,
        )
        results[token] = result
        processed_count += 1

        # 成功判定: num_perturbations > 1 の場合は aligned_sets の数で判定
        if num_perturbations > 1:
            if result.has_n_aligned_sets(num_perturbations):
                success_count += 1
            elif result.has_all_patterns() or result.aligned_sets:
                partial_count += 1
        else:
            if result.has_all_patterns():
                success_count += 1
            elif any(result.mappings.values()):
                partial_count += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  進捗: {i + 1}/{len(tokens)} tokens (成功: {success_count})")

    logger.info(
        f"生成完了: 処理={processed_count}, 全パターン成功={success_count}, "
        f"部分成功={partial_count}, "
        f"失敗={processed_count - success_count - partial_count}"
    )

    # フィルタリング
    if require_all_patterns:
        if num_perturbations > 1:
            # 複数摂動が必要な場合は aligned_sets の数でフィルタ
            filtered_results = {
                token: result
                for token, result in results.items()
                if result.has_n_aligned_sets(num_perturbations)
            }
            logger.info(
                f"{num_perturbations}セット以上要件でフィルタ: "
                f"{len(filtered_results)}/{len(results)} tokens"
            )
        else:
            filtered_results = {
                token: result for token, result in results.items() if result.has_all_patterns()
            }
            logger.info(f"全パターン要件でフィルタ: {len(filtered_results)}/{len(results)} tokens")
        results = filtered_results

    # 出力
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "metadata": {
                "total_tokens": len(tokens),
                "successful_tokens": len(results),
                "require_all_patterns": require_all_patterns,
                "seed": seed,
                "check_subword": check_subword,
                "num_perturbations": num_perturbations,
            },
            "mappings": {token: result.to_dict() for token, result in results.items()},
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"マッピングテーブルを保存: {output_path}")

    return results


def replace_word_with_boundary(
    text: str,
    word: str,
    replacement: str,
    max_count: int = 0,
    search_end_position: int | None = None,
) -> tuple[str, int]:
    """単語境界を考慮した置換を行う.

    サブワードの部分マッチを避け、完全な単語のみを置換する.
    例: "following"の中の"follow"は置換しない.

    Args:
        text: 置換対象のテキスト
        word: 置換する単語
        replacement: 置換後の単語
        max_count: 最大置換回数（0=無制限）
        search_end_position: 検索範囲の終了位置（この位置より前のみを検索対象とする）
                            Noneの場合は全体を対象とする

    Returns:
        (置換後テキスト, 置換回数)
    """
    # 単語境界を考慮した正規表現パターン
    # \b は単語境界（英数字と非英数字の境目）にマッチ
    pattern = r"\b" + re.escape(word) + r"\b"

    if search_end_position is not None:
        # 検索範囲が指定されている場合、範囲内のみで置換
        search_text = text[:search_end_position]
        rest_text = text[search_end_position:]

        if max_count == 0:
            result_search, count = re.subn(
                pattern, replacement, search_text, flags=re.IGNORECASE
            )
        else:
            result_search, count = re.subn(
                pattern, replacement, search_text, count=max_count, flags=re.IGNORECASE
            )

        return result_search + rest_text, count
    else:
        # 検索範囲が指定されていない場合、全体を対象
        if max_count == 0:
            result, count = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        else:
            result, count = re.subn(
                pattern, replacement, text, count=max_count, flags=re.IGNORECASE
            )

        return result, count


def apply_pattern_perturbation(
    text: str,
    mapping_table: dict[str, PatternPerturbationResult],
    pattern: PatternType,
    max_perturbations: int = 0,
    search_end_position: int | None = None,
) -> tuple[str, list[dict]]:
    """テキストにパターン摂動を適用.

    Args:
        text: 入力テキスト
        mapping_table: 摂動マッピングテーブル
        pattern: 適用するパターン
        max_perturbations: 最大摂動回数（0=無制限、1=1回のみ、など）
        search_end_position: 検索範囲の終了位置（この位置より前のみを検索対象とする）
                            Noneの場合は全体を対象とする.
                            MMLUなどで選択肢を除外する場合に使用.

    Returns:
        (摂動後テキスト, 適用された摂動のリスト)
    """
    applied_perturbations = []
    result_text = text
    total_perturbation_count = 0

    # 検索対象テキスト（検索範囲が指定されている場合は範囲内のみ）
    search_text = text[:search_end_position] if search_end_position else text

    # マッピングテーブル内のトークンを長い順にソート（部分一致を避けるため）
    sorted_tokens = sorted(mapping_table.keys(), key=len, reverse=True)

    for token in sorted_tokens:
        # 最大摂動回数に達したら終了
        if max_perturbations > 0 and total_perturbation_count >= max_perturbations:
            break

        # 単語境界を考慮したマッチング確認（検索範囲内のみ）
        word_pattern = r"\b" + re.escape(token) + r"\b"
        if not re.search(word_pattern, search_text, flags=re.IGNORECASE):
            continue

        mapping_result = mapping_table[token]
        perturbation = mapping_result.mappings.get(pattern)

        if perturbation is None:
            continue

        if max_perturbations > 0:
            # 最大摂動回数が指定されている場合、最初の1つのみ置換
            result_text, count = replace_word_with_boundary(
                result_text,
                token,
                perturbation.perturbed,
                max_count=1,
                search_end_position=search_end_position,
            )
            if count > 0:
                total_perturbation_count += 1
        else:
            # 無制限の場合、テキスト内の全出現を置換（範囲内のみ）
            result_text, count = replace_word_with_boundary(
                result_text,
                token,
                perturbation.perturbed,
                max_count=0,
                search_end_position=search_end_position,
            )

        if count > 0:
            applied_perturbations.append(
                {
                    "original": token,
                    "perturbed": perturbation.perturbed,
                    "pattern": pattern,
                    "edit_type": perturbation.edit_type,
                    "count": count,
                }
            )

    return result_text, applied_perturbations


def load_mapping_table(path: Path) -> dict[str, PatternPerturbationResult]:
    """マッピングテーブルをファイルから読み込み.

    Args:
        path: マッピングテーブルのJSONファイルパス

    Returns:
        マッピングテーブル
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results: dict[str, PatternPerturbationResult] = {}

    for token, result_data in data["mappings"].items():
        mappings: dict[PatternType, PerturbationMapping | None] = {}

        for pattern_name, mapping_data in result_data["mappings"].items():
            if mapping_data is None:
                mappings[pattern_name] = None
            else:
                # 品詞データの読み込み（文字列または旧形式のリストに対応）
                original_pos_data = mapping_data.get("original_pos")
                perturbed_pos_data = mapping_data.get("perturbed_pos")

                # 旧形式（リスト）の場合は最初の要素を使用、新形式（文字列）はそのまま
                if isinstance(original_pos_data, list):
                    original_pos = original_pos_data[0] if original_pos_data else None
                else:
                    original_pos = original_pos_data

                if isinstance(perturbed_pos_data, list):
                    perturbed_pos = perturbed_pos_data[0] if perturbed_pos_data else None
                else:
                    perturbed_pos = perturbed_pos_data

                mappings[pattern_name] = PerturbationMapping(
                    original=mapping_data["original"],
                    perturbed=mapping_data["perturbed"],
                    pattern=mapping_data["pattern"],
                    edit_operation=mapping_data["edit_operation"],
                    edit_type=mapping_data.get("edit_type", "replace"),
                    original_pos=original_pos,
                    perturbed_pos=perturbed_pos,
                )

        results[token] = PatternPerturbationResult(
            token=token,
            mappings=mappings,
            all_candidates=[],
        )

    logger.info(f"マッピングテーブルを読み込み: {len(results)} tokens from {path}")
    return results


@dataclass
class AlignedPerturbationApplyResult:
    """同一箇所摂動適用結果.

    全パターンで同じ箇所に摂動を適用した結果.

    Attributes:
        perturbed_texts: 各パターンの摂動後テキスト
        applied_perturbations: 各パターンの適用された摂動詳細
        target_tokens: 摂動対象として選択されたトークンリスト
    """

    perturbed_texts: dict[PatternType, str]
    applied_perturbations: dict[PatternType, list[dict]]
    target_tokens: list[str]

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "perturbed_texts": self.perturbed_texts,
            "applied_perturbations": self.applied_perturbations,
            "target_tokens": self.target_tokens,
        }


def apply_aligned_perturbations(
    text: str,
    mapping_table: dict[str, PatternPerturbationResult],
    num_perturbations: int = 1,
    search_end_position: int | None = None,
) -> AlignedPerturbationApplyResult | None:
    """テキストに対して全パターンで同じ箇所に摂動を適用.

    複数回摂動を行う場合、Pattern 1/2/3 全てで同じ単語に摂動を適用する.
    これにより、パターン間で公平な比較が可能になる.

    Args:
        text: 入力テキスト
        mapping_table: 摂動マッピングテーブル
        num_perturbations: 摂動する単語の数
        search_end_position: 検索範囲の終了位置（この位置より前のみを検索対象とする）
                            Noneの場合は全体を対象とする.
                            MMLUなどで選択肢を除外する場合に使用.

    Returns:
        AlignedPerturbationApplyResult: 各パターンの摂動テキストと詳細
        全パターンで摂動可能な単語が十分にない場合はNone
    """
    patterns: list[PatternType] = ["pattern1", "pattern2", "pattern3"]

    # 検索対象テキスト（検索範囲が指定されている場合は範囲内のみ）
    search_text = text[:search_end_position] if search_end_position else text

    # 全パターンが揃っているトークンをテキスト内から探す
    # マッピングテーブル内のトークンを長い順にソート（部分一致を避けるため）
    sorted_tokens = sorted(mapping_table.keys(), key=len, reverse=True)

    # テキスト内に存在し、かつ全パターンが揃っているトークンを収集
    # 単語境界を考慮してマッチング（検索範囲内のみ）
    available_tokens: list[str] = []
    for token in sorted_tokens:
        # 単語境界を考慮したマッチング確認（検索範囲内のみ）
        word_pattern = r"\b" + re.escape(token) + r"\b"
        if not re.search(word_pattern, search_text, flags=re.IGNORECASE):
            continue

        mapping_result = mapping_table[token]

        # 全パターンが揃っているか確認
        if not mapping_result.has_all_patterns():
            continue

        available_tokens.append(token)

        # 必要数に達したら終了
        if len(available_tokens) >= num_perturbations:
            break

    # 必要数に満たない場合はNone
    if len(available_tokens) < num_perturbations:
        logger.debug(
            f"全パターンで摂動可能な単語が不足: "
            f"必要={num_perturbations}, 利用可能={len(available_tokens)}"
        )
        return None

    # 各パターンで同じ単語に摂動を適用
    perturbed_texts: dict[PatternType, str] = {}
    applied_perturbations: dict[PatternType, list[dict]] = {}

    for pattern in patterns:
        result_text = text
        applied_list: list[dict] = []

        for token in available_tokens:
            mapping_result = mapping_table[token]
            perturbation = mapping_result.mappings.get(pattern)

            if perturbation is None:
                # 理論上ここには来ないはず（has_all_patternsでチェック済み）
                continue

            # 最初の1つのみ置換（同じ単語が複数回出現する場合）
            # 単語境界を考慮した置換を使用（検索範囲内のみ）
            result_text, count = replace_word_with_boundary(
                result_text,
                token,
                perturbation.perturbed,
                max_count=1,
                search_end_position=search_end_position,
            )

            if count > 0:
                applied_list.append(
                    {
                        "original": token,
                        "perturbed": perturbation.perturbed,
                        "pattern": pattern,
                        "edit_type": perturbation.edit_type,
                        "edit_operation": perturbation.edit_operation,
                        "original_pos": perturbation.original_pos,
                        "perturbed_pos": perturbation.perturbed_pos,
                    }
                )

        perturbed_texts[pattern] = result_text
        applied_perturbations[pattern] = applied_list

    return AlignedPerturbationApplyResult(
        perturbed_texts=perturbed_texts,
        applied_perturbations=applied_perturbations,
        target_tokens=available_tokens,
    )
