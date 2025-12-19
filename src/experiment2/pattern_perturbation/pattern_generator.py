"""パターン摂動生成モジュール.

決定論的なパターン摂動（Pattern 1/2/3）を生成する.
ベンチマーク間での比較を可能にするため、
同じトークンに対して同じ摂動を適用する.

摂動パターン:
- Pattern 1: 摂動後のトークンが実在 + 同品詞
- Pattern 2: 摂動後のトークンが実在 + 異品詞
- Pattern 3: 摂動後のトークンが非実在（UNK）
"""

import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.perturbation.wordnet_utils import (
    classify_perturbation,
    find_perturbation_candidates,
    get_word_info,
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
        original_pos: 元のトークンの品詞セット
        perturbed_pos: 摂動後のトークンの品詞セット
    """

    original: str
    perturbed: str
    pattern: PatternType
    edit_operation: str
    original_pos: set[str] = field(default_factory=set)
    perturbed_pos: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "pattern": self.pattern,
            "edit_operation": self.edit_operation,
            "original_pos": list(self.original_pos),
            "perturbed_pos": list(self.perturbed_pos),
        }


@dataclass
class PatternPerturbationResult:
    """パターン摂動生成結果.

    Attributes:
        token: 対象トークン
        mappings: 各パターンへのマッピング辞書
        all_candidates: 全候補リスト（デバッグ用）
    """

    token: str
    mappings: dict[PatternType, PerturbationMapping | None] = field(default_factory=dict)
    all_candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "token": self.token,
            "mappings": {
                pattern: mapping.to_dict() if mapping else None
                for pattern, mapping in self.mappings.items()
            },
            "all_candidates_count": len(self.all_candidates),
        }

    def has_all_patterns(self) -> bool:
        """全パターンが生成できたかどうか."""
        return all(self.mappings.get(p) is not None for p in ["pattern1", "pattern2", "pattern3"])


def generate_pattern3_candidate(word: str, seed: int | None = None) -> str:
    """Pattern 3（非実在語）の候補を生成.

    1文字をランダムな文字に置換して非実在語を生成.

    Args:
        word: 元の単語
        seed: 乱数シード（再現性のため）

    Returns:
        非実在語候補
    """
    if seed is not None:
        random.seed(seed)

    if len(word) < 2:
        # 1文字の単語は置換
        new_char = random.choice(string.ascii_lowercase.replace(word.lower(), ""))
        return new_char

    # ランダムな位置を選択
    pos = random.randint(0, len(word) - 1)
    original_char = word[pos].lower()

    # 元の文字と異なるランダムな文字を選択
    available_chars = string.ascii_lowercase.replace(original_char, "")
    new_char = random.choice(available_chars)

    # 大文字小文字を保持
    if word[pos].isupper():
        new_char = new_char.upper()

    result = word[:pos] + new_char + word[pos + 1 :]
    return result


def generate_pattern_perturbations(
    token: str,
    max_candidates: int = 100,
    seed: int | None = None,
) -> PatternPerturbationResult:
    """トークンに対してパターン摂動を生成.

    Args:
        token: 対象トークン
        max_candidates: Pattern 3生成時の最大試行回数
        seed: 乱数シード（再現性のため）

    Returns:
        パターン摂動結果
    """
    result = PatternPerturbationResult(token=token)

    # 元のトークン情報を取得
    original_info = get_word_info(token)
    if not original_info.exists:
        logger.warning(f"トークン '{token}' はWordNetに存在しません")
        return result

    # 1文字置換の候補を探索
    candidates_by_pattern = find_perturbation_candidates(token)

    # 編集位置を特定するヘルパー関数
    def find_edit_pos(original: str, perturbed: str) -> int:
        for i, (a, b) in enumerate(zip(original.lower(), perturbed.lower(), strict=False)):
            if a != b:
                return i
        return -1

    # 全候補をフラット化してall_candidatesに格納
    all_candidates_flat = []
    for _, candidates in candidates_by_pattern.items():
        for c in candidates:
            edit_pos = find_edit_pos(token, c.perturbed)
            all_candidates_flat.append(
                {
                    "perturbed": c.perturbed,
                    "pattern": c.pattern,
                    "edit_pos": edit_pos,
                }
            )
    result.all_candidates = all_candidates_flat

    # Pattern 1: 同品詞の実在語
    pattern1_candidates = candidates_by_pattern.get("pattern1", [])
    if pattern1_candidates:
        # 最初の候補を使用（決定論的）
        c = pattern1_candidates[0]
        perturbed_info = get_word_info(c.perturbed)
        pos = find_edit_pos(token, c.perturbed)
        edit_op = f"replace '{token[pos]}' with '{c.perturbed[pos]}' at pos {pos}"
        result.mappings["pattern1"] = PerturbationMapping(
            original=token,
            perturbed=c.perturbed,
            pattern="pattern1",
            edit_operation=edit_op,
            original_pos=original_info.pos_set,
            perturbed_pos=perturbed_info.pos_set if perturbed_info else set(),
        )

    # Pattern 2: 異品詞の実在語
    pattern2_candidates = candidates_by_pattern.get("pattern2", [])
    if pattern2_candidates:
        c = pattern2_candidates[0]
        perturbed_info = get_word_info(c.perturbed)
        pos = find_edit_pos(token, c.perturbed)
        edit_op = f"replace '{token[pos]}' with '{c.perturbed[pos]}' at pos {pos}"
        result.mappings["pattern2"] = PerturbationMapping(
            original=token,
            perturbed=c.perturbed,
            pattern="pattern2",
            edit_operation=edit_op,
            original_pos=original_info.pos_set,
            perturbed_pos=perturbed_info.pos_set if perturbed_info else set(),
        )

    # Pattern 3: 非実在語
    if seed is not None:
        random.seed(seed)

    for _attempt in range(max_candidates):
        candidate = generate_pattern3_candidate(token, seed=None)  # seedは上で設定済み
        pattern = classify_perturbation(token, candidate)

        if pattern == "pattern3":
            # 置換位置を特定
            edit_pos = -1
            for i, (a, b) in enumerate(zip(token, candidate, strict=False)):
                if a != b:
                    edit_pos = i
                    break

            edit_op = f"replace '{token[edit_pos]}' with '{candidate[edit_pos]}' at pos {edit_pos}"
            result.mappings["pattern3"] = PerturbationMapping(
                original=token,
                perturbed=candidate,
                pattern="pattern3",
                edit_operation=edit_op,
                original_pos=original_info.pos_set,
                perturbed_pos=set(),
            )
            break

    return result


def generate_perturbation_mapping_table(
    tokens: list[str],
    output_path: Path | None = None,
    require_all_patterns: bool = True,
    seed: int = 42,
) -> dict[str, PatternPerturbationResult]:
    """トークンリストに対してパターン摂動マッピングテーブルを生成.

    Args:
        tokens: トークンリスト
        output_path: 出力ファイルパス（JSON形式）
        require_all_patterns: 全パターンが必要な場合True
        seed: 乱数シード

    Returns:
        トークンからPatternPerturbationResultへのマッピング
    """
    logger.info(f"パターン摂動マッピングを生成: {len(tokens)} tokens")

    results: dict[str, PatternPerturbationResult] = {}
    success_count = 0
    partial_count = 0

    for i, token in enumerate(tokens):
        # 各トークンに対して決定論的なシードを使用
        token_seed = seed + hash(token) % 10000
        result = generate_pattern_perturbations(token, seed=token_seed)
        results[token] = result

        if result.has_all_patterns():
            success_count += 1
        elif any(result.mappings.values()):
            partial_count += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  進捗: {i + 1}/{len(tokens)} tokens")

    logger.info(
        f"生成完了: 全パターン成功={success_count}, "
        f"部分成功={partial_count}, "
        f"失敗={len(tokens) - success_count - partial_count}"
    )

    # フィルタリング
    if require_all_patterns:
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
            },
            "mappings": {token: result.to_dict() for token, result in results.items()},
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"マッピングテーブルを保存: {output_path}")

    return results


def apply_pattern_perturbation(
    text: str,
    mapping_table: dict[str, PatternPerturbationResult],
    pattern: PatternType,
) -> tuple[str, list[dict]]:
    """テキストにパターン摂動を適用.

    Args:
        text: 入力テキスト
        mapping_table: 摂動マッピングテーブル
        pattern: 適用するパターン

    Returns:
        (摂動後テキスト, 適用された摂動のリスト)
    """
    applied_perturbations = []
    result_text = text

    # マッピングテーブル内のトークンを長い順にソート（部分一致を避けるため）
    sorted_tokens = sorted(mapping_table.keys(), key=len, reverse=True)

    for token in sorted_tokens:
        if token not in result_text:
            continue

        mapping_result = mapping_table[token]
        perturbation = mapping_result.mappings.get(pattern)

        if perturbation is None:
            continue

        # テキスト内の全出現を置換
        count = result_text.count(token)
        result_text = result_text.replace(token, perturbation.perturbed)

        applied_perturbations.append(
            {
                "original": token,
                "perturbed": perturbation.perturbed,
                "pattern": pattern,
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
                mappings[pattern_name] = PerturbationMapping(
                    original=mapping_data["original"],
                    perturbed=mapping_data["perturbed"],
                    pattern=mapping_data["pattern"],
                    edit_operation=mapping_data["edit_operation"],
                    original_pos=set(mapping_data.get("original_pos", [])),
                    perturbed_pos=set(mapping_data.get("perturbed_pos", [])),
                )

        results[token] = PatternPerturbationResult(
            token=token,
            mappings=mappings,
            all_candidates=[],
        )

    logger.info(f"マッピングテーブルを読み込み: {len(results)} tokens from {path}")
    return results
