#!/usr/bin/env python3
"""摂動マッピング生成スクリプト.

頻出単語に対する摂動パターン（Pattern 1/2/3）のマッピングを
事前に生成してJSONファイルに保存する.

単語存在確認（厳格モード）:
- トークナイザーで単一トークンとして存在
- かつ、spaCyまたはWordNetで品詞情報が取得できる

使用例:
    # 英語頻出単語から摂動マッピングを生成
    PYTHONPATH=. uv run python scripts/generate_perturbation_mapping.py \
        --input data/processed/english/frequent_words_top300.json \
        --model gemma-3-1b-it \
        --output data/perturbation_mappings/english_top300.json

    # ベンチマークから抽出した単語を使用
    PYTHONPATH=. uv run python scripts/generate_perturbation_mapping.py \
        --benchmark gsm8k --top-n 200 \
        --model gemma-3-1b-it \
        --output data/perturbation_mappings/gsm8k_top200.json

    # 全パターン生成可能な単語を200件取得（目標件数指定）
    PYTHONPATH=. uv run python scripts/generate_perturbation_mapping.py \
        --benchmark gsm8k --top-n 1000 \
        --model gemma-3-1b-it \
        --require-all-patterns --target-count 200 \
        --output data/perturbation_mappings/gsm8k_allpatterns_200.json
"""

import argparse
import json
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer

from src.experiment2.pattern_perturbation.pattern_generator import (
    generate_perturbation_mapping_table,
)
from src.models.model_loader import SUPPORTED_MODELS
from src.utils.logger import logger


def load_words_from_file(file_path: Path) -> list[str]:
    """ファイルから単語リストを読み込み.

    Args:
        file_path: 入力ファイルパス（JSON形式）

    Returns:
        単語リスト
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # frequent_words形式のJSONの場合
    if "words" in data:
        return [w["word"] for w in data["words"]]
    # tokens形式の場合
    if "tokens" in data:
        return [t["token"] for t in data["tokens"]]
    # リスト形式の場合
    if isinstance(data, list):
        return data

    raise ValueError(f"不明なJSON形式: {file_path}")


def extract_words_from_benchmark(benchmark_name: str, top_n: int = 200) -> list[str]:
    """ベンチマークから頻出単語を抽出.

    Args:
        benchmark_name: ベンチマーク名
        top_n: 上位N件

    Returns:
        単語リスト
    """
    import re
    from collections import Counter

    from src.benchmarks.benchmark_loader import load_benchmark
    from src.experiment2.token_extraction.benchmark_token_extractor import (
        extract_question_text,
    )

    # ベンチマークをロード
    benchmark_data = load_benchmark(name=benchmark_name, max_samples=None)
    examples = benchmark_data.examples

    # 単語を抽出
    word_counter: Counter = Counter()
    for example in examples:
        text = extract_question_text(example, benchmark_name)
        # 英単語を抽出（3文字以上）
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        word_counter.update(words)

    # 上位N件を返す
    return [word for word, _ in word_counter.most_common(top_n)]


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析."""
    parser = argparse.ArgumentParser(
        description="摂動マッピング生成: 頻出単語に対する摂動パターンを事前生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 入力ソース（いずれか必須）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="入力ファイルパス（頻出単語リストのJSON）",
    )
    input_group.add_argument(
        "--benchmark",
        type=str,
        choices=["gsm8k", "bbh", "mmlu", "truthfulqa"],
        help="ベンチマーク名（ベンチマークから単語を抽出）",
    )

    # 出力
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力ファイルパス（JSON形式）",
    )

    # オプション
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="ベンチマークから抽出する上位N件（デフォルト: 200）",
    )
    parser.add_argument(
        "--require-all-patterns",
        action="store_true",
        help="全パターン（Pattern 1/2/3）が生成できた単語のみ保存",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="使用するモデル名（トークナイザーのロードに使用）",
    )
    parser.add_argument(
        "--check-subword",
        action="store_true",
        help="サブワードトークン分割をチェック（摂動後もトークン数が同じか確認）",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="目標件数（--require-all-patterns時、この件数に達したら処理を終了）",
    )
    parser.add_argument(
        "--num-perturbations",
        type=int,
        default=1,
        help="1文あたりの摂動セット数（デフォルト: 1）。"
        "複数指定時、全パターンで同一位置・同一編集タイプの摂動を生成",
    )

    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("摂動マッピング生成")
    logger.info("=" * 60)

    # トークナイザーをロード
    if args.model in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[args.model].get("hf_name", args.model)
    else:
        hf_name = args.model
    logger.info(f"トークナイザーをロード中: {hf_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        logger.info(f"トークナイザーロード完了: vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"トークナイザーのロードに失敗: {e}")
        sys.exit(1)

    # 単語リストを取得
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"入力ファイルが見つかりません: {input_path}")
            sys.exit(1)
        words = load_words_from_file(input_path)
        logger.info(f"入力ファイル: {input_path}")
    else:
        words = extract_words_from_benchmark(args.benchmark, args.top_n)
        logger.info(f"ベンチマーク: {args.benchmark}, 上位{args.top_n}件")

    logger.info(f"対象単語数: {len(words)}")
    logger.info(f"出力先: {args.output}")
    logger.info(f"全パターン必須: {args.require_all_patterns}")
    logger.info(f"目標件数: {args.target_count if args.target_count else '指定なし'}")
    logger.info(f"乱数シード: {args.seed}")
    logger.info(f"サブワードチェック: {args.check_subword}")
    logger.info(f"摂動セット数: {args.num_perturbations}")
    logger.info("=" * 60)

    # 摂動マッピングを生成
    output_path = Path(args.output)
    mapping_table = generate_perturbation_mapping_table(
        tokens=words,
        output_path=output_path,
        require_all_patterns=args.require_all_patterns,
        seed=args.seed,
        tokenizer=tokenizer,
        check_subword=args.check_subword,
        target_count=args.target_count,
        num_perturbations=args.num_perturbations,
    )

    # 統計を表示
    total = len(words)
    success = len(mapping_table)
    pattern_counts = {"pattern1": 0, "pattern2": 0, "pattern3": 0}
    aligned_sets_stats = []  # 各トークンの aligned_sets 数

    for result in mapping_table.values():
        for pattern in ["pattern1", "pattern2", "pattern3"]:
            if result.mappings.get(pattern) is not None:
                pattern_counts[pattern] += 1
        # aligned_sets の統計
        complete_sets = len([s for s in result.aligned_sets if s.is_complete()])
        aligned_sets_stats.append(complete_sets)

    logger.info("")
    logger.info("=" * 60)
    logger.info("生成結果サマリー")
    logger.info("=" * 60)
    logger.info(f"入力単語数: {total}")
    logger.info(f"マッピング生成成功: {success} ({success / total * 100:.1f}%)")
    logger.info(f"  Pattern 1 (同品詞): {pattern_counts['pattern1']}")
    logger.info(f"  Pattern 2 (異品詞): {pattern_counts['pattern2']}")
    logger.info(f"  Pattern 3 (非実在): {pattern_counts['pattern3']}")
    if args.num_perturbations > 1:
        # 複数摂動セットの統計
        min_sets = min(aligned_sets_stats) if aligned_sets_stats else 0
        max_sets = max(aligned_sets_stats) if aligned_sets_stats else 0
        avg_sets = sum(aligned_sets_stats) / len(aligned_sets_stats) if aligned_sets_stats else 0
        logger.info(
            f"  同一位置摂動セット数: 最小={min_sets}, 最大={max_sets}, 平均={avg_sets:.1f}"
        )
        # 必要数を満たすトークン数
        sufficient_count = sum(1 for s in aligned_sets_stats if s >= args.num_perturbations)
        logger.info(f"  {args.num_perturbations}セット以上: {sufficient_count}/{success}")
    logger.info(f"出力ファイル: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
