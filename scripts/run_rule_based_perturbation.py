#!/usr/bin/env python
"""規則ベースの摂動生成スクリプト.

頻出単語リストを入力として、1文字置換のtypoで変化しうる単語のうち、
パターン.1（置換後の単語が存在し、かつ置換前の単語と品詞が一致する）に該当するものを列挙する.

使用例:
    # 英語頻出単語（上位100件）でテスト
    PYTHONPATH=. uv run python scripts/run_rule_based_perturbation.py \\
        --input data/processed/english/frequent_words_top2000.json \\
        --output data/processed/english/pattern1_perturbations.json \\
        --top-n 100

    # 全件処理
    PYTHONPATH=. uv run python scripts/run_rule_based_perturbation.py \\
        --input data/processed/english/frequent_words_top2000.json \\
        --output data/processed/english/pattern1_perturbations_top2000.json
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perturbation.rule_based_perturbator import process_frequent_words
from src.utils.logger import logger


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="規則ベースの摂動生成（パターン.1: 品詞一致）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
パターンの説明:
  パターン.1: 置換後の単語がWordNetに存在し、かつ置換前の単語と品詞が一致する場合
              例: much -> such（両方とも副詞として使用可能）

出力形式:
  {
    "metadata": {...},
    "results": [
      {
        "rank": 1,
        "original_word": "you",
        "perturbed_words": ["you", ...]  // またはnull
      },
      ...
    ]
  }
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="入力JSONファイルのパス（frequent_words_top{n}.json）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="出力JSONファイルのパス",
    )
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=None,
        help="処理する上位N件（指定しない場合は全件処理）",
    )

    args = parser.parse_args()

    # 入力ファイルの存在確認
    if not args.input.exists():
        logger.error(f"入力ファイルが見つかりません: {args.input}")
        sys.exit(1)

    # 処理開始
    logger.info("=" * 60)
    logger.info("規則ベース摂動生成を開始")
    logger.info(f"入力ファイル: {args.input}")
    logger.info(f"出力ファイル: {args.output}")
    if args.top_n:
        logger.info(f"処理件数: 上位{args.top_n}件")
    else:
        logger.info("処理件数: 全件")
    logger.info("=" * 60)

    start_time = datetime.now(UTC)

    # 処理実行
    results = process_frequent_words(
        input_path=args.input,
        output_path=args.output,
        top_n=args.top_n,
    )

    end_time = datetime.now(UTC)
    elapsed = (end_time - start_time).total_seconds()

    # 結果サマリーを表示
    logger.info("=" * 60)
    logger.info("処理完了")
    logger.info(f"処理時間: {elapsed:.2f}秒")
    logger.info(f"処理単語数: {len(results)}")

    words_with_perturbations = sum(1 for r in results if r.perturbed_words)
    logger.info(
        f"パターン.1該当単語数: {words_with_perturbations}/{len(results)} "
        f"({words_with_perturbations / len(results) * 100:.1f}%)"
    )

    # 上位5件の例を表示
    logger.info("\n=== 例（上位5件） ===")
    for result in results[:5]:
        if result.perturbed_words:
            words_preview = result.perturbed_words[:5]
            if len(result.perturbed_words) > 5:
                words_preview_str = ", ".join(words_preview) + f", ... (計{len(result.perturbed_words)}件)"
            else:
                words_preview_str = ", ".join(words_preview)
            logger.info(f"  {result.original_word} -> [{words_preview_str}]")
        else:
            logger.info(f"  {result.original_word} -> (該当なし)")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
