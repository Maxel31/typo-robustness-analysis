"""前処理実行スクリプト.

言語リソースごとに頻出単語を抽出する.

使用方法:
    # 両言語を処理（デフォルトN=500）
    uv run python scripts/run_preprocessing.py

    # 英語のみ処理
    uv run python scripts/run_preprocessing.py --language english

    # 日本語のみ処理
    uv run python scripts/run_preprocessing.py --language japanese

    # Nを指定
    uv run python scripts/run_preprocessing.py --top-n 1000

    # 両方指定
    uv run python scripts/run_preprocessing.py --language english --top-n 300
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.english_words import extract_english_frequent_words
from src.preprocessing.japanese_words import extract_japanese_frequent_words
from src.utils.config import DEFAULT_TOP_N
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする.

    Returns:
        パースされた引数
    """
    parser = argparse.ArgumentParser(
        description="言語リソースごとに頻出単語を抽出する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        choices=["english", "japanese", "both"],
        default="both",
        help="処理する言語 (default: both)",
    )

    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"抽出する単語数 (default: {DEFAULT_TOP_N})",
    )

    return parser.parse_args()


def main() -> None:
    """メイン関数."""
    args = parse_args()

    logger.info(f"=== 前処理を開始: language={args.language}, top_n={args.top_n} ===")

    results: dict[str, Path] = {}

    # 英語処理
    if args.language in ["english", "both"]:
        try:
            result_path = extract_english_frequent_words(top_n=args.top_n)
            results["english"] = result_path
            logger.info(f"英語処理完了: {result_path}")
        except FileNotFoundError as e:
            logger.error(f"英語処理エラー: {e}")
            if args.language == "english":
                sys.exit(1)

    # 日本語処理
    if args.language in ["japanese", "both"]:
        try:
            result_path = extract_japanese_frequent_words(top_n=args.top_n)
            results["japanese"] = result_path
            logger.info(f"日本語処理完了: {result_path}")
        except FileNotFoundError as e:
            logger.error(f"日本語処理エラー: {e}")
            if args.language == "japanese":
                sys.exit(1)

    # 結果サマリー
    logger.info("=== 前処理完了 ===")
    for lang, path in results.items():
        logger.info(f"  {lang}: {path}")


if __name__ == "__main__":
    main()
