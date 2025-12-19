#!/usr/bin/env python3
"""ケーススタディ分析実行スクリプト.

少数のサンプルを使用して、摂動パターンによる
エントロピー変化を詳細に分析する.

使用例:
    # GSM8Kで10サンプルのケーススタディを実行
    PYTHONPATH=. uv run python scripts/run_case_study.py \
        --benchmark gsm8k --model gemma-3-1b-it --num-samples 10

    # クイック分析（5サンプル、デバッグ用）
    PYTHONPATH=. uv run python scripts/run_case_study.py \
        --benchmark gsm8k --model gemma-3-1b-it --quick

    # BBHの特定サブセットで分析
    PYTHONPATH=. uv run python scripts/run_case_study.py \
        --benchmark bbh --subset boolean_expressions --model gemma-3-1b-it
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiment2.entropy_analysis.case_study_analyzer import (
    run_case_study,
)
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析."""
    parser = argparse.ArgumentParser(
        description="ケーススタディ分析: 摂動パターンによるエントロピー変化を分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # GSM8Kで10サンプルのケーススタディを実行
  PYTHONPATH=. uv run python scripts/run_case_study.py \\
      --benchmark gsm8k --model gemma-3-1b-it --num-samples 10

  # クイック分析（5サンプル、デバッグ用）
  PYTHONPATH=. uv run python scripts/run_case_study.py \\
      --benchmark gsm8k --model gemma-3-1b-it --quick

  # 出力先を指定
  PYTHONPATH=. uv run python scripts/run_case_study.py \\
      --benchmark gsm8k --model gemma-3-1b-it \\
      --output results/experiment2/case_study/my_analysis.json
        """,
    )

    # 必須引数
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["gsm8k", "bbh", "mmlu"],
        help="ベンチマーク名",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="モデル名（例: gemma-3-1b-it, Mistral-7B-Instruct-v0.3）",
    )

    # オプション引数
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="分析するサンプル数（デフォルト: 10）",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="サブセット名（BBH/MMLUの場合）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="生成する最大トークン数（デフォルト: 50）",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="使用するGPU ID（デフォルト: 0）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ファイルパス（省略時は自動生成）",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="クイック分析モード（5サンプル、30トークン）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="サンプル選択の乱数シード（デフォルト: 42）",
    )

    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    # クイックモードの設定
    if args.quick:
        num_samples = 5
        max_new_tokens = 30
        logger.info("クイック分析モード: 5サンプル、30トークン")
    else:
        num_samples = args.num_samples
        max_new_tokens = args.max_new_tokens

    # 出力パスの設定
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("results/experiment2/case_study")
        output_dir.mkdir(parents=True, exist_ok=True)
        subset_suffix = f"_{args.subset}" if args.subset else ""
        model_safe = args.model.replace("/", "_").replace("-", "_")
        output_path = output_dir / f"{args.benchmark}{subset_suffix}_{model_safe}_analysis.json"

    logger.info("=" * 60)
    logger.info("ケーススタディ分析")
    logger.info("=" * 60)
    logger.info(f"ベンチマーク: {args.benchmark}")
    logger.info(f"モデル: {args.model}")
    logger.info(f"サンプル数: {num_samples}")
    logger.info(f"最大生成トークン数: {max_new_tokens}")
    logger.info(f"GPU ID: {args.gpu_id}")
    logger.info(f"乱数シード: {args.seed}")
    logger.info(f"出力先: {output_path}")
    if args.subset:
        logger.info(f"サブセット: {args.subset}")
    logger.info("=" * 60)

    # ケーススタディを実行
    result = run_case_study(
        benchmark_name=args.benchmark,
        model_name=args.model,
        num_samples=num_samples,
        subset=args.subset,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )

    # 結果サマリーを表示
    logger.info("")
    logger.info("=" * 60)
    logger.info("分析結果サマリー")
    logger.info("=" * 60)
    logger.info(f"処理サンプル数: {len(result.samples)}")

    if result.aggregate_statistics:
        logger.info("")
        logger.info("パターン別平均エントロピー:")
        for pattern in ["original", "pattern1", "pattern2", "pattern3"]:
            if pattern in result.aggregate_statistics:
                stats = result.aggregate_statistics[pattern]
                logger.info(
                    f"  {pattern:10s}: "
                    f"平均={stats['avg_mean_entropy']:.4f}, "
                    f"最大={stats['avg_max_entropy']:.4f}"
                )

        if "differences" in result.aggregate_statistics:
            logger.info("")
            logger.info("オリジナルからの変化:")
            for pattern, diff in result.aggregate_statistics["differences"].items():
                increase = diff["avg_entropy_increase"]
                ratio = diff["avg_entropy_ratio"]
                direction = "↑" if increase > 0 else "↓"
                logger.info(
                    f"  {pattern:10s}: 変化量={increase:+.4f} {direction}, 比率={ratio:.4f}x"
                )

    logger.info("")
    logger.info(f"結果を保存しました: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
