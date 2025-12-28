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
import os
import sys
from pathlib import Path


def setup_gpu(gpu_id: str) -> None:
    """GPU環境を設定（PyTorch/CUDA初期化前に呼び出す必要あり）.

    Args:
        gpu_id: 使用するGPU ID
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def get_gpu_id_from_args() -> str:
    """コマンドライン引数からgpu-idを先行取得.

    Returns:
        GPU ID（デフォルト: "0"）
    """
    # 簡易的な引数解析（torchインポート前に実行するため）
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu-id" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"


# GPU設定を最初に行う（PyTorch/CUDAの初期化前に必須）
setup_gpu(get_gpu_id_from_args())

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# GPU設定後にtorchを使用するモジュールをインポート
from src.experiment2.entropy_analysis.case_study_analyzer import (  # noqa: E402
    run_case_study,
)
from src.utils.logger import logger  # noqa: E402


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
        choices=["gsm8k", "bbh", "mmlu", "truthfulqa"],
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
        type=str,
        default="10",
        help="分析するサンプル数（デフォルト: 10）。'all'を指定すると全件使用",
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
    parser.add_argument(
        "--mapping-file",
        type=str,
        required=True,
        help="摂動マッピングファイルパス（scripts/generate_perturbation_mapping.py で生成）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="推論時のバッチサイズ（デフォルト: 1）。大きくするとGPUメモリを多く使用するが高速化",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="生成時のtop-kサンプリング（デフォルト: 1=greedy）。2以上で上位k個からサンプリング",
    )
    parser.add_argument(
        "--require-all-patterns",
        action="store_true",
        default=True,
        help="全パターン（original+pattern1-3）が揃ったサンプルのみ出力（デフォルト: True）",
    )
    parser.add_argument(
        "--no-require-all-patterns",
        action="store_false",
        dest="require_all_patterns",
        help="パターンが欠けているサンプルも出力に含める",
    )
    parser.add_argument(
        "--num-perturbations",
        type=int,
        default=1,
        help="1文あたりの摂動箇所数（デフォルト: 1）。"
        "全パターン（Pattern 1/2/3）で同じ単語に摂動を適用する",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="MMLUで各トピックからnum-samples件ずつサンプリング（デフォルト: True）",
    )
    parser.add_argument(
        "--no-stratified",
        action="store_false",
        dest="stratified",
        help="MMLUで全体からランダムサンプリング（層別サンプリングを無効化）",
    )

    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    # クイックモードの設定
    if args.quick:
        num_samples: int | None = 5
        max_new_tokens = 30
        logger.info("クイック分析モード: 5サンプル、30トークン")
    else:
        # num_samplesの解析: "all"の場合はNone、それ以外は整数
        if args.num_samples.lower() == "all":
            num_samples = None
        else:
            try:
                num_samples = int(args.num_samples)
            except ValueError as e:
                raise ValueError(
                    f"--num-samples は整数または'all'を指定してください: {args.num_samples}"
                ) from e
        max_new_tokens = args.max_new_tokens

    # 出力パスの設定
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("results/experiment2/case_study")
        output_dir.mkdir(parents=True, exist_ok=True)
        subset_suffix = f"_{args.subset}" if args.subset else ""
        model_safe = args.model.replace("/", "_").replace("-", "_")
        # num_perturbationsが1より大きい場合はファイル名に含める
        np_suffix = f"_np{args.num_perturbations}" if args.num_perturbations > 1 else ""
        output_path = (
            output_dir / f"{args.benchmark}{subset_suffix}_{model_safe}{np_suffix}_analysis.json"
        )

    logger.info("=" * 60)
    logger.info("ケーススタディ分析")
    logger.info("=" * 60)
    mapping_path = Path(args.mapping_file)

    logger.info(f"ベンチマーク: {args.benchmark}")
    logger.info(f"モデル: {args.model}")
    logger.info(f"サンプル数: {num_samples if num_samples is not None else 'all'}")
    logger.info(f"最大生成トークン数: {max_new_tokens}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"Top-kサンプリング: {args.top_k}")
    logger.info(f"GPU ID: {args.gpu_id}")
    logger.info(f"乱数シード: {args.seed}")
    logger.info(f"摂動マッピングファイル: {mapping_path}")
    logger.info(f"全パターン必須: {args.require_all_patterns}")
    logger.info(f"摂動箇所数/文: {args.num_perturbations}")
    logger.info(f"層別サンプリング: {args.stratified}")
    logger.info(f"出力先: {output_path}")
    if args.subset:
        logger.info(f"サブセット: {args.subset}")
    logger.info("=" * 60)

    # ケーススタディを実行
    result = run_case_study(
        benchmark_name=args.benchmark,
        model_name=args.model,
        num_samples=num_samples,
        mapping_file=mapping_path,
        subset=args.subset,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        gpu_id=args.gpu_id,
        seed=args.seed,
        batch_size=args.batch_size,
        top_k_sampling=args.top_k,
        require_all_patterns=args.require_all_patterns,
        num_perturbations=args.num_perturbations,
        stratified=args.stratified,
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
