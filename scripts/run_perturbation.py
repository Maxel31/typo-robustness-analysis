"""摂動データ生成スクリプト.

ベンチマークデータに対して、頻出単語ごとに摂動を適用したデータを生成する.

使用例:
    # 英語ベンチマーク (GSM8K, BBH, MMLU) で摂動生成
    PYTHONPATH=. uv run python scripts/run_perturbation.py --language english --top-n 10

    # 日本語ベンチマーク (llm-jp-eval) で摂動生成
    PYTHONPATH=. uv run python scripts/run_perturbation.py --language japanese --top-n 10

    # 特定のベンチマークのみ
    PYTHONPATH=. uv run python scripts/run_perturbation.py --language english --benchmarks gsm8k

    # サンプルデータでテスト
    PYTHONPATH=. uv run python scripts/run_perturbation.py \\
        --language english --top-n 3 --use-sample-data
"""

import argparse
import json
from pathlib import Path

import pykakasi

from src.benchmarks import get_available_benchmarks, load_benchmark
from src.perturbation.benchmark_perturbator import (
    OriginalBenchmarkData,
    generate_perturbed_data,
)
from src.utils.logger import logger


def _convert_kanji_to_hiragana(text: str) -> str | None:
    """漢字をひらがなに変換する.

    Args:
        text: 変換対象のテキスト

    Returns:
        ひらがな変換後のテキスト（変換がない場合はNone）
    """
    kks = pykakasi.kakasi()
    result = kks.convert(text)
    hiragana = "".join([item["hira"] for item in result])

    # 元のテキストと同じ場合は変換不要
    if hiragana == text:
        return None

    return hiragana


def load_frequent_words(language: str, top_n: int) -> tuple[list[str], dict[str, float]]:
    """頻出単語リストを読み込む.

    Args:
        language: 言語 ("english" or "japanese")
        top_n: 上位N件

    Returns:
        (単語リスト, 単語→スコアの辞書)
    """
    freq_file = Path(f"data/processed/{language}/frequent_words_top2000.json")

    if not freq_file.exists():
        raise FileNotFoundError(
            f"頻出単語ファイルが見つかりません: {freq_file}\n"
            "先に run_preprocessing.py を実行してください。"
        )

    with open(freq_file, encoding="utf-8") as f:
        data = json.load(f)

    words: list[str] = []
    word_scores: dict[str, float] = {}

    if language == "japanese":
        # 日本語の場合はlemmaフィールドを使用し、漢字はひらがな変換も追加
        for item in data["words"][:top_n]:
            lemma = item["lemma"]
            score = item["score"]
            words.append(lemma)
            word_scores[lemma] = score

            # 漢字が含まれる場合はひらがなバージョンも追加
            # ひらがな変換語は元の語のスコアを継承
            hiragana = _convert_kanji_to_hiragana(lemma)
            if hiragana and hiragana not in words:
                words.append(hiragana)
                word_scores[hiragana] = score  # 元の語のスコアを使用
                logger.debug(f"  ひらがな変換追加: {lemma} → {hiragana} (score={score})")

        logger.info(
            f"頻出単語を読み込み: {top_n}件 → {len(words)}件 (ひらがな変換含む) ({language})"
        )
    else:
        # 英語の場合はwordフィールドを使用
        for item in data["words"][:top_n]:
            word = item["word"]
            score = item["score"]
            words.append(word)
            word_scores[word] = score
        logger.info(f"頻出単語を読み込み: {len(words)}件 ({language})")

    return words, word_scores


def create_sample_benchmark_data(language: str) -> OriginalBenchmarkData:
    """サンプルベンチマークデータを作成（テスト用）.

    実際の実装では、lm-evaluation-harnessやllm-jp-evalからデータをロードする.

    Args:
        language: 言語

    Returns:
        サンプルベンチマークデータ
    """
    if language == "english":
        # GSM8K風のサンプルデータ
        examples = [
            {
                "id": 0,
                "question": "You have 5 apples. You give 2 apples to your friend. "
                "How many apples do you have left?",
                "answer": "3",
            },
            {
                "id": 1,
                "question": "The store has 10 oranges. If you buy 3 oranges, "
                "how many oranges are left in the store?",
                "answer": "7",
            },
            {
                "id": 2,
                "question": "Sarah has 8 books. She gives 2 books to Tom and 3 books to Mary. "
                "How many books does Sarah have now?",
                "answer": "3",
            },
            {
                "id": 3,
                "question": "A farmer has 15 chickens. 5 chickens run away. "
                "How many chickens does the farmer have?",
                "answer": "10",
            },
            {
                "id": 4,
                "question": "You need to buy 4 notebooks. Each notebook costs 2 dollars. "
                "How much money do you need?",
                "answer": "8",
            },
        ]
        return OriginalBenchmarkData(
            benchmark_name="gsm8k_sample",
            language="english",
            text_field="question",
            examples=examples,
        )
    else:
        # 日本語サンプルデータ
        examples = [
            {
                "id": 0,
                "question": "りんごが5個あります。友達に2個あげました。残りは何個ですか？",
                "answer": "3",
            },
            {
                "id": 1,
                "question": "お店にみかんが10個あります。3個買うと、お店には何個残りますか？",
                "answer": "7",
            },
            {
                "id": 2,
                "question": "太郎さんは本を8冊持っています。"
                "花子さんに2冊、次郎さんに3冊あげました。太郎さんの本は何冊ですか？",
                "answer": "3",
            },
            {
                "id": 3,
                "question": "農家には鶏が15羽います。5羽逃げました。農家には何羽の鶏がいますか？",
                "answer": "10",
            },
            {
                "id": 4,
                "question": "ノートを4冊買います。1冊200円です。全部でいくらですか？",
                "answer": "800",
            },
        ]
        return OriginalBenchmarkData(
            benchmark_name="jcommonsenseqa_sample",
            language="japanese",
            text_field="question",
            examples=examples,
        )


def process_benchmark(
    benchmark_data: OriginalBenchmarkData,
    frequent_words: list[str],
    word_scores: dict[str, float],
    output_dir: Path,
    replace_prob: float,
    insert_prob: float,
    delete_prob: float,
    seed: int,
) -> dict[str, Path]:
    """単一ベンチマークに対して摂動処理を実行.

    Args:
        benchmark_data: ベンチマークデータ
        frequent_words: 頻出単語リスト
        word_scores: 単語→スコアの辞書
        output_dir: 出力ディレクトリ
        replace_prob: 置換確率
        insert_prob: 挿入確率
        delete_prob: 削除確率
        seed: 乱数シード

    Returns:
        摂動データの保存パス
    """
    logger.info(f"ベンチマーク '{benchmark_data.benchmark_name}' の摂動処理を開始")
    logger.info(f"  サンプル数: {len(benchmark_data.examples)}")

    saved_paths = generate_perturbed_data(
        benchmark_data=benchmark_data,
        frequent_words=frequent_words,
        word_scores=word_scores,
        output_dir=output_dir,
        replace_prob=replace_prob,
        insert_prob=insert_prob,
        delete_prob=delete_prob,
        base_seed=seed,
        case_sensitive=False,
    )

    logger.info(f"  摂動データ生成完了: {len(saved_paths)}単語分")
    return saved_paths


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="ベンチマークデータに摂動を適用してデータを生成")
    parser.add_argument(
        "--language",
        type=str,
        choices=["english", "japanese"],
        required=True,
        help="言語 (english or japanese)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="処理するベンチマーク名 (指定しない場合は言語に対応する全ベンチマーク)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="使用する頻出単語の数 (デフォルト: 10)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="ベンチマークごとの最大サンプル数 (デフォルト: 全件)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/perturbed",
        help="出力ディレクトリ (デフォルト: data/perturbed)",
    )
    parser.add_argument(
        "--replace-prob",
        type=float,
        default=0.2,
        help="各文字に対する置換確率 (デフォルト: 0.2)",
    )
    parser.add_argument(
        "--insert-prob",
        type=float,
        default=0.2,
        help="各文字に対する挿入確率 (デフォルト: 0.2)",
    )
    parser.add_argument(
        "--delete-prob",
        type=float,
        default=0.2,
        help="各文字に対する削除確率 (デフォルト: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)",
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="サンプルデータを使用（テスト用）",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("摂動データ生成開始")
    logger.info(f"  言語: {args.language}")
    logger.info(f"  頻出単語数: {args.top_n}")
    logger.info(
        f"  摂動確率: replace={args.replace_prob}, "
        f"insert={args.insert_prob}, delete={args.delete_prob}"
    )
    logger.info("=" * 60)

    # 頻出単語を読み込み
    frequent_words, word_scores = load_frequent_words(args.language, args.top_n)
    logger.info(f"対象単語: {frequent_words[:10]}...")

    # 処理するベンチマークを決定
    if args.use_sample_data:
        # サンプルデータモード
        benchmark_data = create_sample_benchmark_data(args.language)
        benchmarks_to_process = [(benchmark_data.benchmark_name, benchmark_data)]
        logger.info("サンプルデータを使用")
    else:
        # 実際のベンチマークをロード
        if args.benchmarks:
            benchmark_names = args.benchmarks
        else:
            benchmark_names = get_available_benchmarks(args.language)

        logger.info(f"処理対象ベンチマーク: {benchmark_names}")

        benchmarks_to_process = []
        for name in benchmark_names:
            try:
                data = load_benchmark(name, max_samples=args.max_samples)
                benchmarks_to_process.append((name, data))
            except Exception as e:
                logger.error(f"ベンチマーク '{name}' のロードに失敗: {e}")
                continue

    # 各ベンチマークに対して摂動処理
    output_dir = Path(args.output_dir)
    all_saved_paths: dict[str, dict[str, Path]] = {}

    for bench_name, bench_data in benchmarks_to_process:
        saved_paths = process_benchmark(
            benchmark_data=bench_data,
            frequent_words=frequent_words,
            word_scores=word_scores,
            output_dir=output_dir,
            replace_prob=args.replace_prob,
            insert_prob=args.insert_prob,
            delete_prob=args.delete_prob,
            seed=args.seed,
        )
        all_saved_paths[bench_name] = saved_paths

    # 結果サマリーを表示
    logger.info("=" * 60)
    logger.info("摂動データ生成完了")
    logger.info(f"出力ディレクトリ: {output_dir}")

    for bench_name, paths in all_saved_paths.items():
        logger.info(f"\n[{bench_name}]")
        logger.info(f"  元データ: {output_dir / bench_name / 'original' / 'examples.json'}")
        logger.info(f"  摂動データ: {len(paths)}単語分")
        for word, path in list(paths.items())[:5]:
            logger.info(f"    - {word}: {path}")
        if len(paths) > 5:
            logger.info(f"    ... 他 {len(paths) - 5}単語")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
