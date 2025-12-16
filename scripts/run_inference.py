"""推論実行スクリプト.

摂動データに対してモデル推論を実行し、影響度を分析する.

使用例:
    # GSM8Kベンチマークでgemma-2-2b（ptモデル）を8-shot CoTで評価
    PYTHONPATH=. uv run python scripts/run_inference.py \
        --model gemma-2-2b --benchmark gsm8k --use-pt-model

    # itモデルを使用（チャットテンプレート適用）
    PYTHONPATH=. uv run python scripts/run_inference.py \
        --model gemma-3-1b-it --benchmark gsm8k

    # 推論結果を保存してエラー分析
    PYTHONPATH=. uv run python scripts/run_inference.py \
        --model gemma-2-9b --benchmark gsm8k --use-pt-model --save-inference-results

    # GPU指定
    PYTHONPATH=. uv run python scripts/run_inference.py \
        --model Meta-Llama-3-8B --benchmark bbh --use-pt-model --gpu-ids 0,1
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation import (
    AnalysisResult,
    BenchmarkEvaluator,
    EvaluationResult,
    analyze_perturbation_impact,
    generate_ranking,
)
from src.evaluation.analyzer import save_analysis_result
from src.models import (
    SUPPORTED_MODELS,
    GenerationConfig,
    InferenceResult,
    MMLUInferenceResult,
    get_supported_models,
    load_model,
    setup_device,
)
from src.models.inference import (
    BENCHMARK_SHOTS,
    create_prompt,
    run_inference_mmlu,
    run_inference_mmlu_perturbed,
)
from src.utils.logger import logger
from src.visualization import generate_and_save_wordcloud


def load_perturbed_data(benchmark_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """摂動データを読み込む.

    Args:
        benchmark_dir: ベンチマークディレクトリ (data/perturbed/{benchmark})

    Returns:
        (メタデータ, サンプルリスト)
    """
    examples_file = benchmark_dir / "examples.json"

    if not examples_file.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {examples_file}")

    with open(examples_file, encoding="utf-8") as f:
        data = json.load(f)

    return data["metadata"], data["examples"]


def run_model_inference(
    model: Any,
    examples: list[dict[str, Any]],
    benchmark_name: str,
    text_field: str,
    batch_size: int,
    generation_config: GenerationConfig,
    is_perturbed: bool = False,
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[InferenceResult]:
    """モデル推論を実行.

    Args:
        model: 推論に使用するモデル
        examples: サンプルデータ
        benchmark_name: ベンチマーク名
        text_field: テキストフィールド名
        batch_size: バッチサイズ
        generation_config: 生成設定
        is_perturbed: 摂動データかどうか
        use_chat_format: チャット形式を使用するか（False=ptモデル用テキストプロンプト）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    # プロンプト生成
    prompts = []
    for example in examples:
        # 摂動データの場合はperturbed_textを使用
        if is_perturbed:
            text = example.get("perturbed_text", example.get(text_field, ""))
        else:
            text = example.get(text_field, "")
        prompt = create_prompt(benchmark_name, text, use_chat_format=use_chat_format)
        prompts.append(prompt)

    # 推論実行
    if hasattr(model, "generate_batch"):
        generated_texts = model.generate_batch(
            prompts,
            config=generation_config,
            batch_size=batch_size,
        )
    else:
        generated_texts = model.generate(prompts, config=generation_config)

    # 結果を整理
    results = []
    for i, (example, generated) in enumerate(zip(examples, generated_texts, strict=True)):
        original_text = example.get(text_field, "")
        perturbed_text = example.get("perturbed_text") if is_perturbed else None

        # プロンプトを文字列として保存（オプション）
        prompt_str = None
        if save_prompts:
            p = prompts[i]
            if isinstance(p, list):
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in p)
            else:
                prompt_str = p

        result = InferenceResult(
            example_id=example.get("id", i),
            original_text=original_text,
            generated_text=generated,
            expected_answer=str(example.get("answer", "")),
            is_perturbed=is_perturbed,
            perturbed_text=perturbed_text,
            prompt=prompt_str,
        )
        results.append(result)

    return results


def save_inference_results(
    results: list[InferenceResult] | list[MMLUInferenceResult],
    output_path: Path,
    metadata: dict[str, Any] | None = None,
    evaluation_result: EvaluationResult | None = None,
) -> None:
    """推論結果を保存（正誤判定付き）.

    Args:
        results: 推論結果のリスト（InferenceResultまたはMMLUInferenceResult）
        output_path: 出力ファイルパス
        metadata: メタデータ
        evaluation_result: 評価結果（per_sample_resultsに正誤情報を含む）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 評価結果をIDでインデックス化
    eval_by_id: dict[int, dict[str, Any]] = {}
    if evaluation_result and evaluation_result.per_sample_results:
        for sample in evaluation_result.per_sample_results:
            eval_by_id[sample["id"]] = sample

    # 推論結果に正誤情報を追加
    results_with_eval = []
    for r in results:
        result_dict = asdict(r)
        # 評価結果から正誤情報を取得
        if r.example_id in eval_by_id:
            eval_info = eval_by_id[r.example_id]
            result_dict["is_correct"] = eval_info.get("is_correct", None)
            result_dict["extracted_answer"] = eval_info.get("predicted", None)
        else:
            result_dict["is_correct"] = None
            result_dict["extracted_answer"] = None
        results_with_eval.append(result_dict)

    data = {
        "metadata": metadata or {},
        "results": results_with_eval,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"推論結果を保存: {output_path}")


def process_benchmark_mmlu(
    model: Any,
    perturbed_dir: Path,
    output_dir: Path,
    top_n: int | None = None,
    font_path: str | None = None,
    normalization_method: str = "per_perturbation",
    use_chat_format: bool = True,
    save_inference: bool = False,
) -> AnalysisResult:
    """MMLUベンチマークの推論・評価・分析を実行（ログ確率方式、lm-eval-harness公式準拠）.

    lm-eval-harness公式のMMLU評価方式:
    - output_type: multiple_choice
    - metric: acc (選択肢のログ確率比較)

    Args:
        model: 推論に使用するモデル
        perturbed_dir: 摂動データディレクトリ
        output_dir: 出力ディレクトリ
        top_n: 処理する単語数の上限
        font_path: ワードクラウド用フォントパス
        normalization_method: 正規化手法
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_inference: 推論結果を保存するか

    Returns:
        分析結果
    """
    benchmark_name = "mmlu"
    evaluator = BenchmarkEvaluator(benchmark_name)

    # ベースラインデータの読み込みと推論
    logger.info("=" * 60)
    logger.info("MMLU ベースライン推論を開始（ログ確率方式）")
    original_dir = perturbed_dir / "original"
    baseline_metadata, baseline_examples = load_perturbed_data(original_dir)
    text_field = baseline_metadata.get("text_field", "question")

    # MMLU専用のログ確率方式で推論
    baseline_results = run_inference_mmlu(
        model=model,
        examples=baseline_examples,
        text_field=text_field,
        use_chat_format=use_chat_format,
        save_prompts=save_inference,
    )

    # ログ確率方式で評価
    baseline_evaluation = evaluator.evaluate_mmlu_with_logprobs(baseline_results, baseline_examples)
    logger.info(f"ベースライン精度: {baseline_evaluation.accuracy:.4f}")
    logger.info(f"  ({baseline_evaluation.correct_samples}/{baseline_evaluation.total_samples})")

    # ベースライン推論結果を保存（オプション）
    if save_inference:
        inference_dir = output_dir / "inference_results"
        actual_num_shots = 0 if use_chat_format else BENCHMARK_SHOTS.get(benchmark_name, 0)
        save_inference_results(
            baseline_results,
            inference_dir / "baseline_inference.json",
            metadata={
                "model_name": model.model_name,
                "benchmark_name": benchmark_name,
                "evaluation_method": "logprobs",  # ログ確率方式を明示
                "use_chat_format": use_chat_format,
                "num_shots": actual_num_shots,
                "accuracy": baseline_evaluation.accuracy,
                "correct_samples": baseline_evaluation.correct_samples,
                "total_samples": baseline_evaluation.total_samples,
                "created_at": datetime.now(UTC).isoformat(),
            },
            evaluation_result=baseline_evaluation,
        )

    # ベースライン推論結果をIDでインデックス化
    baseline_results_by_id = {result.example_id: result for result in baseline_results}

    # 摂動データの推論
    logger.info("=" * 60)
    logger.info("MMLU 摂動データ推論を開始（ログ確率方式）")

    perturbed_words_dir = perturbed_dir / "perturbed"
    if not perturbed_words_dir.exists():
        raise FileNotFoundError(f"摂動データディレクトリが見つかりません: {perturbed_words_dir}")

    word_dirs = sorted([d for d in perturbed_words_dir.iterdir() if d.is_dir()])

    if top_n:
        word_dirs = word_dirs[:top_n]
        logger.info(f"上位 {top_n} 単語のみ処理")

    logger.info(f"処理対象単語数: {len(word_dirs)}")

    # ベースラインサンプルをIDでインデックス化
    baseline_by_id = {ex.get("id", i): ex for i, ex in enumerate(baseline_examples)}

    # 各単語の摂動データを推論・評価
    perturbed_results: dict[str, tuple[EvaluationResult, dict[str, Any]]] = {}

    for i, word_dir in enumerate(word_dirs):
        target_word = word_dir.name
        logger.info(f"[{i + 1}/{len(word_dirs)}] 単語: {target_word}")

        try:
            metadata, perturbed_examples = load_perturbed_data(word_dir)

            # 摂動サンプルに対してログ確率方式で推論
            perturbed_inference_results = run_inference_mmlu_perturbed(
                model=model,
                perturbed_examples=perturbed_examples,
                use_chat_format=use_chat_format,
                save_prompts=save_inference,
            )

            # 摂動推論結果を保存（オプション）
            if save_inference:
                perturbed_evaluation_only = evaluator.evaluate_mmlu_with_logprobs(
                    perturbed_inference_results, perturbed_examples
                )
                perturb_num_shots = 0 if use_chat_format else BENCHMARK_SHOTS.get(benchmark_name, 0)
                save_inference_results(
                    perturbed_inference_results,
                    inference_dir / f"perturbed_{target_word}_inference.json",
                    metadata={
                        "model_name": model.model_name,
                        "benchmark_name": benchmark_name,
                        "target_word": target_word,
                        "evaluation_method": "logprobs",
                        "use_chat_format": use_chat_format,
                        "num_shots": perturb_num_shots,
                        "perturbed_occurrences": metadata.get("perturbed_occurrences", 0),
                        "accuracy": perturbed_evaluation_only.accuracy,
                        "correct_samples": perturbed_evaluation_only.correct_samples,
                        "total_samples": perturbed_evaluation_only.total_samples,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    evaluation_result=perturbed_evaluation_only,
                )

            # 摂動推論結果をIDでインデックス化
            perturbed_results_by_id = {
                result.example_id: result for result in perturbed_inference_results
            }

            # 全サンプルの推論結果をマージ（評価用）
            merged_results: list[MMLUInferenceResult] = []
            merged_examples: list[dict[str, Any]] = []

            for example_id, baseline_ex in baseline_by_id.items():
                if example_id in perturbed_results_by_id:
                    merged_results.append(perturbed_results_by_id[example_id])
                    merged_examples.append(baseline_ex)
                else:
                    merged_results.append(baseline_results_by_id[example_id])
                    merged_examples.append(baseline_ex)

            # 評価（全サンプルに対して、ログ確率方式）
            evaluation = evaluator.evaluate_mmlu_with_logprobs(merged_results, merged_examples)

            perturbed_results[target_word] = (
                evaluation,
                {
                    "target_word_score": metadata.get("target_word_score", 0.0),
                    "total_occurrences": metadata.get("total_occurrences", 0),
                    "perturbed_occurrences": metadata.get("perturbed_occurrences", 0),
                    "num_examples": len(perturbed_examples),
                },
            )

            accuracy_drop = baseline_evaluation.accuracy - evaluation.accuracy
            logger.info(
                f"  精度: {evaluation.accuracy:.4f} (低下: {accuracy_drop:.4f}) "
                f"[摂動サンプル: {len(perturbed_examples)}/{len(merged_examples)}]"
            )

        except Exception as e:
            logger.error(f"  エラー: {e}")
            continue

    # 分析実行
    logger.info("=" * 60)
    logger.info("影響度分析を実行")

    analysis = analyze_perturbation_impact(
        baseline_result=baseline_evaluation,
        perturbed_results=perturbed_results,
        model_name=model.model_name,
        benchmark_name=benchmark_name,
        normalization_method=normalization_method,
    )

    # 結果保存
    logger.info("=" * 60)
    logger.info("結果を保存")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_analysis_result(analysis, output_dir)

    # ワードクラウド生成
    logger.info("ワードクラウドを生成")
    generate_and_save_wordcloud(analysis, output_dir, font_path=font_path)

    return analysis


def process_benchmark(
    model: Any,
    benchmark_name: str,
    perturbed_dir: Path,
    output_dir: Path,
    batch_size: int,
    generation_config: GenerationConfig,
    top_n: int | None = None,
    font_path: str | None = None,
    normalization_method: str = "per_perturbation",
    use_chat_format: bool = True,
    save_inference: bool = False,
) -> AnalysisResult:
    """ベンチマークの推論・評価・分析を実行.

    MMLUの場合はログ確率方式（lm-eval-harness公式準拠）を使用.
    その他のベンチマークはテキスト生成方式を使用.

    Args:
        model: 推論に使用するモデル
        benchmark_name: ベンチマーク名
        perturbed_dir: 摂動データディレクトリ
        output_dir: 出力ディレクトリ
        batch_size: バッチサイズ
        generation_config: 生成設定
        top_n: 処理する単語数の上限
        font_path: ワードクラウド用フォントパス
        normalization_method: 正規化手法
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_inference: 推論結果を保存するか

    Returns:
        分析結果
    """
    # MMLUの場合はログ確率方式（lm-eval-harness公式準拠）を使用
    if benchmark_name == "mmlu":
        return process_benchmark_mmlu(
            model=model,
            perturbed_dir=perturbed_dir,
            output_dir=output_dir,
            top_n=top_n,
            font_path=font_path,
            normalization_method=normalization_method,
            use_chat_format=use_chat_format,
            save_inference=save_inference,
        )

    evaluator = BenchmarkEvaluator(benchmark_name)

    # ベースラインデータの読み込みと推論
    logger.info("=" * 60)
    logger.info("ベースライン推論を開始")
    original_dir = perturbed_dir / "original"
    baseline_metadata, baseline_examples = load_perturbed_data(original_dir)
    text_field = baseline_metadata.get("text_field", "question")

    baseline_results = run_model_inference(
        model=model,
        examples=baseline_examples,
        benchmark_name=benchmark_name,
        text_field=text_field,
        batch_size=batch_size,
        generation_config=generation_config,
        is_perturbed=False,
        use_chat_format=use_chat_format,
        save_prompts=save_inference,
    )

    baseline_evaluation = evaluator.evaluate(baseline_results, baseline_examples)
    logger.info(f"ベースライン精度: {baseline_evaluation.accuracy:.4f}")
    logger.info(f"  ({baseline_evaluation.correct_samples}/{baseline_evaluation.total_samples})")

    # ベースライン推論結果を保存（オプション）
    if save_inference:
        inference_dir = output_dir / "inference_results"
        # ITモデル（チャット形式）: 0-shot、PTモデル（テキスト形式）: BENCHMARK_SHOTSの値
        actual_num_shots = 0 if use_chat_format else BENCHMARK_SHOTS.get(benchmark_name, 0)
        save_inference_results(
            baseline_results,
            inference_dir / "baseline_inference.json",
            metadata={
                "model_name": model.model_name,
                "benchmark_name": benchmark_name,
                "use_chat_format": use_chat_format,
                "num_shots": actual_num_shots,
                "accuracy": baseline_evaluation.accuracy,
                "correct_samples": baseline_evaluation.correct_samples,
                "total_samples": baseline_evaluation.total_samples,
                "created_at": datetime.now(UTC).isoformat(),
            },
            evaluation_result=baseline_evaluation,
        )

    # ベースライン推論結果をIDでインデックス化（摂動評価時に再利用）
    baseline_results_by_id = {result.example_id: result for result in baseline_results}

    # 摂動データの推論
    logger.info("=" * 60)
    logger.info("摂動データ推論を開始")

    perturbed_words_dir = perturbed_dir / "perturbed"
    if not perturbed_words_dir.exists():
        raise FileNotFoundError(f"摂動データディレクトリが見つかりません: {perturbed_words_dir}")

    # 摂動単語ディレクトリを取得
    word_dirs = sorted([d for d in perturbed_words_dir.iterdir() if d.is_dir()])

    if top_n:
        word_dirs = word_dirs[:top_n]
        logger.info(f"上位 {top_n} 単語のみ処理")

    logger.info(f"処理対象単語数: {len(word_dirs)}")

    # ベースラインサンプルをIDでインデックス化
    baseline_by_id = {ex.get("id", i): ex for i, ex in enumerate(baseline_examples)}

    # 各単語の摂動データを推論・評価
    perturbed_results: dict[str, tuple[EvaluationResult, dict[str, Any]]] = {}

    for i, word_dir in enumerate(word_dirs):
        target_word = word_dir.name
        logger.info(f"[{i + 1}/{len(word_dirs)}] 単語: {target_word}")

        try:
            metadata, perturbed_examples = load_perturbed_data(word_dir)

            # 摂動サンプルのみに対して推論を実行（効率化）
            # 摂動がないサンプルはベースライン結果を再利用
            perturbed_inference_results = run_model_inference(
                model=model,
                examples=perturbed_examples,
                benchmark_name=benchmark_name,
                text_field=text_field,
                batch_size=batch_size,
                generation_config=generation_config,
                is_perturbed=True,  # perturbed_textフィールドを使用
                use_chat_format=use_chat_format,
                save_prompts=save_inference,
            )

            # 摂動推論結果を保存（オプション）
            if save_inference:
                # 摂動データのみの評価を実行（正誤情報を取得）
                perturbed_evaluation = evaluator.evaluate(
                    perturbed_inference_results, perturbed_examples
                )
                # actual_num_shotsはベースライン保存時に計算済み
                perturb_num_shots = 0 if use_chat_format else BENCHMARK_SHOTS.get(benchmark_name, 0)
                save_inference_results(
                    perturbed_inference_results,
                    inference_dir / f"perturbed_{target_word}_inference.json",
                    metadata={
                        "model_name": model.model_name,
                        "benchmark_name": benchmark_name,
                        "target_word": target_word,
                        "use_chat_format": use_chat_format,
                        "num_shots": perturb_num_shots,
                        "perturbed_occurrences": metadata.get("perturbed_occurrences", 0),
                        "accuracy": perturbed_evaluation.accuracy,
                        "correct_samples": perturbed_evaluation.correct_samples,
                        "total_samples": perturbed_evaluation.total_samples,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    evaluation_result=perturbed_evaluation,
                )

            # 摂動推論結果をIDでインデックス化
            perturbed_results_by_id = {
                result.example_id: result for result in perturbed_inference_results
            }

            # 全サンプルの推論結果をマージ（評価用）
            # - 摂動があるサンプル: 新しい推論結果を使用
            # - 摂動がないサンプル: ベースライン結果を再利用
            merged_results: list[InferenceResult] = []
            merged_examples: list[dict[str, Any]] = []

            for example_id, baseline_ex in baseline_by_id.items():
                if example_id in perturbed_results_by_id:
                    # 摂動があるサンプル: 新しい推論結果を使用
                    merged_results.append(perturbed_results_by_id[example_id])
                    merged_examples.append(baseline_ex)
                else:
                    # 摂動がないサンプル: ベースライン結果を再利用
                    merged_results.append(baseline_results_by_id[example_id])
                    merged_examples.append(baseline_ex)

            # 評価（全サンプルに対して）
            evaluation = evaluator.evaluate(merged_results, merged_examples)

            # メタデータを含めて保存
            perturbed_results[target_word] = (
                evaluation,
                {
                    "target_word_score": metadata.get("target_word_score", 0.0),
                    "total_occurrences": metadata.get("total_occurrences", 0),
                    "perturbed_occurrences": metadata.get("perturbed_occurrences", 0),
                    "num_examples": len(perturbed_examples),  # 摂動があったサンプル数
                },
            )

            accuracy_drop = baseline_evaluation.accuracy - evaluation.accuracy
            logger.info(
                f"  精度: {evaluation.accuracy:.4f} (低下: {accuracy_drop:.4f}) "
                f"[摂動サンプル: {len(perturbed_examples)}/{len(merged_examples)}]"
            )

        except Exception as e:
            logger.error(f"  エラー: {e}")
            continue

    # 分析実行
    logger.info("=" * 60)
    logger.info("影響度分析を実行")

    analysis = analyze_perturbation_impact(
        baseline_result=baseline_evaluation,
        perturbed_results=perturbed_results,
        model_name=model.model_name,
        benchmark_name=benchmark_name,
        normalization_method=normalization_method,
    )

    # 結果保存
    logger.info("=" * 60)
    logger.info("結果を保存")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_analysis_result(analysis, output_dir)

    # ワードクラウド生成
    logger.info("ワードクラウドを生成")
    generate_and_save_wordcloud(analysis, output_dir, font_path=font_path)

    return analysis


def main() -> None:
    """メイン処理."""
    parser = argparse.ArgumentParser(description="摂動データに対する推論と影響度分析を実行")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"使用するモデル名。サポートモデル: {get_supported_models()}",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="ベンチマーク名 (gsm8k, bbh, mmlu, jamp, jnli, niilc, jsquad, jcommonsenseqa)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="使用するGPU ID (カンマ区切り、デフォルト: 0。vLLM使用時は小型モデルでは1GPU推奨)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="バッチサイズ (デフォルト: 8)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大生成トークン数 (デフォルト: 512、CoT推論に対応)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="処理する単語数の上限 (デフォルト: 全単語)",
    )
    parser.add_argument(
        "--perturbed-dir",
        type=str,
        default="data/perturbed",
        help="摂動データディレクトリ (デフォルト: data/perturbed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiment1",
        help="出力ディレクトリ (デフォルト: results/experiment1)",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="ワードクラウド用フォントパス (日本語の場合は必要)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="per_perturbation",
        choices=[
            "per_perturbation",
            "per_example",
            "per_occurrence",
            "log_perturbation",
            "log_occurrence",
        ],
        help=(
            "正規化手法 (デフォルト: per_perturbation). "
            "per_perturbation: 摂動1回あたり, "
            "per_example: サンプル1件あたり, "
            "per_occurrence: 出現1回あたり, "
            "log_perturbation: 摂動回数の対数, "
            "log_occurrence: 出現回数の対数"
        ),
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="vLLMを使用して高速推論を行う (Flash Attention 2自動使用)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPUメモリ使用率 (vLLM使用時のみ、デフォルト: 0.9)",
    )
    parser.add_argument(
        "--use-pt-model",
        action="store_true",
        help=(
            "ptモデル（pretrained/base）用のテキストプロンプト形式を使用。"
            "チャットテンプレートを適用せず、few-shot examplesをテキストとして直接埋め込む。"
            "GSM8Kは8-shot CoT、BBHは3-shot、MMLUは5-shotがデフォルト。"
        ),
    )
    parser.add_argument(
        "--save-inference-results",
        action="store_true",
        help="推論結果を完全に保存する（エラー分析用）",
    )

    args = parser.parse_args()

    # モデル情報を取得してit/ptを自動判定（--use-pt-modelで上書き可能）
    model_info = SUPPORTED_MODELS.get(args.model)
    if model_info is None:
        raise ValueError(
            f"サポートされていないモデル: {args.model}. "
            f"サポートモデル: {list(SUPPORTED_MODELS.keys())}"
        )

    # use_chat_formatを決定
    # - --use-pt-model が指定されている場合: False（ptモデル用テキストプロンプト）
    # - それ以外: モデルのis_instruct属性に基づく
    if args.use_pt_model:
        use_chat_format = False
    else:
        use_chat_format = model_info.get("is_instruct", True)

    # 実際に使用するfew-shot数を計算
    # ITモデル（チャット形式）: 0-shot、PTモデル（テキスト形式）: BENCHMARK_SHOTSの値
    actual_num_shots = 0 if use_chat_format else BENCHMARK_SHOTS.get(args.benchmark, 0)

    # 設定表示
    logger.info("=" * 60)
    logger.info("推論・分析開始")
    logger.info(f"  モデル: {args.model}")
    logger.info(f"  ベンチマーク: {args.benchmark}")
    logger.info(f"  GPU: {args.gpu_ids}")
    logger.info(f"  バッチサイズ: {args.batch_size}")
    logger.info(f"  最大生成トークン: {args.max_new_tokens}")
    logger.info(f"  正規化手法: {args.normalization}")
    logger.info(f"  vLLM使用: {args.use_vllm}")
    logger.info(f"  ptモデル形式: {args.use_pt_model}")
    logger.info(f"  チャット形式: {use_chat_format}")
    logger.info(f"  few-shot数: {actual_num_shots}")
    logger.info(f"  推論結果保存: {args.save_inference_results}")
    if args.use_vllm:
        logger.info(f"  GPUメモリ使用率: {args.gpu_memory_utilization}")
    logger.info("=" * 60)

    # モデルロード
    logger.info("モデルをロード中...")
    if args.use_vllm:
        # vLLMを使用（Flash Attention 2自動使用）
        model = load_model(
            args.model,
            use_vllm=True,
            gpu_ids=args.gpu_ids,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        # 通常のHuggingFaceモデル
        device = setup_device(args.gpu_ids)
        model = load_model(args.model, device=device)

    # 生成設定
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    # パス設定
    perturbed_dir = Path(args.perturbed_dir) / args.benchmark
    output_dir = Path(args.output_dir) / args.model / args.benchmark

    if not perturbed_dir.exists():
        raise FileNotFoundError(
            f"摂動データディレクトリが見つかりません: {perturbed_dir}\n"
            f"先に run_perturbation.py を実行してください。"
        )

    # 推論・評価・分析実行
    analysis = process_benchmark(
        model=model,
        benchmark_name=args.benchmark,
        perturbed_dir=perturbed_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        generation_config=generation_config,
        top_n=args.top_n,
        font_path=args.font_path,
        normalization_method=args.normalization,
        use_chat_format=use_chat_format,
        save_inference=args.save_inference_results,
    )

    # 結果サマリー
    logger.info("=" * 60)
    logger.info("処理完了")
    logger.info(f"  ベースライン精度: {analysis.baseline_accuracy:.4f}")
    logger.info(f"  分析単語数: {len(analysis.word_impacts)}")

    if analysis.word_impacts:
        logger.info("\n影響度トップ10:")
        ranking = generate_ranking(analysis, top_n=10)
        for item in ranking:
            logger.info(
                f"  {item['rank']:2d}. {item['target_word']:15s} "
                f"精度低下: {item['accuracy_drop']:.4f} "
                f"正規化影響度: {item['normalized_impact']:.6f}"
            )

    logger.info(f"\n結果保存先: {output_dir}")
    if args.save_inference_results:
        logger.info(f"推論結果保存先: {output_dir / 'inference_results'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
