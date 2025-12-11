"""推論実行モジュール.

各ベンチマークに対するプロンプト生成と推論実行を行う.
ptモデル（pretrained）とitモデル（instruction-tuned）の両方をサポート.
"""

from dataclasses import dataclass
from typing import Any

from src.models.model_loader import BaseModel, GenerationConfig
from src.utils.logger import logger

# ベンチマーク別few-shot設定
# GSM8K: 8-shot (CoT), BBH: 3-shot, MMLU: 5-shot
BENCHMARK_SHOTS: dict[str, int] = {
    "gsm8k": 8,
    "bbh": 3,
    "mmlu": 5,
    # 日本語ベンチマークはデフォルト0-shot
    "jamp": 0,
    "jnli": 0,
    "niilc": 0,
    "jsquad": 0,
    "jcommonsenseqa": 0,
}

# GSM8K用8-shot CoT examples
# lm-eval-harness公式: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
# 形式: Q: {{question}}\n\nA: {{target}}
# fmt: off
GSM8K_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "target": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "target": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "target": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "target": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "target": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "target": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "target": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "target": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]
# fmt: on

# BBH用few-shot examples (3-shot)
# lm-eval-harness公式: 各サブタスクごとにfewshot examplesを定義
# https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/bbh/fewshot
# 形式: Q: {{input}}\nA: Let's think step by step.\n{{target}}
# BBHは各サブタスクで異なるfew-shot examplesを使用するため、実行時に動的に取得
BBH_FEW_SHOT_EXAMPLES: list[dict[str, str]] = []  # サブタスク別に動的設定

# MMLU用few-shot examples (5-shot)
# lm-eval-harness公式: output_type=multiple_choice（ログ確率比較）
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml
# プロンプト形式:
# {{question.strip()}}
# A. {{choices[0]}}
# B. {{choices[1]}}
# C. {{choices[2]}}
# D. {{choices[3]}}
# Answer:
# ログ確率方式ではfew-shotは使用しない（doc_to_choice: ["A", "B", "C", "D"]）
MMLU_FEW_SHOT_EXAMPLES: list[dict[str, str]] = []  # ログ確率方式では不要

# ベンチマーク別few-shot examples
FEW_SHOT_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "gsm8k": GSM8K_FEW_SHOT_EXAMPLES,
    "bbh": BBH_FEW_SHOT_EXAMPLES,
    "mmlu": MMLU_FEW_SHOT_EXAMPLES,
}

# ptモデル用プロンプトテンプレート（チャットテンプレートなし）
# lm-eval-harness公式形式に準拠
PT_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "gsm8k": {
        # 公式: doc_to_text: "Q: {{question}}\n\nA:"
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
        "prefix": "",
        "question_template": "Q: {question}\n\nA:",
        "answer_template": " {target}\n\n",
    },
    "bbh": {
        # 公式: doc_to_target: "{{target}}", target_delimiter: ""
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml
        # CoTプロンプトは公式では含まれていない
        "prefix": "",
        "question_template": "Q: {question}\nA:",
        "answer_template": " {target}\n\n",
    },
    "mmlu": {
        # 公式: output_type: multiple_choice (ログ確率比較)
        # doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\n...\nAnswer:"
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml
        "prefix": "",
        "question_template": "{question}\nAnswer:",
        "answer_template": " {target}\n\n",
    },
}

# itモデル用プロンプトテンプレート（チャットテンプレート用）
# ITモデルでもlm-eval-harnessの公式プロンプト形式を尊重
IT_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "gsm8k": {
        # GSM8K CoT形式: Q/A形式でステップバイステップの推論を促す
        "system": "You are a helpful assistant that solves math problems step by step.",
        "user": "Q: {question}\n\nA:",
    },
    "bbh": {
        # BBH: 公式実装ではCoTプロンプトは含まない
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml
        "system": "You are a helpful assistant that thinks step by step.",
        "user": "Q: {question}\nA:",
    },
    "mmlu": {
        # MMLU: 選択肢A/B/C/Dから選択（ログ確率方式推奨）
        "system": "Answer the following multiple choice question.",
        "user": "{question}\nAnswer:",
    },
    # 日本語ベンチマーク
    "jamp": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "jnli": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "niilc": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "jsquad": {
        "system": "以下の質問に答えてください。",
        "user": "{question}\n\n回答:",
    },
    "jcommonsenseqa": {
        "system": "以下の質問に答えてください。選択肢から最も適切なものを選んでください。",
        "user": "{question}\n\n回答:",
    },
}

# 後方互換性のためのエイリアス
PROMPT_TEMPLATES = IT_PROMPT_TEMPLATES


@dataclass
class InferenceResult:
    """推論結果を保持するデータクラス."""

    example_id: int
    original_text: str
    generated_text: str
    expected_answer: str
    is_perturbed: bool = False
    perturbed_text: str | None = None
    prompt: str | None = None  # 使用したプロンプト（デバッグ用）


def create_pt_prompt(
    benchmark_name: str,
    question: str,
) -> str:
    """ptモデル用のプロンプトを生成（チャットテンプレートなし）.

    few-shot examplesを含むテキストプロンプトを返す.

    Args:
        benchmark_name: ベンチマーク名
        question: 質問文

    Returns:
        テキストプロンプト
    """
    template = PT_PROMPT_TEMPLATES.get(benchmark_name)
    if template is None:
        # デフォルトテンプレート
        template = {
            "prefix": "",
            "question_template": "Q: {question}\nA:",
            "answer_template": " {target}\n\n",
        }

    # few-shot examplesを追加
    few_shot_examples = FEW_SHOT_EXAMPLES.get(benchmark_name, [])
    num_shots = BENCHMARK_SHOTS.get(benchmark_name, 0)

    prompt_parts = []

    # プレフィックス（あれば）
    if template["prefix"]:
        prompt_parts.append(template["prefix"])

    # few-shot examples
    for example in few_shot_examples[:num_shots]:
        example_question = example["question"]
        example_target = example["target"]

        q_part = template["question_template"].format(question=example_question)
        a_part = template["answer_template"].format(target=example_target)
        prompt_parts.append(q_part + a_part)

    # 実際の質問
    final_question = template["question_template"].format(question=question)
    prompt_parts.append(final_question)

    return "".join(prompt_parts)


def create_it_messages(
    benchmark_name: str,
    question: str,
) -> list[dict[str, str]]:
    """itモデル用のメッセージリストを生成（0-shot）.

    チャットテンプレート用の構造化されたメッセージを返す.
    ITモデルは0-shotで推論するため、few-shot examplesは含めない.

    Args:
        benchmark_name: ベンチマーク名
        question: 質問文

    Returns:
        メッセージリスト [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    template = IT_PROMPT_TEMPLATES.get(benchmark_name)
    if template is None:
        # デフォルトテンプレート
        template = {
            "system": "Answer the following question.",
            "user": "{question}\n\nAnswer:",
        }

    # ITモデルは0-shot（few-shot examplesなし）
    messages: list[dict[str, str]] = [
        {"role": "system", "content": template["system"]},
    ]

    # 質問を追加
    user_content = template["user"].format(question=question)
    messages.append({"role": "user", "content": user_content})

    return messages


def create_prompt(
    benchmark_name: str,
    question: str,
    use_chat_format: bool = True,
) -> str | list[dict[str, str]]:
    """ベンチマークに応じたプロンプトを生成.

    Args:
        benchmark_name: ベンチマーク名
        question: 質問文
        use_chat_format: チャット形式（メッセージリスト）を使用するか
            True: itモデル用メッセージリスト
            False: ptモデル用テキストプロンプト

    Returns:
        use_chat_format=True: メッセージリスト（チャットテンプレート用）
        use_chat_format=False: 文字列プロンプト（ptモデル用）
    """
    if use_chat_format:
        return create_it_messages(benchmark_name, question)
    else:
        return create_pt_prompt(benchmark_name, question)


# 後方互換性のためのエイリアス
def create_messages(
    benchmark_name: str,
    question: str,
) -> list[dict[str, str]]:
    """後方互換性のためのエイリアス."""
    return create_it_messages(benchmark_name, question)


def run_inference(
    model: BaseModel,
    examples: list[dict[str, Any]],
    benchmark_name: str,
    text_field: str = "question",
    batch_size: int = 8,
    generation_config: GenerationConfig | None = None,
    use_batch: bool = True,
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[InferenceResult]:
    """ベンチマークデータに対して推論を実行.

    Args:
        model: 推論に使用するモデル
        examples: サンプルデータのリスト
        benchmark_name: ベンチマーク名
        text_field: テキストフィールド名
        batch_size: バッチサイズ
        generation_config: 生成設定
        use_batch: バッチ処理を使用するか
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=512,  # CoT推論に十分な長さ
            temperature=0.0,
            do_sample=False,
        )

    # プロンプト生成
    prompts = []
    for example in examples:
        question = example.get(text_field, "")
        prompt = create_prompt(benchmark_name, question, use_chat_format=use_chat_format)
        prompts.append(prompt)

    logger.info(f"推論開始: {len(prompts)}件のサンプル")
    logger.info(
        f"チャット形式: {use_chat_format}, few-shot: {BENCHMARK_SHOTS.get(benchmark_name, 0)}-shot"
    )

    # 推論実行
    if use_batch and hasattr(model, "generate_batch"):
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
        prompt_str = None
        if save_prompts:
            # プロンプトを文字列として保存
            p = prompts[i]
            if isinstance(p, list):
                # メッセージリストの場合は整形
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in p)
            else:
                prompt_str = p

        result = InferenceResult(
            example_id=example.get("id", i),
            original_text=example.get(text_field, ""),
            generated_text=generated,
            expected_answer=str(example.get("answer", "")),
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"推論完了: {len(results)}件の結果")
    return results


@dataclass
class MMLUInferenceResult:
    """MMLU推論結果（ログ確率ベース）を保持するデータクラス."""

    example_id: int
    original_text: str
    predicted_answer: str  # ログ確率最大の選択肢
    expected_answer: str
    choice_logprobs: list[float]  # 各選択肢のログ確率
    choices: list[str]  # 選択肢リスト（A, B, C, D）
    is_perturbed: bool = False
    perturbed_text: str | None = None
    prompt: str | None = None


def run_inference_mmlu(
    model: BaseModel,
    examples: list[dict[str, Any]],
    text_field: str = "question",
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[MMLUInferenceResult]:
    """MMLUベンチマークに対してログ確率ベースで推論を実行.

    lm-eval-harness方式: output_type=multiple_choice（選択肢のログ確率比較）

    Args:
        model: 推論に使用するモデル
        examples: サンプルデータのリスト
        text_field: テキストフィールド名
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    # モデルがログ確率計算をサポートしているか確認
    if not hasattr(model, "compute_choice_logprobs"):
        raise ValueError(
            f"モデル {model.model_name} はログ確率計算をサポートしていません。"
            "LocalModelまたはVLLMModelを使用してください。"
        )

    logger.info(f"MMLU推論開始（ログ確率方式）: {len(examples)}件のサンプル")
    logger.info(f"チャット形式: {use_chat_format}")

    # 選択肢リスト
    choices = ["A", "B", "C", "D"]

    results = []
    for i, example in enumerate(examples):
        question = example.get(text_field, "")

        # プロンプト生成
        prompt = create_prompt("mmlu", question, use_chat_format=use_chat_format)

        # ログ確率計算
        logprobs = model.compute_choice_logprobs(prompt, choices)

        # 最大ログ確率の選択肢を予測
        max_idx = logprobs.index(max(logprobs))
        predicted = choices[max_idx]

        prompt_str = None
        if save_prompts:
            if isinstance(prompt, list):
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in prompt)
            else:
                prompt_str = prompt

        result = MMLUInferenceResult(
            example_id=example.get("id", i),
            original_text=question,
            predicted_answer=predicted,
            expected_answer=str(example.get("answer", "")),
            choice_logprobs=logprobs,
            choices=choices,
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"MMLU推論完了: {len(results)}件の結果")
    return results


def run_inference_mmlu_perturbed(
    model: BaseModel,
    perturbed_examples: list[dict[str, Any]],
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[MMLUInferenceResult]:
    """摂動済みMMLUデータに対してログ確率ベースで推論を実行.

    Args:
        model: 推論に使用するモデル
        perturbed_examples: 摂動サンプルデータのリスト
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    # モデルがログ確率計算をサポートしているか確認
    if not hasattr(model, "compute_choice_logprobs"):
        raise ValueError(
            f"モデル {model.model_name} はログ確率計算をサポートしていません。"
            "LocalModelまたはVLLMModelを使用してください。"
        )

    logger.info(f"MMLU摂動データ推論開始（ログ確率方式）: {len(perturbed_examples)}件")

    # 選択肢リスト
    choices = ["A", "B", "C", "D"]

    results = []
    for i, example in enumerate(perturbed_examples):
        perturbed_text = example.get("perturbed_text", "")

        # プロンプト生成
        prompt = create_prompt("mmlu", perturbed_text, use_chat_format=use_chat_format)

        # ログ確率計算
        logprobs = model.compute_choice_logprobs(prompt, choices)

        # 最大ログ確率の選択肢を予測
        max_idx = logprobs.index(max(logprobs))
        predicted = choices[max_idx]

        prompt_str = None
        if save_prompts:
            if isinstance(prompt, list):
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in prompt)
            else:
                prompt_str = prompt

        result = MMLUInferenceResult(
            example_id=example.get("id", i),
            original_text=example.get("original_text", ""),
            predicted_answer=predicted,
            expected_answer=str(example.get("answer", "")),
            choice_logprobs=logprobs,
            choices=choices,
            is_perturbed=True,
            perturbed_text=perturbed_text,
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"MMLU摂動データ推論完了: {len(results)}件の結果")
    return results


def run_inference_on_perturbed(
    model: BaseModel,
    perturbed_examples: list[dict[str, Any]],
    benchmark_name: str,
    batch_size: int = 8,
    generation_config: GenerationConfig | None = None,
    use_batch: bool = True,
    use_chat_format: bool = True,
    save_prompts: bool = False,
) -> list[InferenceResult]:
    """摂動データに対して推論を実行.

    Args:
        model: 推論に使用するモデル
        perturbed_examples: 摂動サンプルデータのリスト
        benchmark_name: ベンチマーク名
        batch_size: バッチサイズ
        generation_config: 生成設定
        use_batch: バッチ処理を使用するか
        use_chat_format: チャット形式を使用するか（False=ptモデル用）
        save_prompts: プロンプトを結果に保存するか

    Returns:
        推論結果のリスト
    """
    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=512,  # CoT推論に十分な長さ
            temperature=0.0,
            do_sample=False,
        )

    # 摂動テキストからプロンプト生成
    prompts = []
    for example in perturbed_examples:
        perturbed_text = example.get("perturbed_text", "")
        prompt = create_prompt(benchmark_name, perturbed_text, use_chat_format=use_chat_format)
        prompts.append(prompt)

    logger.info(f"摂動データ推論開始: {len(prompts)}件のサンプル")

    # 推論実行
    if use_batch and hasattr(model, "generate_batch"):
        generated_texts = model.generate_batch(
            prompts,
            config=generation_config,
            batch_size=batch_size,
        )
    else:
        generated_texts = model.generate(prompts, config=generation_config)

    # 結果を整理
    results = []
    for i, (example, generated) in enumerate(zip(perturbed_examples, generated_texts, strict=True)):
        prompt_str = None
        if save_prompts:
            p = prompts[i]
            if isinstance(p, list):
                prompt_str = "\n".join(f"[{m['role']}]: {m['content']}" for m in p)
            else:
                prompt_str = p

        result = InferenceResult(
            example_id=example.get("id", i),
            original_text=example.get("original_text", ""),
            generated_text=generated,
            expected_answer=str(example.get("answer", "")),
            is_perturbed=True,
            perturbed_text=example.get("perturbed_text", ""),
            prompt=prompt_str,
        )
        results.append(result)

    logger.info(f"摂動データ推論完了: {len(results)}件の結果")
    return results
