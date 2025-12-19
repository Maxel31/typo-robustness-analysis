"""モデルロード・推論モジュール.

HuggingFaceモデルおよびOpenAI APIモデルの統一的なインターフェースを提供する.
"""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.utils.logger import logger

# transformersの警告レベルを設定（generation flags警告を抑制）
transformers.logging.set_verbosity_error()
# UserWarningも抑制
warnings.filterwarnings("ignore", message=".*generation flags.*")

# サポートするモデルの定義
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    # ============================================================
    # 英語モデル (pretrained/base) - few-shot推論用
    # ============================================================
    "gemma-3-1b-pt": {
        "hf_name": "google/gemma-3-1b-pt",
        "language": "english",
        "type": "local",
        "is_instruct": False,
    },
    "gemma-3-4b-pt": {
        "hf_name": "google/gemma-3-4b-pt",
        "language": "english",
        "type": "local",
        "is_instruct": False,
    },
    "Mistral-7B-v0.3": {
        "hf_name": "mistralai/Mistral-7B-v0.3",
        "language": "english",
        "type": "local",
        "is_instruct": False,
    },
    "Meta-Llama-3.2-3B": {
        "hf_name": "meta-llama/Llama-3.2-3B",
        "language": "english",
        "type": "local",
        "is_instruct": False,
    },
    # ============================================================
    # 英語モデル (instruction-tuned) - 0-shot推論用
    # ============================================================
    "gemma-3-1b-it": {
        "hf_name": "google/gemma-3-1b-it",
        "language": "english",
        "type": "local",
        "is_instruct": True,
    },
    "gemma-3-4b-it": {
        "hf_name": "google/gemma-3-4b-it",
        "language": "english",
        "type": "local",
        "is_instruct": True,
    },
    "Mistral-7B-Instruct-v0.3": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "language": "english",
        "type": "local",
        "is_instruct": True,
    },
    "Meta-Llama-3.2-3B-Instruct": {
        "hf_name": "meta-llama/Llama-3.2-3B-Instruct",
        "language": "english",
        "type": "local",
        "is_instruct": True,
    },
    # ============================================================
    # APIモデル
    # ============================================================
    "gpt-4-0613": {
        "api_name": "gpt-4-0613",
        "language": "english",
        "type": "api",
        "is_instruct": True,
    },
    # ============================================================
    # 日本語モデル (instruction-tuned)
    # ============================================================
    "Llama-3.1-Swallow-8B-Instruct-v0.5": {
        "hf_name": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
        "language": "japanese",
        "type": "local",
        "is_instruct": True,
    },
    "llm-jp-3-3.7b-instruct": {
        "hf_name": "llm-jp/llm-jp-3-3.7b-instruct",
        "language": "japanese",
        "type": "local",
        "is_instruct": True,
    },
    "llm-jp-3-13b-instruct": {
        "hf_name": "llm-jp/llm-jp-3-13b-instruct",
        "language": "japanese",
        "type": "local",
        "is_instruct": True,
    },
}


def setup_device(gpu_ids: str = "0") -> torch.device:
    """GPUの利用可能性を確認し、適切なデバイスを返す.

    Args:
        gpu_ids: 使用するGPU ID（カンマ区切りで複数指定可能, 例: "0,1"）

    Returns:
        torch.device: 使用するデバイス
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        logger.info(f"GPUを使用: {gpu_count}台 - {gpu_names} (GPU IDs: {gpu_ids})")
    else:
        device = torch.device("cpu")
        logger.warning("GPUが利用できません。CPUを使用します。")

    return device


@dataclass
class GenerationConfig:
    """テキスト生成の設定."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    do_sample: bool = False
    num_return_sequences: int = 1


class BaseModel(ABC):
    """モデルの基底クラス."""

    def __init__(self, model_name: str) -> None:
        """初期化.

        Args:
            model_name: モデル名
        """
        self.model_name = model_name
        self.model_info = SUPPORTED_MODELS.get(model_name)
        if self.model_info is None:
            raise ValueError(
                f"サポートされていないモデル: {model_name}. "
                f"サポートモデル: {list(SUPPORTED_MODELS.keys())}"
            )

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """テキストを生成.

        Args:
            prompts: 入力プロンプトのリスト
            config: 生成設定

        Returns:
            生成されたテキストのリスト
        """
        pass

    @property
    def language(self) -> str:
        """モデルの対象言語を返す."""
        return self.model_info["language"]


class LocalModel(BaseModel):
    """HuggingFaceからロードするローカルモデル."""

    # トークナイザーの最大長（モデルに依存しないデフォルト値）
    DEFAULT_MAX_LENGTH = 4096

    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = False,
    ) -> None:
        """初期化.

        Args:
            model_name: モデル名
            device: 使用するデバイス
            dtype: モデルのデータ型
            use_flash_attention: Flash Attention 2を使用するか
        """
        super().__init__(model_name)

        if self.model_info["type"] != "local":
            raise ValueError(f"モデル {model_name} はローカルモデルではありません")

        self.device = device if device is not None else torch.device("cuda")
        self.dtype = dtype
        self.hf_name = self.model_info["hf_name"]

        logger.info(f"モデルをロード中: {self.hf_name}")

        # トークナイザーをロード
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.hf_name,
            trust_remote_code=True,
        )

        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # max_lengthの設定（モデルに設定がない場合はデフォルト値を使用）
        if self.tokenizer.model_max_length is None or self.tokenizer.model_max_length > 1e9:
            self.tokenizer.model_max_length = self.DEFAULT_MAX_LENGTH

        # モデルをロード
        attn_implementation = "flash_attention_2" if use_flash_attention else "eager"

        try:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                self.hf_name,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
        except Exception as e:
            logger.warning(f"Flash Attention 2でのロードに失敗: {e}")
            logger.info("通常のattentionでリトライします")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_name,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model.eval()
        logger.info(f"モデルロード完了: {self.hf_name}")

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """メッセージリストにチャットテンプレートを適用.

        Args:
            messages: メッセージリスト [{"role": "...", "content": "..."}]

        Returns:
            チャットテンプレートが適用されたプロンプト文字列
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # フォールバック: 単純な結合
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant:"
            return prompt

    def _prepare_prompt(
        self,
        prompt: str | list[dict[str, str]],
    ) -> str:
        """プロンプトを準備（メッセージリストの場合はチャットテンプレートを適用）.

        Args:
            prompt: 文字列またはメッセージリスト

        Returns:
            チャットテンプレートが適用されたプロンプト文字列
        """
        if isinstance(prompt, list):
            return self._apply_chat_template(prompt)
        return prompt

    def generate(
        self,
        prompts: list[str | list[dict[str, str]]],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """テキストを生成.

        Args:
            prompts: 入力プロンプトのリスト（文字列またはメッセージリスト）
            config: 生成設定

        Returns:
            生成されたテキストのリスト
        """
        if config is None:
            config = GenerationConfig()

        results = []

        with torch.no_grad():
            for prompt in prompts:
                # メッセージリストの場合はチャットテンプレートを適用
                formatted_prompt = self._prepare_prompt(prompt)

                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)

                # do_sample=Falseの場合、temperature/top_p/top_kは無視される
                generate_kwargs = {
                    "max_new_tokens": config.max_new_tokens,
                    "do_sample": config.do_sample,
                    "num_return_sequences": config.num_return_sequences,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if config.do_sample:
                    generate_kwargs["temperature"] = config.temperature
                    generate_kwargs["top_p"] = config.top_p
                    generate_kwargs["top_k"] = config.top_k

                outputs = self.model.generate(**inputs, **generate_kwargs)

                # 入力部分を除いた生成テキストを取得
                generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                results.append(generated_text)

        return results

    def compute_choice_logprobs(
        self,
        prompt: str | list[dict[str, str]],
        choices: list[str],
    ) -> list[float]:
        """各選択肢のログ確率を計算（lm-eval-harness multiple_choice方式）.

        Args:
            prompt: プロンプト（文字列またはメッセージリスト）
            choices: 選択肢のリスト（例: ["A", "B", "C", "D"]）

        Returns:
            各選択肢のログ確率のリスト
        """
        # メッセージリストの場合はチャットテンプレートを適用
        formatted_prompt = self._prepare_prompt(prompt)

        logprobs_list = []

        with torch.no_grad():
            for choice in choices:
                # プロンプト + 選択肢を連結
                full_text = formatted_prompt + choice

                # トークナイズ
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                ).to(self.model.device)

                # プロンプト部分のトークン数を取得
                prompt_inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                ).to(self.model.device)
                prompt_length = prompt_inputs["input_ids"].shape[1]

                # モデルの出力を取得
                outputs = self.model(**inputs)
                logits = outputs.logits

                # 選択肢部分のログ確率を計算
                # logits[i]はi+1番目のトークンの予測なので、ずらして計算
                choice_logprobs = 0.0
                for i in range(prompt_length - 1, inputs["input_ids"].shape[1] - 1):
                    # 次のトークンのログ確率を取得
                    next_token_id = inputs["input_ids"][0, i + 1]
                    log_probs = torch.log_softmax(logits[0, i], dim=-1)
                    choice_logprobs += log_probs[next_token_id].item()

                logprobs_list.append(choice_logprobs)

        return logprobs_list

    def compute_choice_logprobs_batch(
        self,
        prompts: list[str | list[dict[str, str]]],
        choices_list: list[list[str]],
    ) -> list[list[float]]:
        """バッチでログ確率を計算.

        Args:
            prompts: プロンプトのリスト
            choices_list: 各プロンプトに対する選択肢リストのリスト

        Returns:
            各プロンプトに対する選択肢のログ確率リストのリスト
        """
        results = []
        for prompt, choices in zip(prompts, choices_list, strict=True):
            logprobs = self.compute_choice_logprobs(prompt, choices)
            results.append(logprobs)
        return results

    def generate_batch(
        self,
        prompts: list[str | list[dict[str, str]]],
        config: GenerationConfig | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """バッチ処理でテキストを生成.

        Args:
            prompts: 入力プロンプトのリスト（文字列またはメッセージリスト）
            config: 生成設定
            batch_size: バッチサイズ

        Returns:
            生成されたテキストのリスト
        """
        if config is None:
            config = GenerationConfig()

        results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(
                range(0, len(prompts), batch_size),
                total=total_batches,
                desc="推論中",
                unit="batch",
            ):
                batch_prompts = prompts[i : i + batch_size]

                # メッセージリストの場合はチャットテンプレートを適用
                formatted_prompts = [self._prepare_prompt(p) for p in batch_prompts]

                inputs = self.tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)

                # do_sample=Falseの場合、temperature/top_p/top_kは無視される
                generate_kwargs = {
                    "max_new_tokens": config.max_new_tokens,
                    "do_sample": config.do_sample,
                    "num_return_sequences": config.num_return_sequences,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if config.do_sample:
                    generate_kwargs["temperature"] = config.temperature
                    generate_kwargs["top_p"] = config.top_p
                    generate_kwargs["top_k"] = config.top_k

                outputs = self.model.generate(**inputs, **generate_kwargs)

                for j, output in enumerate(outputs):
                    input_length = inputs["input_ids"][j].shape[0]
                    generated_ids = output[input_length:]
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    results.append(generated_text)

        return results


class APIModel(BaseModel):
    """OpenAI APIを使用するモデル."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
    ) -> None:
        """初期化.

        Args:
            model_name: モデル名
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
        """
        super().__init__(model_name)

        if self.model_info["type"] != "api":
            raise ValueError(f"モデル {model_name} はAPIモデルではありません")

        self.api_name = self.model_info["api_name"]

        # APIキーの設定
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI APIキーが設定されていません。"
                "環境変数 OPENAI_API_KEY を設定するか、api_key引数で指定してください。"
            )

        # OpenAIクライアントの初期化（遅延インポート）
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
        except ImportError as e:
            raise ImportError("openaiパッケージがインストールされていません: uv add openai") from e

        logger.info(f"APIモデル初期化完了: {self.api_name}")

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """テキストを生成.

        Args:
            prompts: 入力プロンプトのリスト
            config: 生成設定

        Returns:
            生成されたテキストのリスト
        """
        if config is None:
            config = GenerationConfig()

        results = []

        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.api_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                n=config.num_return_sequences,
            )

            generated_text = response.choices[0].message.content or ""
            results.append(generated_text)

        return results


class VLLMModel(BaseModel):
    """vLLMを使用した高速推論モデル.

    Flash Attention 2を自動的に使用し、高速なバッチ推論を実現する.
    """

    def __init__(
        self,
        model_name: str,
        gpu_ids: str = "0",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ) -> None:
        """初期化.

        Args:
            model_name: モデル名
            gpu_ids: 使用するGPU ID（カンマ区切り）
            dtype: データ型 ("bfloat16", "float16", "auto")
            gpu_memory_utilization: GPUメモリ使用率 (0.0-1.0)
            max_model_len: 最大シーケンス長（Noneの場合はモデルのデフォルト）
        """
        super().__init__(model_name)

        if self.model_info["type"] != "local":
            raise ValueError(f"モデル {model_name} はローカルモデルではありません")

        self.hf_name = self.model_info["hf_name"]

        # GPU設定
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        tensor_parallel_size = len(gpu_ids.split(","))

        logger.info(f"vLLMでモデルをロード中: {self.hf_name}")
        logger.info(f"  GPU: {gpu_ids} (tensor_parallel_size={tensor_parallel_size})")
        logger.info(f"  dtype: {dtype}")
        logger.info(f"  gpu_memory_utilization: {gpu_memory_utilization}")

        # vLLMでサポートが限定的なモデルの警告
        vllm_limited_models = ["gemma-3-1b-it", "gemma-3-4b-it"]
        if model_name in vllm_limited_models:
            logger.warning(
                f"警告: {model_name} はvLLMでネイティブ実装がないため、"
                "Transformersフォールバックが使用されます。"
                "出力品質に問題が生じる可能性があります。"
                "LocalModelの使用を推奨します（--use-vllmを外してください）。"
            )

        try:
            from vllm import LLM

            llm_kwargs = {
                "model": self.hf_name,
                "dtype": dtype,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": True,
            }

            if max_model_len is not None:
                llm_kwargs["max_model_len"] = max_model_len

            self.llm = LLM(**llm_kwargs)

            # トークナイザーを取得（チャットテンプレート用）
            self.tokenizer = self.llm.get_tokenizer()

        except ImportError as e:
            raise ImportError("vllmパッケージがインストールされていません: uv add vllm") from e

        logger.info(f"vLLMモデルロード完了: {self.hf_name}")

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """メッセージリストにチャットテンプレートを適用.

        Args:
            messages: メッセージリスト [{"role": "...", "content": "..."}]

        Returns:
            チャットテンプレートが適用されたプロンプト文字列
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # フォールバック: 単純な結合
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant:"
            return prompt

    def _prepare_prompt(
        self,
        prompt: str | list[dict[str, str]],
    ) -> str:
        """プロンプトを準備（メッセージリストの場合はチャットテンプレートを適用）.

        Args:
            prompt: 文字列またはメッセージリスト

        Returns:
            チャットテンプレートが適用されたプロンプト文字列
        """
        if isinstance(prompt, list):
            return self._apply_chat_template(prompt)
        return prompt

    def generate(
        self,
        prompts: list[str | list[dict[str, str]]],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """テキストを生成.

        vLLMは自動的にバッチ処理を最適化するため、
        すべてのプロンプトを一度に処理する.

        Args:
            prompts: 入力プロンプトのリスト（文字列またはメッセージリスト）
            config: 生成設定

        Returns:
            生成されたテキストのリスト
        """
        if config is None:
            config = GenerationConfig()

        from vllm import SamplingParams

        # メッセージリストの場合はチャットテンプレートを適用
        formatted_prompts = [self._prepare_prompt(p) for p in prompts]

        # SamplingParams設定
        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature if config.do_sample else 0.0,
            top_p=config.top_p if config.do_sample else 1.0,
            top_k=config.top_k if config.do_sample else -1,
            n=config.num_return_sequences,
        )

        # vLLMで一括生成
        outputs = self.llm.generate(formatted_prompts, sampling_params)

        # 結果を抽出
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        return results

    def generate_batch(
        self,
        prompts: list[str | list[dict[str, str]]],
        config: GenerationConfig | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """バッチ処理でテキストを生成.

        vLLMは内部で最適なバッチ処理を行うため、
        generate()と同じ処理を行う.

        Args:
            prompts: 入力プロンプトのリスト（文字列またはメッセージリスト）
            config: 生成設定
            batch_size: バッチサイズ（vLLMでは無視される）

        Returns:
            生成されたテキストのリスト
        """
        # vLLMは自動的に最適なバッチ処理を行うため、generate()を使用
        return self.generate(prompts, config)

    def compute_choice_logprobs(
        self,
        prompt: str | list[dict[str, str]],
        choices: list[str],
    ) -> list[float]:
        """各選択肢のログ確率を計算（lm-eval-harness multiple_choice方式）.

        vLLMを使用してログ確率を計算する.

        Args:
            prompt: プロンプト（文字列またはメッセージリスト）
            choices: 選択肢のリスト（例: ["A", "B", "C", "D"]）

        Returns:
            各選択肢のログ確率のリスト
        """
        from vllm import SamplingParams

        # メッセージリストの場合はチャットテンプレートを適用
        formatted_prompt = self._prepare_prompt(prompt)

        # prompt_logprobs=1で最初のトークンのログ確率を取得
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=1,
        )

        logprobs_list = []
        for choice in choices:
            # プロンプト + 選択肢を連結
            full_text = formatted_prompt + choice

            # vLLMで推論（ログ確率取得）
            outputs = self.llm.generate([full_text], sampling_params)
            output = outputs[0]

            # 選択肢トークンのログ確率を取得
            # prompt_logprobsから選択肢部分のログ確率を抽出
            if output.prompt_logprobs:
                # 最後のトークン（選択肢）のログ確率
                choice_logprob = 0.0
                prompt_len = len(self.tokenizer.encode(formatted_prompt))
                for i, logprob_dict in enumerate(output.prompt_logprobs):
                    if i >= prompt_len and logprob_dict:
                        # トークンIDに対応するログ確率を取得
                        for _token_id, logprob_info in logprob_dict.items():
                            choice_logprob += logprob_info.logprob
                logprobs_list.append(choice_logprob)
            else:
                # フォールバック：0を返す
                logprobs_list.append(0.0)

        return logprobs_list

    def compute_choice_logprobs_batch(
        self,
        prompts: list[str | list[dict[str, str]]],
        choices_list: list[list[str]],
    ) -> list[list[float]]:
        """バッチでログ確率を計算.

        Args:
            prompts: プロンプトのリスト
            choices_list: 各プロンプトに対する選択肢リストのリスト

        Returns:
            各プロンプトに対する選択肢のログ確率リストのリスト
        """
        results = []
        for prompt, choices in zip(prompts, choices_list, strict=True):
            logprobs = self.compute_choice_logprobs(prompt, choices)
            results.append(logprobs)
        return results


def load_model(
    model_name: str,
    device: torch.device | None = None,
    api_key: str | None = None,
    use_vllm: bool = False,
    gpu_ids: str = "0",
    **kwargs: Any,
) -> BaseModel:
    """モデルをロード.

    Args:
        model_name: モデル名
        device: 使用するデバイス（ローカルモデルの場合、use_vllm=Falseの時のみ使用）
        api_key: APIキー（APIモデルの場合）
        use_vllm: vLLMを使用するか（高速推論、Flash Attention 2自動使用）
        gpu_ids: 使用するGPU ID（カンマ区切り、vLLM使用時のみ）
        **kwargs: その他のオプション

    Returns:
        ロードされたモデル
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"サポートされていないモデル: {model_name}. "
            f"サポートモデル: {list(SUPPORTED_MODELS.keys())}"
        )

    model_info = SUPPORTED_MODELS[model_name]

    if model_info["type"] == "local":
        if use_vllm:
            return VLLMModel(model_name, gpu_ids=gpu_ids, **kwargs)
        else:
            return LocalModel(model_name, device=device, **kwargs)
    elif model_info["type"] == "api":
        return APIModel(model_name, api_key=api_key)
    else:
        raise ValueError(f"不明なモデルタイプ: {model_info['type']}")


def get_supported_models(language: str | None = None) -> list[str]:
    """サポートされているモデルのリストを取得.

    Args:
        language: フィルタする言語（None の場合は全モデル）

    Returns:
        モデル名のリスト
    """
    if language is None:
        return list(SUPPORTED_MODELS.keys())

    return [name for name, info in SUPPORTED_MODELS.items() if info["language"] == language]
