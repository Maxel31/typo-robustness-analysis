"""モデルロード・推論モジュール.

HuggingFaceモデルおよびOpenAI APIモデルの統一的なインターフェースを提供する.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.utils.logger import logger

# サポートするモデルの定義
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    # 英語モデル
    "gemma-3-1b-it": {
        "hf_name": "google/gemma-3-1b-it",
        "language": "english",
        "type": "local",
    },
    "gemma-3-4b-it": {
        "hf_name": "google/gemma-3-4b-it",
        "language": "english",
        "type": "local",
    },
    "Mistral-3-8B-Instruct-2512": {
        "hf_name": "mistralai/Mistral-3-8B-Instruct-2512",
        "language": "english",
        "type": "local",
    },
    "Meta-Llama-3-8B-Instruct": {
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "language": "english",
        "type": "local",
    },
    "gpt-4-0613": {
        "api_name": "gpt-4-0613",
        "language": "english",
        "type": "api",
    },
    # 日本語モデル
    "Llama-3.1-Swallow-8B-Instruct-v0.5": {
        "hf_name": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
        "language": "japanese",
        "type": "local",
    },
    "llm-jp-3-3.7b-instruct": {
        "hf_name": "llm-jp/llm-jp-3-3.7b-instruct",
        "language": "japanese",
        "type": "local",
    },
    "llm-jp-3-13b-instruct": {
        "hf_name": "llm-jp/llm-jp-3-13b-instruct",
        "language": "japanese",
        "type": "local",
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

    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ) -> None:
        """初期化.

        Args:
            model_name: モデル名
            device: 使用するデバイス
            torch_dtype: モデルのデータ型
            use_flash_attention: Flash Attention 2を使用するか
        """
        super().__init__(model_name)

        if self.model_info["type"] != "local":
            raise ValueError(f"モデル {model_name} はローカルモデルではありません")

        self.device = device if device is not None else torch.device("cuda")
        self.torch_dtype = torch_dtype
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

        # モデルをロード
        attn_implementation = "flash_attention_2" if use_flash_attention else "eager"

        try:
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                self.hf_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
        except Exception as e:
            logger.warning(f"Flash Attention 2でのロードに失敗: {e}")
            logger.info("通常のattentionでリトライします")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model.eval()
        logger.info(f"モデルロード完了: {self.hf_name}")

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

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature if config.do_sample else None,
                    top_p=config.top_p if config.do_sample else None,
                    do_sample=config.do_sample,
                    num_return_sequences=config.num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # 入力部分を除いた生成テキストを取得
                generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                results.append(generated_text)

        return results

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """バッチ処理でテキストを生成.

        Args:
            prompts: 入力プロンプトのリスト
            config: 生成設定
            batch_size: バッチサイズ

        Returns:
            生成されたテキストのリスト
        """
        if config is None:
            config = GenerationConfig()

        results = []

        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature if config.do_sample else None,
                    top_p=config.top_p if config.do_sample else None,
                    do_sample=config.do_sample,
                    num_return_sequences=config.num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

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
        except ImportError:
            raise ImportError("openaiパッケージがインストールされていません: uv add openai")

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


def load_model(
    model_name: str,
    device: torch.device | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseModel:
    """モデルをロード.

    Args:
        model_name: モデル名
        device: 使用するデバイス（ローカルモデルの場合）
        api_key: APIキー（APIモデルの場合）
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
