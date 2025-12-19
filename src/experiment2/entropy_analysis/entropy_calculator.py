"""エントロピー計算モジュール.

生成時の各タイムステップにおけるエントロピーを計算し、
摂動パターンによる確信度の変化を分析する.

エントロピー: H_t = -Σ p_i log(p_i)
- 低エントロピー: 高確信度（特定のトークンに確率が集中）
- 高エントロピー: 低確信度（確率が分散）
"""

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class TokenEntropy:
    """トークンごとのエントロピー情報.

    Attributes:
        token_id: トークンID
        token_str: トークン文字列
        entropy: エントロピー値
        top_probs: 上位確率のリスト（デバッグ用）
        position: 生成位置（0から開始）
    """

    token_id: int
    token_str: str
    entropy: float
    top_probs: list[tuple[str, float]] = field(default_factory=list)
    position: int = 0

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "token_id": self.token_id,
            "token_str": self.token_str,
            "entropy": self.entropy,
            "top_probs": [{"token": t, "prob": p} for t, p in self.top_probs[:5]],
            "position": self.position,
        }


@dataclass
class GenerationEntropyResult:
    """生成全体のエントロピー結果.

    Attributes:
        prompt: 入力プロンプト
        generated_text: 生成されたテキスト
        token_entropies: 各トークンのエントロピー
        mean_entropy: 平均エントロピー
        max_entropy: 最大エントロピー
        min_entropy: 最小エントロピー
        entropy_trajectory: エントロピーの推移（位置順）
    """

    prompt: str
    generated_text: str
    token_entropies: list[TokenEntropy] = field(default_factory=list)
    mean_entropy: float = 0.0
    max_entropy: float = 0.0
    min_entropy: float = float("inf")
    entropy_trajectory: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "prompt": self.prompt[:200] + "..." if len(self.prompt) > 200 else self.prompt,
            "generated_text": self.generated_text,
            "statistics": {
                "mean_entropy": self.mean_entropy,
                "max_entropy": self.max_entropy,
                "min_entropy": self.min_entropy if self.min_entropy != float("inf") else 0.0,
                "num_tokens": len(self.token_entropies),
            },
            "entropy_trajectory": self.entropy_trajectory,
            "token_details": [t.to_dict() for t in self.token_entropies],
        }

    def compute_statistics(self) -> None:
        """統計情報を計算."""
        if not self.token_entropies:
            return

        entropies = [t.entropy for t in self.token_entropies]
        self.entropy_trajectory = entropies
        self.mean_entropy = sum(entropies) / len(entropies)
        self.max_entropy = max(entropies)
        self.min_entropy = min(entropies)


def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """ロジットからエントロピーを計算.

    Args:
        logits: モデル出力のロジット (vocab_size,)
        temperature: 温度パラメータ

    Returns:
        エントロピー値（自然対数ベース）
    """
    # 温度スケーリング
    scaled_logits = logits / temperature

    # ソフトマックスで確率分布に変換
    probs = torch.softmax(scaled_logits, dim=-1)

    # エントロピー計算: H = -Σ p_i log(p_i)
    # log(0)を避けるため、小さな値を加算
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs).item()

    return entropy


def get_top_tokens(
    logits: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """上位トークンと確率を取得.

    Args:
        logits: モデル出力のロジット
        tokenizer: トークナイザー
        top_k: 上位何件を取得するか

    Returns:
        (トークン文字列, 確率)のリスト
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=True):
        token_str = tokenizer.decode([idx])
        results.append((token_str, prob))

    return results


def generate_with_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    device: str | torch.device = "cuda",
    top_k_for_analysis: int = 5,
) -> GenerationEntropyResult:
    """エントロピーを記録しながらテキストを生成.

    Args:
        model: 言語モデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        max_new_tokens: 生成する最大トークン数
        temperature: 生成温度
        device: デバイス
        top_k_for_analysis: 分析用に記録する上位トークン数

    Returns:
        エントロピー分析結果
    """
    model.eval()
    result = GenerationEntropyResult(prompt=prompt, generated_text="")

    # プロンプトをトークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_ids = input_ids.clone()
    generated_tokens = []

    with torch.no_grad():
        for position in range(max_new_tokens):
            # モデル出力を取得
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]  # 最後のトークンのロジット

            # エントロピー計算
            entropy = compute_entropy(logits[0], temperature)

            # 上位トークン取得
            top_tokens = get_top_tokens(logits[0], tokenizer, top_k_for_analysis)

            # 次のトークンをサンプリング（greedy）
            next_token_id = torch.argmax(logits, dim=-1).item()
            next_token_str = tokenizer.decode([next_token_id])

            # 終了条件チェック
            if next_token_id == tokenizer.eos_token_id:
                break

            # 結果を記録
            token_entropy = TokenEntropy(
                token_id=next_token_id,
                token_str=next_token_str,
                entropy=entropy,
                top_probs=top_tokens,
                position=position,
            )
            result.token_entropies.append(token_entropy)
            generated_tokens.append(next_token_str)

            # 次のステップのために追加
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=-1)

    # 生成テキストを結合
    result.generated_text = "".join(generated_tokens)

    # 統計情報を計算
    result.compute_statistics()

    return result


def compute_entropy_for_given_sequence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target_sequence: str,
    device: str | torch.device = "cuda",
) -> GenerationEntropyResult:
    """指定されたシーケンスに対するエントロピーを計算.

    実際に生成せず、既知のシーケンスに対する
    各トークン位置でのエントロピーを計算する.

    Args:
        model: 言語モデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        target_sequence: 対象シーケンス
        device: デバイス

    Returns:
        エントロピー分析結果
    """
    model.eval()
    result = GenerationEntropyResult(
        prompt=prompt,
        generated_text=target_sequence,
    )

    # 全体をトークナイズ
    full_text = prompt + target_sequence
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    full_ids = full_inputs["input_ids"][0]

    # プロンプト部分のトークン数を取得
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = prompt_inputs["input_ids"].shape[1]

    # ターゲットシーケンスの各トークンに対してエントロピーを計算
    with torch.no_grad():
        for i in range(prompt_length, len(full_ids)):
            # i番目のトークンまでを入力
            context_ids = full_ids[:i].unsqueeze(0)

            # モデル出力を取得
            outputs = model(context_ids)
            logits = outputs.logits[:, -1, :]

            # エントロピー計算
            entropy = compute_entropy(logits[0])

            # 上位トークン取得
            top_tokens = get_top_tokens(logits[0], tokenizer)

            # 実際のトークン
            actual_token_id = full_ids[i].item()
            actual_token_str = tokenizer.decode([actual_token_id])

            token_entropy = TokenEntropy(
                token_id=actual_token_id,
                token_str=actual_token_str,
                entropy=entropy,
                top_probs=top_tokens,
                position=i - prompt_length,
            )
            result.token_entropies.append(token_entropy)

    result.compute_statistics()
    return result


def compare_entropy_trajectories(
    results: dict[str, GenerationEntropyResult],
) -> dict:
    """複数の生成結果のエントロピー推移を比較.

    Args:
        results: ラベルからGenerationEntropyResultへのマッピング

    Returns:
        比較結果の辞書
    """
    comparison = {
        "labels": list(results.keys()),
        "statistics": {},
        "trajectories": {},
    }

    for label, result in results.items():
        comparison["statistics"][label] = {
            "mean_entropy": result.mean_entropy,
            "max_entropy": result.max_entropy,
            "min_entropy": result.min_entropy if result.min_entropy != float("inf") else 0.0,
            "num_tokens": len(result.token_entropies),
        }
        comparison["trajectories"][label] = result.entropy_trajectory

    # 差分計算（originalがある場合）
    if "original" in results:
        original_mean = results["original"].mean_entropy
        comparison["differences"] = {}
        for label, result in results.items():
            if label != "original":
                comparison["differences"][label] = {
                    "mean_entropy_diff": result.mean_entropy - original_mean,
                    "mean_entropy_ratio": (
                        result.mean_entropy / original_mean if original_mean > 0 else float("inf")
                    ),
                }

    return comparison
