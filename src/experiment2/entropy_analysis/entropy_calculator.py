"""エントロピー・パープレキシティ計算モジュール.

生成時の各タイムステップにおけるエントロピーとパープレキシティを計算し、
摂動パターンによる確信度の変化を分析する.

エントロピー: H_t = -Σ p_i log(p_i)
- 低エントロピー: 高確信度（特定のトークンに確率が集中）
- 高エントロピー: 低確信度（確率が分散）

パープレキシティ: PPL = exp(-1/N Σ log p(x_i))
- 低パープレキシティ: モデルの予測精度が高い
- 高パープレキシティ: モデルの予測精度が低い
"""

import math
from dataclasses import dataclass, field

import torch
import torch.distributions as dist
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class TokenEntropy:
    """トークンごとのエントロピー情報.

    Attributes:
        token_id: トークンID
        token_str: トークン文字列
        entropy: エントロピー値
        log_prob: 対数確率（パープレキシティ計算用）
        top_probs: 上位確率のリスト（デバッグ用）
        position: 生成位置（0から開始）
    """

    token_id: int
    token_str: str
    entropy: float
    log_prob: float = 0.0
    top_probs: list[tuple[str, float]] = field(default_factory=list)
    position: int = 0

    def to_dict(self) -> dict:
        """辞書形式に変換."""
        return {
            "token_id": self.token_id,
            "token_str": self.token_str,
            "entropy": self.entropy,
            "log_prob": self.log_prob,
            "top_k_predictions": dict(self.top_probs),
            "position": self.position,
        }


@dataclass
class GenerationEntropyResult:
    """生成全体のエントロピー・パープレキシティ結果.

    Attributes:
        prompt: 入力プロンプト
        generated_text: 生成されたテキスト
        token_entropies: 各トークンのエントロピー
        mean_entropy: 平均エントロピー
        max_entropy: 最大エントロピー
        min_entropy: 最小エントロピー
        perplexity: パープレキシティ
        entropy_trajectory: エントロピーの推移（位置順）
    """

    prompt: str
    generated_text: str
    token_entropies: list[TokenEntropy] = field(default_factory=list)
    mean_entropy: float = 0.0
    max_entropy: float = 0.0
    min_entropy: float = float("inf")
    perplexity: float = 0.0
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
                "perplexity": self.perplexity,
                "num_tokens": len(self.token_entropies),
            },
            "entropy_trajectory": self.entropy_trajectory,
            "token_details": [t.to_dict() for t in self.token_entropies],
        }

    def compute_statistics(self) -> None:
        """統計情報を計算."""
        if not self.token_entropies:
            return

        # 有効なエントロピー値のみをフィルタリング
        valid_entropies = [t.entropy for t in self.token_entropies if math.isfinite(t.entropy)]
        valid_log_probs = [t.log_prob for t in self.token_entropies if math.isfinite(t.log_prob)]

        # エントロピー統計
        if valid_entropies:
            self.entropy_trajectory = [
                t.entropy if math.isfinite(t.entropy) else 0.0 for t in self.token_entropies
            ]
            self.mean_entropy = sum(valid_entropies) / len(valid_entropies)
            self.max_entropy = max(valid_entropies)
            self.min_entropy = min(valid_entropies)
        else:
            self.entropy_trajectory = [0.0] * len(self.token_entropies)
            self.mean_entropy = 0.0
            self.max_entropy = 0.0
            self.min_entropy = 0.0

        # パープレキシティ計算: PPL = exp(-1/N Σ log p(x_i))
        if valid_log_probs:
            mean_log_prob = sum(valid_log_probs) / len(valid_log_probs)
            self.perplexity = math.exp(-mean_log_prob)
        else:
            self.perplexity = float("inf")


def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """ロジットからエントロピーを計算.

    数値安定性のためCategorical分布を使用.

    Args:
        logits: モデル出力のロジット (vocab_size,)
        temperature: 温度パラメータ

    Returns:
        エントロピー値（自然対数ベース）
    """
    # float32に変換して数値安定性を確保
    logits = logits.float()

    # 温度スケーリング
    scaled_logits = logits / temperature

    try:
        # Categorical分布を使用してエントロピーを計算（数値安定）
        categorical = dist.Categorical(logits=scaled_logits)
        entropy = categorical.entropy().item()

        # NaNまたはInfのチェック
        if not math.isfinite(entropy):
            return 0.0

        return entropy
    except (ValueError, RuntimeError):
        # 分布が作成できない場合（すべて同じ値など）
        return 0.0


def compute_log_prob(
    logits: torch.Tensor,
    target_token_id: int,
    temperature: float = 1.0,
) -> float:
    """特定トークンの対数確率を計算.

    Args:
        logits: モデル出力のロジット (vocab_size,)
        target_token_id: 対象トークンのID
        temperature: 温度パラメータ

    Returns:
        対数確率
    """
    # float32に変換
    logits = logits.float()

    # 温度スケーリング
    scaled_logits = logits / temperature

    # log_softmaxで数値安定に対数確率を計算
    log_probs = torch.log_softmax(scaled_logits, dim=-1)

    log_prob = log_probs[target_token_id].item()

    # NaNまたはInfのチェック
    if not math.isfinite(log_prob):
        return float("-inf")

    return log_prob


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
    # float32に変換して数値安定性を確保
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[-1]))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=True):
        token_str = tokenizer.decode([idx])
        results.append((token_str, prob))

    return results


def get_model_device(model: PreTrainedModel) -> torch.device:
    """モデルの実際のデバイスを取得.

    device_map="auto"使用時でも正しいデバイスを返す.

    Args:
        model: 言語モデル

    Returns:
        モデルのデバイス
    """
    try:
        # まずパラメータからデバイスを取得
        return next(model.parameters()).device
    except StopIteration:
        # パラメータがない場合（まれ）
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_with_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
    top_k_for_analysis: int = 5,
    top_k_sampling: int = 1,
    seed: int | None = 42,
) -> GenerationEntropyResult:
    """エントロピーを記録しながらテキストを生成.

    Args:
        model: 言語モデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        max_new_tokens: 生成する最大トークン数
        temperature: 生成温度
        device: デバイス（Noneの場合はモデルのデバイスを自動検出）
        top_k_for_analysis: 分析用に記録する上位トークン数
        top_k_sampling: 生成時のtop-kサンプリング（1=greedy, 2以上=top-kからサンプリング）
        seed: 乱数シード（top-k>1の場合に使用、Noneで非決定論的）

    Returns:
        エントロピー分析結果
    """
    model.eval()
    result = GenerationEntropyResult(prompt=prompt, generated_text="")

    # デバイスを自動検出（device_map="auto"対応）
    if device is None:
        device = get_model_device(model)
    elif isinstance(device, str):
        device = torch.device(device)

    # プロンプトをトークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_ids = input_ids.clone()
    generated_tokens = []

    # top-k sampling用のgeneratorを作成（決定論的サンプリング）
    generator = None
    if top_k_sampling > 1 and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    with torch.no_grad():
        for position in range(max_new_tokens):
            # モデル出力を取得
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]  # 最後のトークンのロジット

            # エントロピー計算
            entropy = compute_entropy(logits[0], temperature)

            # 上位トークン取得
            top_tokens = get_top_tokens(logits[0], tokenizer, top_k_for_analysis)

            # 次のトークンをサンプリング
            if top_k_sampling <= 1:
                # greedy decoding
                next_token_id = torch.argmax(logits.float(), dim=-1).item()
            else:
                # top-k sampling（generatorで決定論的）
                logits_float = logits[0].float() / temperature
                top_k_logits, top_k_indices = torch.topk(logits_float, top_k_sampling)
                probs = torch.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1, generator=generator).item()
                next_token_id = top_k_indices[sampled_idx].item()
            next_token_str = tokenizer.decode([next_token_id])

            # 対数確率を計算（パープレキシティ用）
            log_prob = compute_log_prob(logits[0], next_token_id, temperature)

            # 終了条件チェック
            if next_token_id == tokenizer.eos_token_id:
                break

            # 結果を記録
            token_entropy = TokenEntropy(
                token_id=next_token_id,
                token_str=next_token_str,
                entropy=entropy,
                log_prob=log_prob,
                top_probs=top_tokens,
                position=position,
            )
            result.token_entropies.append(token_entropy)
            generated_tokens.append(next_token_str)

            # 次のステップのために追加（同じデバイスを使用）
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=-1)

    # 生成テキストを結合
    result.generated_text = "".join(generated_tokens)

    # 統計情報を計算
    result.compute_statistics()

    return result


def generate_with_entropy_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
    top_k_for_analysis: int = 5,
    top_k_sampling: int = 1,
    seed: int | None = 42,
) -> list[GenerationEntropyResult]:
    """バッチでエントロピーを記録しながらテキストを生成.

    Args:
        model: 言語モデル
        tokenizer: トークナイザー
        prompts: 入力プロンプトのリスト
        max_new_tokens: 生成する最大トークン数
        temperature: 生成温度
        device: デバイス（Noneの場合はモデルのデバイスを自動検出）
        top_k_for_analysis: 分析用に記録する上位トークン数
        top_k_sampling: 生成時のtop-kサンプリング（1=greedy, 2以上=top-kからサンプリング）
        seed: 乱数シード（top-k>1の場合に使用、Noneで非決定論的）

    Returns:
        エントロピー分析結果のリスト
    """
    if not prompts:
        return []

    model.eval()

    # デバイスを自動検出（device_map="auto"対応）
    if device is None:
        device = get_model_device(model)
    elif isinstance(device, str):
        device = torch.device(device)

    # パディングトークンを設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # バッチでトークナイズ（左パディング：生成時に必要）
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    batch_size = len(prompts)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 各プロンプトの結果を初期化
    results = [GenerationEntropyResult(prompt=p, generated_text="") for p in prompts]
    generated_tokens_list: list[list[str]] = [[] for _ in range(batch_size)]

    # 終了フラグ（各サンプル）
    finished = [False] * batch_size

    generated_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()

    # top-k sampling用のgeneratorを作成（各サンプルごとに独立したgeneratorを使用）
    generators: list[torch.Generator | None] = []
    if top_k_sampling > 1 and seed is not None:
        for i in range(batch_size):
            gen = torch.Generator(device=device)
            # サンプルごとに異なるシードを設定（再現性を保ちながらサンプル間の独立性を確保）
            gen.manual_seed(seed + i)
            generators.append(gen)
    else:
        generators = [None] * batch_size

    with torch.no_grad():
        for position in range(max_new_tokens):
            # 全サンプルが終了したらbreak
            if all(finished):
                break

            # モデル出力を取得
            outputs = model(generated_ids, attention_mask=current_attention_mask)
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

            # 各サンプルを処理
            next_token_ids = []
            for i in range(batch_size):
                if finished[i]:
                    # 終了済みの場合はパディングトークンを追加
                    next_token_ids.append(tokenizer.pad_token_id)
                    continue

                sample_logits = logits[i]

                # エントロピー計算
                entropy = compute_entropy(sample_logits, temperature)

                # 上位トークン取得
                top_tokens = get_top_tokens(sample_logits, tokenizer, top_k_for_analysis)

                # 次のトークンをサンプリング
                if top_k_sampling <= 1:
                    # greedy decoding
                    next_token_id = torch.argmax(sample_logits.float(), dim=-1).item()
                else:
                    # top-k sampling（generatorで決定論的）
                    logits_float = sample_logits.float() / temperature
                    top_k_logits, top_k_indices = torch.topk(logits_float, top_k_sampling)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    sampled_idx = torch.multinomial(
                        probs, num_samples=1, generator=generators[i]
                    ).item()
                    next_token_id = top_k_indices[sampled_idx].item()
                next_token_str = tokenizer.decode([next_token_id])

                # 対数確率を計算（パープレキシティ用）
                log_prob = compute_log_prob(sample_logits, next_token_id, temperature)

                # 終了条件チェック
                if next_token_id == tokenizer.eos_token_id:
                    finished[i] = True
                    next_token_ids.append(tokenizer.pad_token_id)
                    continue

                next_token_ids.append(next_token_id)

                # 結果を記録
                token_entropy = TokenEntropy(
                    token_id=next_token_id,
                    token_str=next_token_str,
                    entropy=entropy,
                    log_prob=log_prob,
                    top_probs=top_tokens,
                    position=position,
                )
                results[i].token_entropies.append(token_entropy)
                generated_tokens_list[i].append(next_token_str)

            # 次のステップのために追加
            next_token_tensor = torch.tensor(next_token_ids, device=device).unsqueeze(1)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=-1)

            # attention_maskを更新（新しいトークンは全てattend）
            new_attention = torch.ones(
                (batch_size, 1), device=device, dtype=current_attention_mask.dtype
            )
            current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=-1)

    # 生成テキストを結合して統計を計算
    for i in range(batch_size):
        results[i].generated_text = "".join(generated_tokens_list[i])
        results[i].compute_statistics()

    return results


def compute_entropy_for_given_sequence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target_sequence: str,
    device: str | torch.device | None = None,
) -> GenerationEntropyResult:
    """指定されたシーケンスに対するエントロピーを計算.

    実際に生成せず、既知のシーケンスに対する
    各トークン位置でのエントロピーを計算する.

    Args:
        model: 言語モデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        target_sequence: 対象シーケンス
        device: デバイス（Noneの場合はモデルのデバイスを自動検出）

    Returns:
        エントロピー分析結果
    """
    model.eval()
    result = GenerationEntropyResult(
        prompt=prompt,
        generated_text=target_sequence,
    )

    # デバイスを自動検出（device_map="auto"対応）
    if device is None:
        device = get_model_device(model)
    elif isinstance(device, str):
        device = torch.device(device)

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

            # 対数確率を計算（パープレキシティ用）
            log_prob = compute_log_prob(logits[0], actual_token_id)

            token_entropy = TokenEntropy(
                token_id=actual_token_id,
                token_str=actual_token_str,
                entropy=entropy,
                log_prob=log_prob,
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
        "token_details": {},
    }

    for label, result in results.items():
        comparison["statistics"][label] = {
            "mean_entropy": result.mean_entropy,
            "max_entropy": result.max_entropy,
            "min_entropy": result.min_entropy if result.min_entropy != float("inf") else 0.0,
            "perplexity": result.perplexity if math.isfinite(result.perplexity) else None,
            "num_tokens": len(result.token_entropies),
        }
        comparison["trajectories"][label] = result.entropy_trajectory
        # 各タイムステップのtop-kトークンと確率を追加
        comparison["token_details"][label] = [t.to_dict() for t in result.token_entropies]

    # 差分計算（originalがある場合）
    if "original" in results:
        original_mean = results["original"].mean_entropy
        original_ppl = results["original"].perplexity
        comparison["differences"] = {}
        for label, result in results.items():
            if label != "original":
                diff_entry = {
                    "mean_entropy_diff": result.mean_entropy - original_mean,
                    "mean_entropy_ratio": (
                        result.mean_entropy / original_mean if original_mean > 0 else float("inf")
                    ),
                }
                # パープレキシティの差分（両方が有限の場合のみ）
                if math.isfinite(original_ppl) and math.isfinite(result.perplexity):
                    diff_entry["perplexity_diff"] = result.perplexity - original_ppl
                    diff_entry["perplexity_ratio"] = (
                        result.perplexity / original_ppl if original_ppl > 0 else float("inf")
                    )
                comparison["differences"][label] = diff_entry

    return comparison
