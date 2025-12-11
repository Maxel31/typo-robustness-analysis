# NLP2026: LLMの文字レベル摂動に対する脆弱性分析

LLMが文字レベルの摂動(typo)に対して脆弱であることを実証し、言語リソースごとに影響の大きい単語を特定する研究プロジェクト。

## セットアップ

### 必要条件

- Python 3.11以上
- [uv](https://github.com/astral-sh/uv) パッケージマネージャー

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/Maxel31/NLP2026.git
cd NLP2026

# 依存パッケージをインストール
uv sync
```

### データセットの配置

以下のデータセットを手動で配置してください:

**英語 (SUBTLEXus)**
- ファイル: `SUBTLEX-US.xlsx`
- 配置先: `data/raw/english/SUBTLEX-US.xlsx`
- 取得元: [SUBTLEXus](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus)

**日本語 (現代日本語書き言葉均衡コーパス短単位語彙表)**
- ファイル: `BCCWJ_frequencylist_suw_ver1_0.tsv`
- 配置先: `data/raw/japanese/BCCWJ_frequencylist_suw_ver1_0.tsv`
- 取得元: [BCCWJ](https://clrd.ninjal.ac.jp/bccwj/)

## 使用方法

### 実装.1: 頻出単語抽出

言語リソースから頻出単語上位N件を抽出します。

```bash
# 英語頻出単語を抽出 (デフォルト: top 500)
uv run python scripts/run_preprocessing.py --language english --top-n 500

# 日本語頻出単語を抽出
uv run python scripts/run_preprocessing.py --language japanese --top-n 500

# 両言語を同時に処理
uv run python scripts/run_preprocessing.py --language both --top-n 500
```

**オプション**:
- `--language`: 処理する言語 (`english`, `japanese`, `both`)
- `--top-n`: 抽出する単語数 (デフォルト: 500)

**出力形式**:

出力ファイルは `data/processed/{language}/frequent_words_top{N}.json` に保存されます。

```json
{
  "metadata": {
    "source": "SUBTLEXus",
    "language": "english",
    "top_n": 500,
    "created_at": "2025-12-09T08:25:47.115437+00:00",
    "score_formula": "log(FREQcount) * log(CDcount)"
  },
  "words": [
    {
      "rank": 1,
      "word": "you",
      "score": 131.65605160097505,
      "freq_count": 2134713,
      "cd_count": 8381
    }
  ]
}
```

**日本語の場合**:
```json
{
  "metadata": {
    "source": "BCCWJ (現代日本語書き言葉均衡コーパス短単位語彙表)",
    "language": "japanese",
    "top_n": 500,
    "created_at": "2025-12-09T08:26:51.105875+00:00",
    "score_formula": "PMW (100万語単位での出現頻度)"
  },
  "words": [
    {
      "rank": 1,
      "word": "ノ",
      "lemma": "の",
      "pos": "助詞-格助詞",
      "score": 48383.909433,
      "frequency": 5061558
    }
  ]
}
```

### 実装.2: 摂動データ生成

頻出単語に対して文字レベルの摂動（typo）を適用したベンチマークデータを生成します。

```bash
# 英語ベンチマーク (GSM8K, BBH, MMLU) で摂動生成
PYTHONPATH=. uv run python scripts/run_perturbation.py \
    --language english --top-n 10

# 日本語ベンチマーク (Jamp, JNLI, NIILC, JSQuAD, JCommonsenseQA) で摂動生成
PYTHONPATH=. uv run python scripts/run_perturbation.py \
    --language japanese --top-n 10

# 特定のベンチマークのみ
PYTHONPATH=. uv run python scripts/run_perturbation.py \
    --language japanese --benchmarks jcommonsenseqa jsquad --top-n 10

# サンプルデータでテスト
PYTHONPATH=. uv run python scripts/run_perturbation.py \
    --language english --top-n 3 --use-sample-data

# カスタム確率で実行
PYTHONPATH=. uv run python scripts/run_perturbation.py \
    --language english --top-n 10 \
    --replace-prob 0.1 --insert-prob 0.1 --delete-prob 0.1
```

**オプション**:
- `--language`: 処理する言語 (`english`, `japanese`)
- `--benchmarks`: 処理するベンチマーク名（指定しない場合は言語に対応する全ベンチマーク）
  - 英語: `gsm8k`, `bbh`, `mmlu`
  - 日本語: `jamp`, `jnli`, `niilc`, `jsquad`, `jcommonsenseqa`
- `--top-n`: 対象とする頻出単語数 (デフォルト: 10)
- `--max-samples`: ベンチマークごとの最大サンプル数 (デフォルト: 全件)
- `--output-dir`: 出力ディレクトリ (デフォルト: `data/perturbed`)
- `--replace-prob`: 各文字に対する置換確率 (デフォルト: 0.2)
- `--insert-prob`: 各文字に対する挿入確率 (デフォルト: 0.2)
- `--delete-prob`: 各文字に対する削除確率 (デフォルト: 0.2)
- `--seed`: 乱数シード (デフォルト: 42)
- `--use-sample-data`: サンプルデータを使用（テスト用）

**摂動の仕組み**:
- 単語全体に対して最大1回の摂動操作（置換・挿入・削除のいずれか）
- 文字種を保持した摂動（ひらがな→ひらがな、カタカナ→カタカナ、英字→英字）
- 摂動が発生しなかったサンプルは保存されない
- 各テキスト例ごとに一意のシードで再現性を確保
- 日本語の選択肢付き問題（JCommonsenseQA等）では、選択肢部分は摂動対象外

**出力形式**:

出力ファイルは以下の構造で保存されます:

```
data/perturbed/
├── {benchmark_name}/
│   ├── original/
│   │   └── examples.json      # 元のベンチマークデータ
│   └── perturbed/
│       └── {target_word}/
│           └── examples.json  # 摂動済みデータ
```

摂動済みデータの形式:
```json
{
  "metadata": {
    "benchmark_name": "jcommonsenseqa",
    "target_word": "ます",
    "language": "japanese",
    "replace_prob": 0.2,
    "insert_prob": 0.2,
    "delete_prob": 0.2,
    "base_seed": 42,
    "num_examples": 100,
    "total_occurrences": 150,
    "perturbed_occurrences": 98,
    "target_word_score": 10639.0901585
  },
  "examples": [
    {
      "id": 0,
      "seed": 420000,
      "original_text": "りんごが5個あります。...",
      "perturbed_text": "りんごが5個ありま。...",
      "perturbations": [
        {
          "occurrence_index": 0,
          "start_position": 8,
          "end_position": 10,
          "original_word": "ます",
          "perturbed_word": "ま",
          "operations": [
            {"position": 1, "operation": "delete", "original_char": "す", "new_char": null}
          ]
        }
      ],
      "total_occurrences_in_example": 2,
      "perturbed_count_in_example": 1,
      "answer": "3"
    }
  ]
}
```

**メタデータフィールド**:
- `target_word_score`: 頻出単語リストにおけるスコア（頻出度合い）
- `total_occurrences`: データセット全体での対象単語の出現数
- `perturbed_occurrences`: 摂動が適用された出現数

### 実装.2: 推論実行と影響度分析

摂動データに対してLLM推論を実行し、各単語の影響度を分析します。

**PTモデル vs ITモデル**:
- **PTモデル（pretrained/base）**: few-shot推論（GSM8K: 8-shot CoT, BBH: 3-shot, MMLU: 5-shot）
- **ITモデル（instruction-tuned）**: 0-shot推論（チャットテンプレート適用）

```bash
# PTモデル（gemma-3-4b-pt）でGSM8Kを8-shot CoT評価
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model gemma-3-4b-pt --benchmark gsm8k

# ITモデル（gemma-3-4b-it）でGSM8Kを0-shot評価
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model gemma-3-4b-it --benchmark gsm8k

# 推論結果を保存（エラー分析用）
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model gemma-3-4b-pt --benchmark gsm8k --save-inference-results

# GPU指定（カンマ区切り）
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model gemma-3-4b-it --benchmark bbh --gpu-ids 0,1

# 上位N件の単語のみ処理（テスト用）
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model gemma-3-1b-pt --benchmark gsm8k --top-n 10

# バッチサイズ調整（メモリ不足時）
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model Mistral-7B-v0.3 --benchmark mmlu --batch-size 4

# 日本語ベンチマーク（フォントパス指定でワードクラウド生成）
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model llm-jp-3-3.7b-instruct --benchmark jcommonsenseqa \
    --font-path /path/to/NotoSansJP-Regular.ttf

# vLLMを使用した高速推論（Mistral, Llama等で推奨）
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model Mistral-7B-Instruct-v0.3 --benchmark bbh \
    --use-vllm --gpu-ids 0,1 --gpu-memory-utilization 0.85
```

**サポートモデル**:

| モデル名 | タイプ | 推論形式 | 言語 |
|---------|--------|---------|------|
| `gemma-3-1b-pt` | PT (base) | few-shot | 英語 |
| `gemma-3-4b-pt` | PT (base) | few-shot | 英語 |
| `Mistral-7B-v0.3` | PT (base) | few-shot | 英語 |
| `Meta-Llama-3.2-3B` | PT (base) | few-shot | 英語 |
| `gemma-3-1b-it` | IT (instruct) | 0-shot | 英語 |
| `gemma-3-4b-it` | IT (instruct) | 0-shot | 英語 |
| `Mistral-7B-Instruct-v0.3` | IT (instruct) | 0-shot | 英語 |
| `Meta-Llama-3.2-3B-Instruct` | IT (instruct) | 0-shot | 英語 |
| `llm-jp-3-3.7b-instruct` | IT (instruct) | 0-shot | 日本語 |
| `llm-jp-3-13b-instruct` | IT (instruct) | 0-shot | 日本語 |
| `Llama-3.1-Swallow-8B-Instruct-v0.5` | IT (instruct) | 0-shot | 日本語 |

**few-shot設定（PTモデル）**:
- GSM8K: 8-shot Chain-of-Thought (Wei et al., 2022)
- BBH: 3-shot
- MMLU: 5-shot
- 日本語ベンチマーク: 0-shot

**⚠️ vLLM互換性に関する注意**:
- `gemma-3-1b-it`、`gemma-3-4b-it`、`gemma-3-1b-pt`、`gemma-3-4b-pt`はvLLMでネイティブ実装がないため、`--use-vllm`オプションとの併用は推奨されません
- これらのモデルを使用する場合は、`--use-vllm`オプションを外してLocalModelを使用してください
- Mistral、LlamaモデルはvLLMで正常に動作します

**オプション**:
- `--model`: 使用するモデル名（必須）
- `--benchmark`: ベンチマーク名（必須）
  - 英語: `gsm8k`, `bbh`, `mmlu`
  - 日本語: `jamp`, `jnli`, `niilc`, `jsquad`, `jcommonsenseqa`
- `--gpu-ids`: 使用するGPU ID（カンマ区切り、デフォルト: `0`）
- `--batch-size`: バッチサイズ（デフォルト: 8）
- `--max-new-tokens`: 最大生成トークン数（デフォルト: 512、CoT推論対応）
- `--top-n`: 処理する単語数の上限（デフォルト: 全単語）
- `--perturbed-dir`: 摂動データディレクトリ（デフォルト: `data/perturbed`）
- `--output-dir`: 出力ディレクトリ（デフォルト: `results/experiment1`）
- `--font-path`: ワードクラウド用フォントパス（日本語の場合は必要）
- `--use-vllm`: vLLMを使用した高速推論（Flash Attention 2自動使用）
- `--gpu-memory-utilization`: GPUメモリ使用率（vLLM使用時のみ、デフォルト: 0.9）
- `--use-pt-model`: PTモデル用テキストプロンプト形式を強制使用
- `--save-inference-results`: 推論結果を完全に保存（エラー分析用）

**処理フロー**:
1. 摂動データディレクトリから元データ（original/）を読み込み
2. ベースライン推論を実行し、精度を測定
3. 各単語の摂動データ（perturbed/{word}/）に対して推論を実行
4. 摂動回数で正規化した影響度を計算
5. 影響度でソートしたランキングをJSON出力
6. ワードクラウドを生成

**評価指標**（lm-eval-harness準拠）:
- **GSM8K**: 数値のExact Match（`####`後の数値、または最後の数値を抽出）
- **BBH**: サブタスク別のExact Match
- **MMLU**: ログ確率ベース評価（`output_type: multiple_choice`）
  - 各選択肢（A/B/C/D）のログ確率を計算し、最大値の選択肢を予測として採用
  - テキスト生成ではなく、確率比較による評価で安定性向上
- **日本語ベンチマーク**: 完全一致または部分一致

**正規化影響度**:
摂動回数のばらつきを考慮した正規化を実施:
```
per_perturbation_impact = accuracy_drop / perturbed_occurrences
```

**出力形式**:

出力ファイルは以下の構造で保存されます:

```
results/experiment1/
├── {model_name}/
│   └── {benchmark_name}/
│       ├── result.json      # ランキング（正規化影響度でソート）
│       └── wordcloud.png    # ワードクラウド
```

result.jsonの形式:
```json
{
  "metadata": {
    "model_name": "gemma-3-1b-it",
    "benchmark_name": "gsm8k",
    "baseline_accuracy": 0.4523,
    "total_words_analyzed": 10,
    "normalization_method": "per_perturbation",
    "created_at": "2025-12-10T12:00:00+00:00"
  },
  "ranking": [
    {
      "rank": 1,
      "target_word": "the",
      "target_word_score": 125.34,
      "baseline_accuracy": 0.4523,
      "perturbed_accuracy": 0.3812,
      "accuracy_drop": 0.0711,
      "total_occurrences": 1250,
      "perturbed_occurrences": 980,
      "num_examples": 450,
      "normalized_impact": 0.000073,
      "per_perturbation_impact": 0.000073
    }
  ]
}
```

## 開発

### コード品質

```bash
# フォーマット
uv run --frozen ruff format .

# リントチェック
uv run --frozen ruff check .

# 自動修正
uv run --frozen ruff check . --fix

# テスト実行
uv run --frozen pytest -v
```

### ブランチ戦略

このプロジェクトはGit Flowに従います。詳細は `CLAUDE.md` を参照してください。

## ライセンス

このプロジェクトは研究目的で作成されています。
