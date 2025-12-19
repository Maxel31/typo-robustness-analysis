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

### 実装.2: ルールベース摂動生成

ランダム摂動とは別に、言語学的な規則に基づいた摂動を生成します。WordNetを使用して、1文字置換で実在する同品詞の単語になる摂動パターンを列挙します。

```bash
# 頻出単語上位20件に対してパターン1摂動を生成
PYTHONPATH=. uv run python scripts/run_rule_based_perturbation.py \
    --input data/processed/english/frequent_words_top2000.json \
    --output data/processed/english/pattern1_perturbations.json \
    --top-n 20

# 全単語を処理（--top-nを省略）
PYTHONPATH=. uv run python scripts/run_rule_based_perturbation.py \
    --input data/processed/english/frequent_words_top500.json \
    --output data/processed/english/pattern1_all.json
```

**オプション**:
- `--input`: 入力ファイル（`frequent_words_top{N}.json`形式）（必須）
- `--output`: 出力ファイルパス（必須）
- `--top-n`: 処理する単語数の上限（省略時は全単語）

**パターン1の定義**:
1文字置換（substitution）により、元の単語と同じ品詞を持つ別の実在単語になる摂動。
- 例: `cat` → `bat`, `cut`, `car` など
- WordNetを使用して単語の存在と品詞を検証
- 名詞(n)、動詞(v)、形容詞(a)、副詞(r)に対応

**出力形式**:
```json
{
  "metadata": {
    "source_file": "data/processed/english/frequent_words_top2000.json",
    "top_n": 20,
    "total_words_processed": 20,
    "words_with_perturbations": 7,
    "total_perturbations": 153,
    "pattern_type": "pattern1_same_pos_substitution",
    "created_at": "2025-12-16T10:30:00+00:00"
  },
  "results": [
    {
      "original_word": "have",
      "original_rank": 4,
      "original_score": 120.5,
      "pos": ["v"],
      "perturbations": [
        {
          "perturbed_word": "hive",
          "operation": "substitute",
          "position": 1,
          "original_char": "a",
          "new_char": "i",
          "shared_pos": ["n", "v"]
        },
        {
          "perturbed_word": "gave",
          "operation": "substitute",
          "position": 0,
          "original_char": "h",
          "new_char": "g",
          "shared_pos": ["v"]
        }
      ],
      "perturbation_count": 2
    }
  ]
}
```

**出力フィールド**:
- `original_word`: 元の単語
- `original_rank`: 頻出単語リストにおけるランク
- `original_score`: 頻出度スコア
- `pos`: 元の単語の品詞（WordNetで取得）
- `perturbations`: パターン1に該当する摂動のリスト
  - `perturbed_word`: 摂動後の単語
  - `operation`: 操作タイプ（常に`substitute`）
  - `position`: 置換位置（0-indexed）
  - `original_char`: 元の文字
  - `new_char`: 置換後の文字
  - `shared_pos`: 元の単語と共有する品詞
- `perturbation_count`: 該当する摂動の総数

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

**評価指標**（lm-eval-harness公式実装に完全準拠）:
- **GSM8K**: 文字列のExact Match
  - 正規化: `regexes_to_ignore: ["#### ", ",", "\\$", "\\."]`, `ignore_case: true`
  - `####`後の数値、または最後の数値を抽出
- **BBH**: サブタスク別のExact Match（厳密な大文字小文字区別）
  - 全27サブタスクの3-shot examplesを内蔵
  - Chain-of-Thought形式: "Let's think step by step."
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

### 実装.3: 摂動パターン vs モデル確信度分析（実験2）

摂動パターンがLLMの生成時確信度（エントロピー）に与える影響を分析します。

**摂動パターン**:
- Pattern 1: 摂動後のトークンが実在 + 同品詞（例: cat → bat）
- Pattern 2: 摂動後のトークンが実在 + 異品詞（例: run → bun）
- Pattern 3: 摂動後のトークンが非実在（UNK）（例: cat → czt）

**エントロピーとは**:
```
H_t = -Σ p_i log(p_i)
```
- 低エントロピー（0に近い）: モデルが高確信度で予測（特定のトークンに確率が集中）
- 高エントロピー（大きい値）: モデルが低確信度で予測（確率が分散）

#### ケーススタディ分析の実行（初期分析）

少数のサンプルを使用して、摂動パターンによるエントロピー変化を詳細に分析します。手法の検証や初期分析に使用します。

```python
# Pythonスクリプトから実行（GPU必要）
from pathlib import Path
from src.experiment2.entropy_analysis.case_study_analyzer import run_case_study

# GSM8Kでケーススタディを実行
result = run_case_study(
    benchmark_name="gsm8k",         # ベンチマーク名
    model_name="gemma-3-1b-it",     # モデル名
    num_samples=10,                 # 分析するサンプル数
    max_new_tokens=30,              # 生成する最大トークン数
    gpu_id="0",                     # 使用するGPU ID
    output_path=Path("results/experiment2/case_study/gsm8k_gemma3_1b_analysis.json"),
)

# 結果サマリーを確認
print(f"分析サンプル数: {result.num_samples}")
for pattern, stats in result.aggregate_statistics.items():
    if pattern != "differences":
        print(f"  {pattern}: 平均エントロピー={stats['avg_mean_entropy']:.4f}")
```

**クイック分析（デバッグ・検証用）**:
```python
from src.experiment2.entropy_analysis.case_study_analyzer import quick_analysis

# 5サンプルで簡易テスト
result = quick_analysis(
    model_name="gemma-3-1b-it",
    num_samples=5,
    gpu_id="0",
)
```

**パラメータ**:
- `benchmark_name`: ベンチマーク名（`gsm8k`, `bbh`, `mmlu`）
- `model_name`: モデル名（サポートモデル一覧は上記参照）
- `num_samples`: 分析するサンプル数（デフォルト: 10）
- `target_tokens`: 摂動対象トークンリスト（Noneで自動抽出）
- `subset`: サブセット名（BBH/MMLUの場合）
- `max_new_tokens`: 生成する最大トークン数（デフォルト: 50）
- `output_path`: 結果の出力先パス
- `gpu_id`: 使用するGPU ID

#### 出力形式

**ケーススタディ結果のJSON構造**:
```json
{
  "metadata": {
    "benchmark_name": "gsm8k",
    "model_name": "gemma-3-1b-it",
    "num_samples": 10
  },
  "aggregate_statistics": {
    "original": {
      "avg_mean_entropy": 2.3456,
      "avg_max_entropy": 5.1234,
      "num_samples": 10
    },
    "pattern1": {
      "avg_mean_entropy": 2.5678,
      "avg_max_entropy": 5.3456,
      "num_samples": 10
    },
    "pattern2": {
      "avg_mean_entropy": 2.8901,
      "avg_max_entropy": 5.6789,
      "num_samples": 10
    },
    "pattern3": {
      "avg_mean_entropy": 3.2345,
      "avg_max_entropy": 6.1234,
      "num_samples": 10
    },
    "differences": {
      "pattern1": {
        "avg_entropy_increase": 0.2222,
        "avg_entropy_ratio": 1.0947
      },
      "pattern2": {
        "avg_entropy_increase": 0.5445,
        "avg_entropy_ratio": 1.2321
      },
      "pattern3": {
        "avg_entropy_increase": 0.8889,
        "avg_entropy_ratio": 1.3789
      }
    }
  },
  "samples": [
    {
      "sample_id": 0,
      "benchmark_name": "gsm8k",
      "original_text": "Janet's ducks lay 16 eggs per day...",
      "perturbed_texts": {
        "pattern1": "Janet's ducks lay 16 eggs per dey...",
        "pattern2": "Janet's ducks ley 16 eggs per day...",
        "pattern3": "Janet's ducks lay 16 egxs per day..."
      },
      "entropy_comparison": {
        "labels": ["original", "pattern1", "pattern2", "pattern3"],
        "statistics": {
          "original": {
            "mean_entropy": 2.1234,
            "max_entropy": 4.5678,
            "min_entropy": 0.2345,
            "num_tokens": 25
          }
        },
        "trajectories": {
          "original": [2.1, 1.8, 2.3, 1.5, ...],
          "pattern1": [2.3, 2.0, 2.5, 1.7, ...]
        },
        "differences": {
          "pattern1": {
            "mean_entropy_diff": 0.1234,
            "mean_entropy_ratio": 1.0581
          }
        }
      },
      "perturbation_details": {
        "pattern1": [
          {"original": "day", "perturbed": "dey", "pattern": "pattern1", "count": 2}
        ]
      }
    }
  ]
}
```

#### 出力フィールドの解説

**aggregate_statistics（集計統計）**:
| フィールド | 説明 |
|-----------|------|
| `avg_mean_entropy` | 全サンプルの平均エントロピーの平均値 |
| `avg_max_entropy` | 全サンプルの最大エントロピーの平均値 |
| `num_samples` | 分析に成功したサンプル数 |

**differences（パターン間差分）**:
| フィールド | 説明 | 解釈 |
|-----------|------|------|
| `avg_entropy_increase` | 元テキスト比でのエントロピー増加量 | 正の値 = 確信度低下 |
| `avg_entropy_ratio` | 元テキスト比でのエントロピー比率 | 1.0 = 変化なし、>1.0 = 確信度低下 |

**entropy_comparison（各サンプルのエントロピー比較）**:
| フィールド | 説明 |
|-----------|------|
| `mean_entropy` | そのサンプルの平均エントロピー |
| `max_entropy` | そのサンプルの最大エントロピー（最も不確実な予測位置） |
| `min_entropy` | そのサンプルの最小エントロピー（最も確信度が高い位置） |
| `num_tokens` | 生成されたトークン数 |
| `trajectories` | タイムステップごとのエントロピー値の推移 |

#### 結果の解釈方法

**エントロピー値の目安**:
- `< 1.0`: 非常に高い確信度（次トークンがほぼ確定）
- `1.0 - 3.0`: 通常の確信度
- `3.0 - 5.0`: やや低い確信度
- `> 5.0`: 低い確信度（複数の候補で迷っている状態）

**パターン間比較の解釈**:
1. **Pattern 1 vs Original**: 同品詞の実在語への摂動の影響
   - 差が小さい場合: 文脈から意味を復元できている
   - 差が大きい場合: 単語の正確性が重要な文脈

2. **Pattern 2 vs Original**: 異品詞の実在語への摂動の影響
   - Pattern 1より大きな影響が予想される
   - 品詞情報がモデルの予測に重要であることを示唆

3. **Pattern 3 vs Original**: 非実在語（UNK）への摂動の影響
   - 最も大きな影響が予想される
   - モデルの未知語処理能力を反映

**期待される傾向**:
```
original < pattern1 < pattern2 < pattern3
（エントロピー増加順）
```

この傾向が見られる場合、モデルは単語の「実在性」と「品詞一致」の両方を文脈理解に活用していると解釈できます。

#### その他のモジュール

**トークン抽出**:
```python
from src.experiment2.token_extraction.benchmark_token_extractor import (
    extract_frequent_tokens,
)

# ベンチマークからサブワード単位で頻出トークンを抽出
result = extract_frequent_tokens(
    benchmark_name="gsm8k",
    model_name="gemma-3-1b-it",
    top_n=300,
    unit="subword",  # "subword" または "word"
)
print(f"ユニークトークン数: {result.unique_tokens}")
```

**パターン摂動マッピング生成**:
```python
from src.experiment2.pattern_perturbation.pattern_generator import (
    generate_perturbation_mapping_table,
)

tokens = ["day", "cat", "run", "have"]
mapping = generate_perturbation_mapping_table(
    tokens,
    require_all_patterns=True,  # 全パターンが必要
    seed=42,
)
print(f"全パターン成功: {len(mapping)} tokens")

# マッピング内容を確認
for token, result in mapping.items():
    print(f"\n{token}:")
    for pattern, m in result.mappings.items():
        if m:
            print(f"  {pattern}: {m.original} -> {m.perturbed}")
```

**出力構造**:
```
results/experiment2/
├── token_extraction/
│   └── {benchmark_name}_{model_name}_tokens.json
├── perturbation_mapping/
│   └── {benchmark_name}_mapping.json
└── case_study/
    └── {benchmark_name}_{model_name}_analysis.json
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
