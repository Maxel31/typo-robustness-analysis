# NLP2026: LLMの文字レベル摂動に対する脆弱性分析

LLMが文字レベルの摂動(typo)に対して脆弱であることを実証し、言語リソースごとに影響の大きい単語を特定する研究プロジェクト。

## 目次

- [セットアップ](#セットアップ)
- [実験概要](#実験概要)
- [実験1: 単語頻度と正解率の関係分析](#実験1-単語頻度と正解率の関係分析)
- [実験2: 摂動パターンとモデル確信度の分析](#実験2-摂動パターンとモデル確信度の分析)
- [開発ガイド](#開発ガイド)

---

## セットアップ

### 必要条件

- Python 3.11以上
- [uv](https://github.com/astral-sh/uv) パッケージマネージャー

### インストール

```bash
git clone https://github.com/Maxel31/NLP2026.git
cd NLP2026
uv sync
```

### データセットの配置

| 言語 | データセット | 配置先 |
|-----|------------|-------|
| 英語 | SUBTLEXus (`SUBTLEX-US.xlsx`) | `data/raw/english/` |
| 日本語 | 現代日本語書き言葉均衡コーパス短単位語彙表 (`BCCWJ_frequencylist_suw_ver1_0.tsv`) | `data/raw/japanese/` |

---

## 実験概要

| 実験 | 目的 | 手法 |
|-----|-----|-----|
| **実験1** | 頻出単語への摂動がLLM正解率に与える影響を測定 | ランダム摂動 → 推論 → 正解率比較 |
| **実験2** | 摂動パターン（同品詞/異品詞/非実在語）がモデル確信度に与える影響を分析 | 決定論的摂動 → エントロピー計算 → パターン比較 |

### サポートベンチマーク

| 言語 | ベンチマーク |
|-----|------------|
| 英語 | GSM8K, BBH, MMLU, TruthfulQA |
| 日本語 | Jamp, JNLI, NIILC, JSQuAD, JCommonsenseQA |

### サポートモデル

| タイプ | モデル | 推論形式 |
|-------|-------|---------|
| PT (base) | gemma-3-1b-pt, gemma-3-4b-pt, Mistral-7B-v0.3, Meta-Llama-3.2-3B | few-shot |
| IT (instruct) | gemma-3-1b-it, gemma-3-4b-it, Mistral-7B-Instruct-v0.3, Meta-Llama-3.2-3B-Instruct | 0-shot |
| 日本語IT | llm-jp-3-3.7b-instruct, llm-jp-3-13b-instruct, Llama-3.1-Swallow-8B-Instruct-v0.5 | 0-shot |

---

## 実験1: 単語頻度と正解率の関係分析

頻出単語に対してランダムな文字レベル摂動を加え、LLMの正解率への影響を測定します。

### 1.1 頻出単語抽出

```bash
# 英語頻出単語を抽出
uv run python scripts/run_preprocessing.py --language english --top-n 500

# 日本語頻出単語を抽出
uv run python scripts/run_preprocessing.py --language japanese --top-n 500
```

出力: `data/processed/{language}/frequent_words_top{N}.json`

### 1.2 摂動データ生成

```bash
# 英語ベンチマークで摂動生成
PYTHONPATH=. uv run python scripts/run_perturbation.py \
    --language english --top-n 10

# カスタム確率で実行
PYTHONPATH=. uv run python scripts/run_perturbation.py \
    --language english --top-n 10 \
    --replace-prob 0.1 --insert-prob 0.1 --delete-prob 0.1
```

### 1.3 推論実行と影響度分析

```bash
# PTモデル（few-shot）で評価
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model gemma-3-4b-pt --benchmark gsm8k

# ITモデル（0-shot）で評価
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model gemma-3-4b-it --benchmark gsm8k

# vLLMを使用した高速推論
PYTHONPATH=. uv run python scripts/run_inference.py \
    --model Mistral-7B-Instruct-v0.3 --benchmark bbh \
    --use-vllm --gpu-ids 0,1
```

**評価指標**（lm-eval-harness公式実装準拠）:
- GSM8K: 文字列のExact Match
- BBH: サブタスク別のExact Match（全27サブタスクの3-shot examples内蔵）
- MMLU: ログ確率ベース評価
- TruthfulQA: MC1/MC2スコア

**正規化影響度**: `per_perturbation_impact = accuracy_drop / perturbed_occurrences`

---

## 実験2: 摂動パターンとモデル確信度の分析

摂動パターン（同品詞/異品詞/非実在語）がLLMの生成時確信度（エントロピー）に与える影響を分析します。

### 2.1 摂動パターンの定義

| パターン | 定義 | 例 |
|---------|------|-----|
| Pattern 1 | 摂動後が辞書に実在 + 同品詞 | cat → bat |
| Pattern 2 | 摂動後が辞書に実在 + 異品詞 | who → mho |
| Pattern 3 | 摂動後が辞書に非実在 | cat → czt |

### 2.2 実験フロー

```
Step 1: 摂動マッピング生成 → Step 2: ケーススタディ実行 → Step 3: Notebook分析
```

### Step 1: 摂動マッピング生成

```bash
# 基本的な使用方法
PYTHONPATH=. uv run python scripts/generate_perturbation_mapping.py \
    --benchmark gsm8k --top-n 200 \
    --model gemma-3-1b-it \
    --output data/perturbation_mappings/gsm8k_top200.json

# 推奨: サブワードチェック + 全パターン必須 + 目標件数指定
PYTHONPATH=. uv run python scripts/generate_perturbation_mapping.py \
    --benchmark mmlu --top-n 2000 \
    --model gemma-3-4b-it \
    --require-all-patterns --target-count 500 \
    --check-subword \
    --output data/perturbation_mappings/mmlu_allpatterns_500.json
```

**主要オプション**:
| オプション | 説明 |
|-----------|------|
| `--benchmark` | ベンチマーク名（gsm8k, bbh, mmlu, truthfulqa） |
| `--model` | トークナイザー用モデル名（必須） |
| `--require-all-patterns` | Pattern 1/2/3すべて生成できた単語のみ保存 |
| `--target-count` | 目標件数（達したら処理終了） |
| `--check-subword` | 摂動後もトークン数が同じか確認 |

### Step 2: ケーススタディ実行

```bash
# 基本的な使用方法
PYTHONPATH=. uv run python scripts/run_case_study.py \
    --benchmark gsm8k --model gemma-3-1b-it --num-samples 10 \
    --mapping-file data/perturbation_mappings/gsm8k_top200.json

# 推奨設定（バッチ処理、全サンプル）
PYTHONPATH=. uv run python scripts/run_case_study.py \
    --benchmark gsm8k --model gemma-3-4b-it \
    --mapping-file data/perturbation_mappings/gsm8k_allpatterns_500.json \
    --num-samples all --max-new-tokens 128 \
    --batch-size 64 --top-k 5 --gpu-id 0
```

**重要**: 選択肢部分（MMLU/TruthfulQAの(A)〜(D)など）は摂動対象から除外されます。`search_end_position`機能により、問題文のみに摂動が適用されます。

**主要オプション**:
| オプション | 説明 |
|-----------|------|
| `--num-samples` | サンプル数（"all"で全件） |
| `--num-perturbations` | 1文あたりの摂動箇所数 |
| `--batch-size` | 推論バッチサイズ |
| `--top-k` | Top-kサンプリング |

### Step 3: Notebook分析

```bash
uv run jupyter notebook notebooks/analysis/token_annotated_heatmap.ipynb
```

**フィルタリングモード**（Cell 3で指定）:
- `all`: 全サンプル
- `p1_only`, `p2_only`, `p3_only`: 特定パターンで正解→不正解
- `p1_or_p3_only`: Pattern 1または3で転落
- `any_correct_to_incorrect`: いずれかのパターンで転落

**分析内容**:
1. サンプル分類（unchanged_correct, changed_correct, changed_incorrect, unchanged_incorrect）
2. エントロピー軌跡の可視化
3. 確信度ヒートマップ
4. 発散位置の特定

### 2.3 分析指標

**エントロピー**: `H_t = -Σ p_i log(p_i)`
- 低い値（0に近い）: 高確信度
- 高い値: 低確信度

**期待される傾向**: `original < pattern1 < pattern2 < pattern3`

---

## 開発ガイド

### コード品質

```bash
uv run --frozen ruff format .    # フォーマット
uv run --frozen ruff check .     # リントチェック
uv run --frozen pytest -v        # テスト実行
```

### ディレクトリ構造

```
NLP2026/
├── data/
│   ├── raw/                    # 元データ
│   ├── processed/              # 処理済みデータ
│   ├── perturbed/              # 摂動済みベンチマーク（実験1）
│   └── perturbation_mappings/  # 摂動マッピング（実験2）
├── src/
│   ├── experiment1/            # 実験1用モジュール
│   ├── experiment2/            # 実験2用モジュール
│   ├── models/                 # モデルロード・推論
│   ├── benchmarks/             # ベンチマークローダー
│   ├── perturbation/           # 摂動生成（共通）
│   └── visualization/          # 可視化
├── scripts/                    # 実行スクリプト
├── notebooks/analysis/         # 分析用Notebook
├── results/                    # 実験結果
└── tests/                      # テストコード
```

### ブランチ戦略

Git Flowに従います。詳細は `CLAUDE.md` を参照してください。

---

## ライセンス

このプロジェクトは研究目的で作成されています。
