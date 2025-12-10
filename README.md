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
