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
