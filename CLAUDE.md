# NLP2026: LLMの文字レベル摂動に対する脆弱性分析プロジェクト

## プロジェクト概要

LLMが文字レベルの摂動(typo)に対して脆弱であることを実証し、言語リソースごとに影響の大きい単語を特定する研究プロジェクト。

**研究目的**:
1. 言語リソースごとに、文字レベルの摂動を加えた際に影響の大きい単語を特定
2. 学習コーパスにおける単語頻度と摂動による推論性能低下の関係性を分析

**Notionリンク**:
- 研究計画: https://www.notion.so/ver-2-0-2c03431bc1d680d782d9f383e306866f
- 実装方針: https://www.notion.so/2c03431bc1d6808fa622ce464517afbe

---

## 実装の重要なルール

### 必須事項

1. **パッケージ管理**: `uv`のみ使用 (pipは使用禁止)
2. **コード品質**: 型ヒント必須、Google-style docstring必須
3. **テスト**: `uv run --frozen pytest`
4. **コミット**: Conventional Commits形式、ユーザ確認後にコミット
5. **フォーマット**: `uv run --frozen ruff format .` / `uv run --frozen ruff check . --fix`

### 禁止事項

1. mainブランチへの直接マージ
2. APIキーのハードコーディング
3. テストをスキップしてのコミット

---

## 実装フロー

### 実装.0: セットアップ ✅ 完了

### 実装.1: 頻出単語の特定 ✅ 完了

### 実装.2: (実験1) 文字レベルの摂動を加えた際に影響の大きい単語の特定 ✅ 完了

**進捗**:
- [x] 摂動モジュール (`src/perturbation/`)
- [x] ベンチマークローダー (`src/benchmarks/benchmark_loader.py`)
- [x] 推論・評価モジュール (`src/models/`, `src/evaluation/`)
- [x] ワードクラウド可視化 (`src/visualization/`)

### 実装.3: (実験2) 摂動パターンとモデル確信度の関係分析 🔄 進行中

**目的**: 摂動パターンがLLMの生成時確信度（エントロピー）に与える影響を分析

**摂動パターン**:
- Pattern 1: 摂動後のトークンが実在 + 同品詞
- Pattern 2: 摂動後のトークンが実在 + 異品詞
- Pattern 3: 摂動後のトークンが非実在（UNK）

**進捗**:
- [x] 品詞判定ユーティリティ (`src/perturbation/pos_utils.py`)
  - NLTKのpos_tagを使用した品詞判定
  - Penn Treebankタグの大分類マッピング
  - 機能語（限定詞、前置詞、代名詞等）の正しい処理
- [x] トークン抽出モジュール (`src/experiment2/token_extraction/`)
- [x] パターン摂動生成モジュール (`src/experiment2/pattern_perturbation/`)
  - 決定論的なパターン摂動生成
  - **search_end_position**: 選択肢部分への摂動を防止
- [x] エントロピー計算モジュール (`src/experiment2/entropy_analysis/`)
  - パープレキシティ計算機能を追加
  - バッチ処理対応
- [x] ケーススタディ分析器 (`src/experiment2/entropy_analysis/case_study_analyzer.py`)
- [x] TruthfulQAベンチマーク対応
- [x] 実行スクリプト (`scripts/run_case_study.py`, `scripts/generate_perturbation_mapping.py`)
- [x] 可視化Notebook (`notebooks/analysis/token_annotated_heatmap.ipynb`)
  - フィルタモード: `any_correct_to_incorrect` 追加
- [ ] テスト作成

---

## 使用するモデル・ベンチマーク

### モデル

**英語モデル**:
- gemma-3-1b-pt/it, gemma-3-4b-pt/it
- Mistral-7B-v0.3, Mistral-7B-Instruct-v0.3
- Meta-Llama-3.2-3B, Meta-Llama-3.2-3B-Instruct

**日本語モデル**:
- llm-jp-3-3.7b-instruct, llm-jp-3-13b-instruct
- Llama-3.1-Swallow-8B-Instruct-v0.5

### ベンチマーク

**英語**:
- GSM8K: 算数推論
- BBH: 多様な推論タスク
- MMLU: 多分野の知識理解
- **TruthfulQA**: 真実性測定（GitHubリポジトリから直接ロード）

**日本語**:
- Jamp, JNLI, NIILC, JSQuAD, JCommonsenseQA

---

## ディレクトリ構造

```
NLP2026/
├── src/
│   ├── benchmarks/            # ベンチマークローダー
│   ├── preprocessing/         # 前処理モジュール
│   ├── perturbation/          # 摂動処理モジュール
│   │   ├── perturbator.py
│   │   ├── benchmark_perturbator.py
│   │   ├── rule_based_perturbator.py
│   │   ├── wordnet_utils.py
│   │   └── pos_utils.py       # 品詞判定ユーティリティ
│   ├── models/                # モデルロード・推論
│   ├── evaluation/            # 評価・分析モジュール
│   ├── visualization/         # 可視化モジュール
│   ├── experiment1/           # 実験1用モジュール
│   ├── experiment2/           # 実験2用モジュール
│   │   ├── token_extraction/
│   │   ├── pattern_perturbation/
│   │   └── entropy_analysis/
│   └── utils/
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_perturbation.py
│   ├── run_inference.py
│   ├── run_case_study.py
│   └── generate_perturbation_mapping.py
├── notebooks/analysis/
├── tests/
└── results/
```

---

## 更新履歴

- 2025-12-28: 実験2機能を拡張
  - TruthfulQAベンチマーク追加（GitHubリポジトリから直接ロード）
  - 選択肢への摂動防止機能（search_end_position）
  - 品詞判定ユーティリティ（pos_utils.py）追加
  - パープレキシティ計算機能追加
  - MMLU抽出ロジックをMMLU-Pro準拠に改善
  - Notebookフィルタモード「any_correct_to_incorrect」追加
- 2025-12-19: 実験2モジュールを実装
- 2025-12-16: ルールベース摂動モジュールを実装
- 2025-12-11: MMLU評価をログ確率方式に変更
- 2025-12-10: 実装.2完了
- 2025-12-09: 実装.1完了、プロジェクト初期設定

---

## 参考リンク

- グローバルCLAUDE.md: `/home/sfukuhata/.claude/CLAUDE.md`
- SuperClaude Framework: `/disk/ssd14tc/sfukuhata/.claude/`
