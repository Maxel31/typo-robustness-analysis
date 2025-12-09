# NLP2026: LLMの文字レベル摂動に対する脆弱性分析プロジェクト

## プロジェクト概要

このプロジェクトは、LLMが文字レベルの摂動(typo)に対して脆弱であることを実証し、言語リソースごとに影響の大きい単語を特定することを目的とする研究プロジェクトです。

**研究背景**:
- 文字レベルの摂動(typo)により、LLMの推論性能が大きく低下することが先行研究で示されている
- どのような単語に対する摂動がより大きな影響を与えるかは未解明
- 学習コーパスにおける単語頻度と摂動による影響の関係性も不明

**研究目的**:
1. 言語リソースごとに、文字レベルの摂動を加えた際に影響の大きい単語を特定
2. 学習コーパスにおける単語頻度と摂動による推論性能低下の関係性を分析

**Notionリンク**:
- 研究計画: https://www.notion.so/ver-2-0-2c03431bc1d680d782d9f383e306866f
- 実装方針: https://www.notion.so/2c03431bc1d6808fa622ce464517afbe

---

## Git Workflow (Git Flow準拠)

### ブランチ戦略

**重要**: このプロジェクトは厳格なGit Flow戦略に従います。

```text
main (本番環境、絶対に直接コミットしない)
├── develop (開発統合ブランチ)
│   ├── experiment1/setup (実装.0: セットアップ)
│   ├── experiment1/preprocessing (実装.1: 頻出単語の特定)
│   │   ├── experiment1/preprocessing/english-words (英語頻出単語処理)
│   │   └── experiment1/preprocessing/japanese-words (日本語頻出単語処理)
│   └── experiment1/perturbation-analysis (実装.2: 摂動分析)
│       ├── experiment1/perturbation-analysis/inference (推論実行)
│       ├── experiment1/perturbation-analysis/evaluation (評価処理)
│       └── experiment1/perturbation-analysis/visualization (可視化)
```

### ブランチ命名規則

- `experiment{N}/{phase}`: 実験番号とフェーズを示す (例: `experiment1/setup`)
- `experiment{N}/{phase}/{feature}`: より詳細な機能を示す (例: `experiment1/preprocessing/english-words`)

### ブランチ操作の鉄則

1. **mainブランチへの直接コミット・マージは絶対禁止**
2. **developブランチからのみ機能ブランチを切る**
3. **1つのブランチ = 1つの機能**
4. **機能完成後はdevelopへのPR経由でマージ**
5. **すべてのコミット前に必ずテストと型チェックを実行**

---

## 実装の重要なルール

### 必須事項

1. **パッケージ管理**: `uv`のみ使用 (pipは使用禁止)
   - インストール: `uv add package`
   - 開発依存: `uv add --dev package`
   - アップグレード: `uv add --dev package --upgrade-package package`

2. **コード品質**
   - すべてのコードに型ヒント必須
   - Google-styleのdocstring必須
   - 関数/クラスの責務を明確に

3. **テスト**
   - 実行: `uv run --frozen pytest`
   - 新機能には必ずテストを追加
   - バグ修正には回帰テストを追加

4. **コミット規則**
   - Conventional Commits形式を使用
   - 1コミット = 1機能
   - 動作テスト・型チェックをすべてパスした場合のみコミット
   - **ユーザに出力結果を確認してもらってからコミット**

5. **フォーマット・リント**
   - フォーマット: `uv run --frozen ruff format .`
   - チェック: `uv run --frozen ruff check .`
   - 修正: `uv run --frozen ruff check . --fix`

### 禁止事項

1. **要件定義の勝手な変更**
2. **mainブランチへの直接マージ**
3. **APIキーのハードコーディング** (.envファイルを使用)
4. **テストをスキップしてのコミット**
5. **複数機能を含む単一コミット**

---

## 実装フロー

### 実装.0: セットアップ ✅ 完了
- [x] Gitディレクトリの初期化
- [x] ディレクトリ構造の構築
- [x] CLAUDE.mdの作成
- [x] Python仮想環境の設定 (uv使用)
- [x] 基本的な依存パッケージのインストール
- [x] .gitignoreファイルの作成
- [x] .env.exampleファイルの作成
- [x] pyproject.tomlの作成と設定
- [x] mainブランチへの初期コミット
- [x] developブランチの作成

### 実装.1: (前処理) 言語リソースごとの頻出単語の特定 ✅ 完了

**目的**: 文字レベルの摂動を加える単語候補リストを作成

**処理内容**:
- 英語: SUBTLEXusから頻出単語上位N件を抽出
  - 指標: (全体頻度のlog) × (映画単位頻度のlog)
- 日本語: 現代日本語書き言葉均衡コーパスから頻出単語上位N件を抽出
  - 指標: PMW (100万語単位での出現頻度)

**入出力**:
- 入力: データセット(Excel形式、ユーザが格納)
- 出力: `data/processed/{language}/frequent_words_top{N}.json`
- 実行時引数: N (300~1000程度)

**完了タスク**:
- [x] ユーティリティモジュール実装 (`src/utils/config.py`, `src/utils/logger.py`)
- [x] 英語頻出単語抽出モジュール実装 (`src/preprocessing/english_words.py`)
- [x] 日本語頻出単語抽出モジュール実装 (`src/preprocessing/japanese_words.py`)
- [x] 実行スクリプト作成 (`scripts/run_preprocessing.py`)
- [x] ユニットテスト作成 (`tests/test_preprocessing/`)
- [x] 実データでの動作確認 (英語: 74,286件、日本語: 185,136件から抽出)

### 実装.2: (実験.1) 文字レベルの摂動を加えた際に影響の大きい単語の特定

**目的**: 各単語への摂動がLLMの推論性能に与える影響を測定

**処理内容**:
1. 候補リスト内の各単語に文字レベルの摂動を加える
   - 摂動方法: 各10%の確率で置換・挿入・削除
   - ベンチマークで推論精度を測定
2. 摂動回数で正規化した推論精度低下度合いを算出
3. 影響の大きい単語をランキング化
4. ワードクラウドで可視化

**出力構造**:
```
results/
├── {model_name}/
│   ├── {benchmark_name}/
│   │   ├── result.json  # ソート済みランキング
│   │   ├── wordcloud.png  # ワードクラウド
│   │   └── {perturbed_word}/
│   │       └── perturbations.json  # 摂動詳細
```

---

## 使用するモデル・データセット・ベンチマーク

### モデル

**英語モデル**:
- gemma-3-1b-it
- gemma-3-4b-it
- Mistral-3-8B-Instruct-2512
- Meta-Llama-3-8B-Instruct
- gpt4-0613 (API経由)

**日本語モデル**:
- Llama-3.1-Swallow-8B-Instruct-v0.5
- llm-jp-3-3.7b-instruct
- llm-jp-3-13b-instruct

### データセット (頻出単語抽出用)

**英語**: SUBTLEXus (Brysbaert et al., 2012)
- 8,338件の映画字幕から構築
- 総語数: 約5,100万語

**日本語**: 現代日本語書き言葉均衡コーパス 短単位語彙表 ver.1.0
- 約1億430万語のコーパス
- UniDicによる形態素解析

### ベンチマーク (推論性能測定用)

**英語**:
- GSM8K: 算数推論
- BBH: 多様な推論タスク
- MMLU: 多分野の知識理解

**日本語**:
- llm-jp-eval: 日本語総合評価

---

## ディレクトリ構造

```
NLP2026/
├── .claude/           # Claude Code設定
├── .git/              # Gitリポジトリ
├── .gitignore         # Git除外設定
├── CLAUDE.md          # このファイル
├── README.md          # プロジェクト説明
├── pyproject.toml     # Python設定 (uv管理)
├── .python-version    # Python バージョン指定
├── .env               # 環境変数 (APIキーなど、Git除外)
├── .env.example       # 環境変数テンプレート
├── data/              # データディレクトリ
│   ├── raw/           # 元データ (ユーザが配置)
│   │   ├── english/   # SUBTLEXus
│   │   └── japanese/  # 現代日本語書き言葉均衡コーパス
│   └── processed/     # 処理済みデータ
│       ├── english/   # 英語頻出単語リスト
│       └── japanese/  # 日本語頻出単語リスト
├── src/               # ソースコード
│   ├── __init__.py
│   ├── preprocessing/ # 前処理モジュール
│   │   ├── __init__.py
│   │   ├── english_words.py
│   │   └── japanese_words.py
│   ├── perturbation/  # 摂動処理モジュール
│   │   ├── __init__.py
│   │   ├── perturbator.py
│   │   └── evaluator.py
│   ├── models/        # モデルロード・推論
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── inference.py
│   └── utils/         # ユーティリティ
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── tests/             # テストコード
│   ├── __init__.py
│   ├── test_preprocessing/
│   ├── test_perturbation/
│   └── test_models/
├── scripts/           # 実行スクリプト
│   ├── run_preprocessing.py
│   └── run_experiment.py
├── results/           # 実験結果
│   └── experiment1/
│       ├── {model_name}/
│       │   └── {benchmark_name}/
└── notebooks/         # Jupyter Notebook (分析用)
    └── analysis/
```

---

## GPU設定

```python
def setup_device(gpu_id: str = "0") -> torch.device:
    """GPUの利用可能性を確認し、適切なデバイスを返す.

    Args:
        gpu_id: 使用するGPU ID (デフォルト: "0")

    Returns:
        torch.device: 使用するデバイス
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} (GPU ID: {gpu_id})")
    return device
```

---

## 開発フロー

### 新しい機能を実装する場合

1. **ブランチ作成**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b experiment1/{phase}/{feature}
   ```

2. **実装**
   - 型ヒント付きでコードを書く
   - docstringを書く
   - テストを書く

3. **品質チェック**
   ```bash
   uv run --frozen ruff format .
   uv run --frozen ruff check . --fix
   uv run --frozen pytest
   ```

4. **動作確認**
   - 出力結果をユーザに確認してもらう
   - 想定通りの動作であることを確認

5. **コミット**
   ```bash
   git add .
   git commit -m "feat(phase): 機能の説明"
   ```

6. **プッシュとPR**
   ```bash
   git push origin experiment1/{phase}/{feature}
   # GitHubでdevelopへのPRを作成
   ```

---

## トラブルシューティング

### よくある問題

1. **uvコマンドが見つからない**
   - uvをインストール: `pip install uv` または公式インストーラを使用

2. **GPUが認識されない**
   - `nvidia-smi`でGPUを確認
   - CUDA_VISIBLE_DEVICESを確認

3. **APIキーエラー**
   - `.env`ファイルが存在するか確認
   - `.env`ファイルに正しいキーが設定されているか確認

---

## 更新履歴

- 2025-12-09: 実装.1完了 - 英語・日本語頻出単語抽出機能を実装
- 2025-12-09: プロジェクト初期設定、CLAUDE.md作成

---

## 参考リンク

- グローバルCLAUDE.md: `/home/sfukuhata/.claude/CLAUDE.md`
- SuperClaude Framework: `/disk/ssd14tc/sfukuhata/.claude/`
- Notion研究計画: https://www.notion.so/ver-2-0-2c03431bc1d680d782d9f383e306866f
- Notion実装方針: https://www.notion.so/2c03431bc1d6808fa622ce464517afbe
