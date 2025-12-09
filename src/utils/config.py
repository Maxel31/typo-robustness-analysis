"""設定管理モジュール."""

from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent.parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 言語別ディレクトリ
ENGLISH_RAW_DIR = RAW_DATA_DIR / "english"
JAPANESE_RAW_DIR = RAW_DATA_DIR / "japanese"
ENGLISH_PROCESSED_DIR = PROCESSED_DATA_DIR / "english"
JAPANESE_PROCESSED_DIR = PROCESSED_DATA_DIR / "japanese"

# 結果ディレクトリ
RESULTS_DIR = PROJECT_ROOT / "results"

# デフォルト設定
DEFAULT_TOP_N = 500

# データセットファイル名
SUBTLEXUS_FILENAME = "SUBTLEX-US.xlsx"
BCCWJ_FILENAME = "BCCWJ_frequencylist_suw_ver1_0.tsv"


def ensure_directories() -> None:
    """必要なディレクトリを作成する."""
    directories = [
        ENGLISH_PROCESSED_DIR,
        JAPANESE_PROCESSED_DIR,
        RESULTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
