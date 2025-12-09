"""日本語頻出単語抽出モジュール.

現代日本語書き言葉均衡コーパス（BCCWJ）短単位語彙表から頻出単語を抽出する.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import (
    BCCWJ_FILENAME,
    DEFAULT_TOP_N,
    JAPANESE_PROCESSED_DIR,
    JAPANESE_RAW_DIR,
    ensure_directories,
)
from src.utils.logger import logger


def load_bccwj(file_path: Path | None = None) -> pd.DataFrame:
    """BCCWJデータセットを読み込む.

    Args:
        file_path: データファイルのパス（Noneの場合はデフォルトパスを使用）

    Returns:
        読み込んだDataFrame

    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    if file_path is None:
        file_path = JAPANESE_RAW_DIR / BCCWJ_FILENAME

    if not file_path.exists():
        raise FileNotFoundError(f"BCCWJファイルが見つかりません: {file_path}")

    logger.info(f"BCCWJを読み込み中: {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    logger.info(f"読み込み完了: {len(df)} 件の単語")

    return df


def extract_top_words(
    df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """上位N件の頻出単語を抽出する.

    Args:
        df: BCCWJのDataFrame
        top_n: 抽出する単語数

    Returns:
        頻出単語のリスト
    """
    logger.info(f"上位 {top_n} 件の頻出単語を抽出中...")

    # 必要なカラムの存在確認
    required_columns = ["lForm", "lemma", "pos", "pmw", "frequency"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"必要なカラムが見つかりません: {col}")

    # 2文字以上の単語のみをフィルタリング（1文字の単語は摂動対象外）
    df = df.copy()
    df = df[df["lForm"].str.len() >= 2]
    logger.info(f"2文字以上の単語にフィルタリング: {len(df)} 件")

    # pmwでソートして上位N件を取得
    df_sorted = df.sort_values("pmw", ascending=False).head(top_n)

    # 結果をリストに変換
    words = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        words.append(
            {
                "rank": rank,
                "word": str(row["lForm"]),  # 語形を使用
                "lemma": str(row["lemma"]),  # 見出し語
                "pos": str(row["pos"]),  # 品詞
                "score": float(row["pmw"]),  # PMWをスコアとして使用
                "frequency": int(row["frequency"]),  # 頻度
            }
        )

    logger.info(f"抽出完了: {len(words)} 件")
    return words


def save_results(
    words: list[dict[str, Any]],
    top_n: int,
    output_path: Path | None = None,
) -> Path:
    """結果をJSON形式で保存する.

    Args:
        words: 頻出単語のリスト
        top_n: 抽出した単語数
        output_path: 出力ファイルパス（Noneの場合はデフォルトパスを使用）

    Returns:
        保存したファイルのパス
    """
    ensure_directories()

    if output_path is None:
        output_path = JAPANESE_PROCESSED_DIR / f"frequent_words_top{top_n}.json"

    result = {
        "metadata": {
            "source": "BCCWJ (現代日本語書き言葉均衡コーパス短単位語彙表)",
            "language": "japanese",
            "top_n": top_n,
            "created_at": datetime.now(UTC).isoformat(),
            "score_formula": "PMW (100万語単位での出現頻度)",
        },
        "words": words,
    }

    logger.info(f"結果を保存中: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("保存完了")
    return output_path


def extract_japanese_frequent_words(
    top_n: int = DEFAULT_TOP_N,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """日本語頻出単語を抽出するメイン関数.

    Args:
        top_n: 抽出する単語数
        input_path: 入力ファイルパス（Noneの場合はデフォルトパスを使用）
        output_path: 出力ファイルパス（Noneの場合はデフォルトパスを使用）

    Returns:
        保存したファイルのパス
    """
    logger.info("=== 日本語頻出単語抽出を開始 ===")

    # データ読み込み
    df = load_bccwj(input_path)

    # 頻出単語抽出
    words = extract_top_words(df, top_n)

    # 結果保存
    result_path = save_results(words, top_n, output_path)

    logger.info("=== 日本語頻出単語抽出を完了 ===")
    return result_path


if __name__ == "__main__":
    # テスト実行
    extract_japanese_frequent_words(top_n=10)
