"""英語頻出単語抽出モジュール.

SUBTLEXusデータセットから頻出単語を抽出する.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import (
    DEFAULT_TOP_N,
    ENGLISH_PROCESSED_DIR,
    ENGLISH_RAW_DIR,
    SUBTLEXUS_FILENAME,
    ensure_directories,
)
from src.utils.logger import logger


def load_subtlexus(file_path: Path | None = None) -> pd.DataFrame:
    """SUBTLEXusデータセットを読み込む.

    Args:
        file_path: データファイルのパス（Noneの場合はデフォルトパスを使用）

    Returns:
        読み込んだDataFrame

    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    if file_path is None:
        file_path = ENGLISH_RAW_DIR / SUBTLEXUS_FILENAME

    if not file_path.exists():
        raise FileNotFoundError(f"SUBTLEXusファイルが見つかりません: {file_path}")

    logger.info(f"SUBTLEXusを読み込み中: {file_path}")
    df = pd.read_excel(file_path)
    logger.info(f"読み込み完了: {len(df)} 件の単語")

    return df


def calculate_importance_score(freq_count: int, cd_count: int) -> float:
    """重要度スコアを計算する.

    指標: log(FREQcount) × log(CDcount)

    Args:
        freq_count: 全体頻度（単語の総出現回数）
        cd_count: 映画単位頻度（何本の映画に出現したか）

    Returns:
        重要度スコア
    """
    # 0以下の値を防ぐため、最小値を1に設定
    freq_count = max(freq_count, 1)
    cd_count = max(cd_count, 1)

    return math.log(freq_count) * math.log(cd_count)


def extract_top_words(
    df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """上位N件の頻出単語を抽出する.

    Args:
        df: SUBTLEXusのDataFrame
        top_n: 抽出する単語数

    Returns:
        頻出単語のリスト
    """
    logger.info(f"上位 {top_n} 件の頻出単語を抽出中...")

    # 必要なカラムの存在確認
    required_columns = ["Word", "FREQcount", "CDcount"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"必要なカラムが見つかりません: {col}")

    # スコアを計算
    df = df.copy()
    df["importance_score"] = df.apply(
        lambda row: calculate_importance_score(
            int(row["FREQcount"]),
            int(row["CDcount"]),
        ),
        axis=1,
    )

    # スコアでソートして上位N件を取得
    df_sorted = df.sort_values("importance_score", ascending=False).head(top_n)

    # 結果をリストに変換
    words = []
    for rank, (_, row) in enumerate(df_sorted.iterrows(), start=1):
        words.append(
            {
                "rank": rank,
                "word": str(row["Word"]),
                "score": float(row["importance_score"]),
                "freq_count": int(row["FREQcount"]),
                "cd_count": int(row["CDcount"]),
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
        output_path = ENGLISH_PROCESSED_DIR / f"frequent_words_top{top_n}.json"

    result = {
        "metadata": {
            "source": "SUBTLEXus",
            "language": "english",
            "top_n": top_n,
            "created_at": datetime.now(UTC).isoformat(),
            "score_formula": "log(FREQcount) * log(CDcount)",
        },
        "words": words,
    }

    logger.info(f"結果を保存中: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("保存完了")
    return output_path


def extract_english_frequent_words(
    top_n: int = DEFAULT_TOP_N,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """英語頻出単語を抽出するメイン関数.

    Args:
        top_n: 抽出する単語数
        input_path: 入力ファイルパス（Noneの場合はデフォルトパスを使用）
        output_path: 出力ファイルパス（Noneの場合はデフォルトパスを使用）

    Returns:
        保存したファイルのパス
    """
    logger.info("=== 英語頻出単語抽出を開始 ===")

    # データ読み込み
    df = load_subtlexus(input_path)

    # 頻出単語抽出
    words = extract_top_words(df, top_n)

    # 結果保存
    result_path = save_results(words, top_n, output_path)

    logger.info("=== 英語頻出単語抽出を完了 ===")
    return result_path


if __name__ == "__main__":
    # テスト実行
    extract_english_frequent_words(top_n=10)
