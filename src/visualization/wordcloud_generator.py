"""ワードクラウド生成モジュール.

影響度に基づくワードクラウドを生成する.
"""

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from src.evaluation.analyzer import AnalysisResult
from src.utils.logger import logger

# 非GUIバックエンドを使用
matplotlib.use("Agg")


def generate_impact_wordcloud(
    analysis: AnalysisResult,
    font_path: str | None = None,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    colormap: str = "Reds",
    max_words: int = 100,
) -> WordCloud:
    """影響度に基づくワードクラウドを生成.

    Args:
        analysis: 分析結果
        font_path: フォントファイルのパス（日本語の場合は必須）
        width: 画像の幅
        height: 画像の高さ
        background_color: 背景色
        colormap: カラーマップ名
        max_words: 最大単語数

    Returns:
        生成されたWordCloudオブジェクト
    """
    # 影響度を正規化して頻度辞書を作成
    # 正規化影響度をそのまま使用（負の値は0にクリップ）
    word_frequencies: dict[str, float] = {}

    for impact in analysis.word_impacts:
        # 正規化影響度が正の場合のみ追加
        if impact.normalized_impact > 0:
            # スコアを1000倍してWordCloud用に調整
            word_frequencies[impact.target_word] = impact.normalized_impact * 1000

    if not word_frequencies:
        logger.warning("影響度が正の単語がありません。空のワードクラウドを生成します。")
        # 最低1つの単語を追加
        if analysis.word_impacts:
            word_frequencies[analysis.word_impacts[0].target_word] = 1.0

    logger.info(f"ワードクラウド生成: {len(word_frequencies)}単語")

    # WordCloud設定
    wc_params: dict[str, Any] = {
        "width": width,
        "height": height,
        "background_color": background_color,
        "colormap": colormap,
        "max_words": max_words,
        "prefer_horizontal": 0.7,
        "relative_scaling": 0.5,
        "min_font_size": 10,
    }

    # 日本語フォント対応
    if font_path:
        wc_params["font_path"] = font_path

    wordcloud = WordCloud(**wc_params)
    wordcloud.generate_from_frequencies(word_frequencies)

    return wordcloud


def save_wordcloud(
    wordcloud: WordCloud,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """ワードクラウドを画像として保存.

    Args:
        wordcloud: WordCloudオブジェクト
        output_path: 出力ファイルパス
        dpi: 画像のDPI

    Returns:
        保存したファイルのパス
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # matplotlibで保存
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"ワードクラウドを保存: {output_path}")
    return output_path


def generate_and_save_wordcloud(
    analysis: AnalysisResult,
    output_dir: Path,
    font_path: str | None = None,
) -> Path:
    """ワードクラウドを生成して保存.

    Args:
        analysis: 分析結果
        output_dir: 出力ディレクトリ
        font_path: フォントファイルのパス

    Returns:
        保存したファイルのパス
    """
    wordcloud = generate_impact_wordcloud(analysis, font_path=font_path)
    output_path = output_dir / "wordcloud.png"
    return save_wordcloud(wordcloud, output_path)
