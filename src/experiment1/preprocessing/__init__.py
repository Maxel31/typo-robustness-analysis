"""実験1: 前処理モジュール.

SUBTLEXus/現代日本語コーパスから頻出単語を抽出する.
"""

# 既存のpreprocessingモジュールを再エクスポート
from src.preprocessing.english_words import (
    calculate_importance_score,
    extract_english_frequent_words,
    load_subtlexus,
)
from src.preprocessing.english_words import (
    extract_top_words as extract_english_top_words,
)
from src.preprocessing.japanese_words import (
    extract_japanese_frequent_words,
    load_bccwj,
)
from src.preprocessing.japanese_words import (
    extract_top_words as extract_japanese_top_words,
)

__all__ = [
    "calculate_importance_score",
    "extract_english_frequent_words",
    "extract_english_top_words",
    "extract_japanese_frequent_words",
    "extract_japanese_top_words",
    "load_bccwj",
    "load_subtlexus",
]
