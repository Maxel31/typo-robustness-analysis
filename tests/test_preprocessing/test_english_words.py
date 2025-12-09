"""英語頻出単語抽出モジュールのテスト."""

import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.english_words import (
    calculate_importance_score,
    extract_top_words,
    load_subtlexus,
    save_results,
)


class TestCalculateImportanceScore:
    """calculate_importance_score関数のテスト."""

    def test_normal_values(self) -> None:
        """通常の値でスコアが正しく計算されることを確認."""
        freq_count = 1000
        cd_count = 100

        expected = math.log(1000) * math.log(100)
        result = calculate_importance_score(freq_count, cd_count)

        assert result == pytest.approx(expected)

    def test_minimum_values(self) -> None:
        """最小値（1）でスコアが0になることを確認."""
        result = calculate_importance_score(1, 1)
        assert result == 0.0

    def test_zero_values_treated_as_one(self) -> None:
        """0以下の値が1として扱われることを確認."""
        result = calculate_importance_score(0, 0)
        assert result == 0.0

        result = calculate_importance_score(-1, -1)
        assert result == 0.0

    def test_mixed_values(self) -> None:
        """片方が0の場合もエラーにならないことを確認."""
        result = calculate_importance_score(1000, 0)
        # 0は1として扱われるので log(1000) * log(1) = log(1000) * 0 = 0
        assert result == 0.0


class TestExtractTopWords:
    """extract_top_words関数のテスト."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """テスト用のサンプルDataFrameを作成."""
        return pd.DataFrame(
            {
                "Word": ["the", "to", "a", "and", "of"],
                "FREQcount": [1000000, 800000, 700000, 600000, 500000],
                "CDcount": [8000, 7500, 7000, 6500, 6000],
            }
        )

    def test_extract_top_n(self, sample_df: pd.DataFrame) -> None:
        """上位N件が正しく抽出されることを確認."""
        result = extract_top_words(sample_df, top_n=3)

        assert len(result) == 3
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2
        assert result[2]["rank"] == 3

    def test_result_structure(self, sample_df: pd.DataFrame) -> None:
        """結果の構造が正しいことを確認."""
        result = extract_top_words(sample_df, top_n=1)

        assert "rank" in result[0]
        assert "word" in result[0]
        assert "score" in result[0]
        assert "freq_count" in result[0]
        assert "cd_count" in result[0]

    def test_sorted_by_score_descending(self, sample_df: pd.DataFrame) -> None:
        """スコアの降順でソートされていることを確認."""
        result = extract_top_words(sample_df, top_n=5)

        scores = [w["score"] for w in result]
        assert scores == sorted(scores, reverse=True)

    def test_missing_column_raises_error(self) -> None:
        """必要なカラムがない場合にエラーが発生することを確認."""
        df = pd.DataFrame({"Word": ["test"], "FREQcount": [100]})

        with pytest.raises(ValueError, match="必要なカラムが見つかりません"):
            extract_top_words(df, top_n=1)


class TestSaveResults:
    """save_results関数のテスト."""

    def test_save_json_format(self) -> None:
        """JSON形式で正しく保存されることを確認."""
        words = [{"rank": 1, "word": "the", "score": 100.0, "freq_count": 1000, "cd_count": 100}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.json"
            result_path = save_results(words, top_n=1, output_path=output_path)

            assert result_path.exists()

            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "words" in data
            assert data["metadata"]["source"] == "SUBTLEXus"
            assert data["metadata"]["language"] == "english"
            assert data["metadata"]["top_n"] == 1
            assert len(data["words"]) == 1


class TestLoadSubtlexus:
    """load_subtlexus関数のテスト."""

    def test_file_not_found(self) -> None:
        """ファイルが見つからない場合にエラーが発生することを確認."""
        with pytest.raises(FileNotFoundError):
            load_subtlexus(Path("/nonexistent/path/file.xlsx"))
