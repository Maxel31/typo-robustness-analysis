"""日本語頻出単語抽出モジュールのテスト."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.japanese_words import (
    extract_top_words,
    load_bccwj,
    save_results,
)


class TestExtractTopWords:
    """extract_top_words関数のテスト."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """テスト用のサンプルDataFrameを作成."""
        return pd.DataFrame(
            {
                "lForm": ["の", "に", "て", "は", "を"],
                "lemma": ["の", "に", "て", "は", "を"],
                "pos": [
                    "助詞-格助詞",
                    "助詞-格助詞",
                    "助詞-接続助詞",
                    "助詞-係助詞",
                    "助詞-格助詞",
                ],
                "pmw": [48383.9, 34188.6, 33391.0, 31448.7, 28000.0],
                "frequency": [5061558, 3576558, 3493117, 3289932, 2900000],
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
        assert "lemma" in result[0]
        assert "pos" in result[0]
        assert "score" in result[0]
        assert "frequency" in result[0]

    def test_sorted_by_pmw_descending(self, sample_df: pd.DataFrame) -> None:
        """PMWの降順でソートされていることを確認."""
        result = extract_top_words(sample_df, top_n=5)

        scores = [w["score"] for w in result]
        assert scores == sorted(scores, reverse=True)

    def test_correct_word_extraction(self, sample_df: pd.DataFrame) -> None:
        """最も頻出の単語が正しく抽出されることを確認."""
        result = extract_top_words(sample_df, top_n=1)

        assert result[0]["word"] == "の"
        assert result[0]["score"] == pytest.approx(48383.9)

    def test_missing_column_raises_error(self) -> None:
        """必要なカラムがない場合にエラーが発生することを確認."""
        df = pd.DataFrame({"lForm": ["test"], "lemma": ["test"]})

        with pytest.raises(ValueError, match="必要なカラムが見つかりません"):
            extract_top_words(df, top_n=1)


class TestSaveResults:
    """save_results関数のテスト."""

    def test_save_json_format(self) -> None:
        """JSON形式で正しく保存されることを確認."""
        words = [
            {
                "rank": 1,
                "word": "の",
                "lemma": "の",
                "pos": "助詞-格助詞",
                "score": 48383.9,
                "frequency": 5061558,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.json"
            result_path = save_results(words, top_n=1, output_path=output_path)

            assert result_path.exists()

            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "words" in data
            assert data["metadata"]["language"] == "japanese"
            assert data["metadata"]["top_n"] == 1
            assert len(data["words"]) == 1

    def test_japanese_characters_preserved(self) -> None:
        """日本語文字が正しく保存されることを確認."""
        words = [
            {
                "rank": 1,
                "word": "日本語",
                "lemma": "日本語",
                "pos": "名詞",
                "score": 100.0,
                "frequency": 1000,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.json"
            save_results(words, top_n=1, output_path=output_path)

            with open(output_path, encoding="utf-8") as f:
                data = json.load(f)

            assert data["words"][0]["word"] == "日本語"


class TestLoadBccwj:
    """load_bccwj関数のテスト."""

    def test_file_not_found(self) -> None:
        """ファイルが見つからない場合にエラーが発生することを確認."""
        with pytest.raises(FileNotFoundError):
            load_bccwj(Path("/nonexistent/path/file.tsv"))
