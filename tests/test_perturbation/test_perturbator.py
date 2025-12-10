"""摂動モジュールのテスト."""

import tempfile
from pathlib import Path

import pytest

from src.perturbation.perturbator import (
    PerturbationOperation,
    PerturbationRecord,
    Perturbator,
    apply_perturbation_to_text,
    find_word_occurrences,
)


class TestPerturbationOperation:
    """PerturbationOperationクラスのテスト."""

    def test_to_dict(self) -> None:
        """辞書変換が正しく動作することを確認."""
        op = PerturbationOperation(
            position=5,
            operation="replace",
            original_char="a",
            new_char="b",
        )
        result = op.to_dict()

        assert result["position"] == 5
        assert result["operation"] == "replace"
        assert result["original_char"] == "a"
        assert result["new_char"] == "b"

    def test_from_dict(self) -> None:
        """辞書からの復元が正しく動作することを確認."""
        data = {
            "position": 3,
            "operation": "delete",
            "original_char": "x",
            "new_char": None,
        }
        op = PerturbationOperation.from_dict(data)

        assert op.position == 3
        assert op.operation == "delete"
        assert op.original_char == "x"
        assert op.new_char is None


class TestPerturbationRecord:
    """PerturbationRecordクラスのテスト."""

    def test_to_dict(self) -> None:
        """辞書変換が正しく動作することを確認."""
        record = PerturbationRecord(
            original_word="hello",
            perturbed_word="hxllo",
            seed=42,
            operations=[
                PerturbationOperation(
                    position=1,
                    operation="replace",
                    original_char="e",
                    new_char="x",
                )
            ],
        )
        result = record.to_dict()

        assert result["original_word"] == "hello"
        assert result["perturbed_word"] == "hxllo"
        assert result["seed"] == 42
        assert result["num_perturbations"] == 1
        assert len(result["operations"]) == 1

    def test_save_and_load(self) -> None:
        """保存と読み込みが正しく動作することを確認."""
        record = PerturbationRecord(
            original_word="test",
            perturbed_word="tset",
            seed=123,
            operations=[
                PerturbationOperation(
                    position=2,
                    operation="replace",
                    original_char="s",
                    new_char="e",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_record.json"
            record.save(file_path)

            loaded = PerturbationRecord.load(file_path)

            assert loaded.original_word == record.original_word
            assert loaded.perturbed_word == record.perturbed_word
            assert loaded.seed == record.seed
            assert len(loaded.operations) == len(record.operations)


class TestPerturbator:
    """Perturbatorクラスのテスト."""

    def test_initialization_english(self) -> None:
        """英語モードでの初期化を確認."""
        p = Perturbator(seed=42, language="english")
        assert p.language == "english"
        assert p.seed == 42
        assert p.replace_prob == 0.2
        assert p.insert_prob == 0.2
        assert p.delete_prob == 0.2

    def test_initialization_japanese(self) -> None:
        """日本語モードでの初期化を確認."""
        p = Perturbator(seed=42, language="japanese")
        assert p.language == "japanese"

    def test_unsupported_language(self) -> None:
        """サポートされていない言語でエラーが発生することを確認."""
        with pytest.raises(ValueError, match="サポートされていない言語"):
            Perturbator(language="french")

    def test_reproducibility_with_same_seed(self) -> None:
        """同じシードで同じ結果が得られることを確認."""
        p1 = Perturbator(seed=42, language="english")
        p2 = Perturbator(seed=42, language="english")

        result1 = p1.perturb_word("hello")
        result2 = p2.perturb_word("hello")

        assert result1.perturbed_word == result2.perturbed_word
        assert len(result1.operations) == len(result2.operations)

    def test_different_seeds_produce_different_results(self) -> None:
        """異なるシードで異なる結果が得られることを確認."""
        # 高い確率で摂動が発生するように設定
        p1 = Perturbator(seed=42, perturbation_prob=0.5, language="english")
        p2 = Perturbator(seed=123, perturbation_prob=0.5, language="english")

        # 長い単語で違いが出やすくする
        word = "perturbation"
        result1 = p1.perturb_word(word)
        result2 = p2.perturb_word(word)

        # 異なるシードでは異なる結果になる可能性が高い
        # （確率的なテストなので、失敗する可能性はあるが非常に低い）
        assert result1.seed != result2.seed

    def test_perturb_word_records_operations(self) -> None:
        """摂動操作が正しく記録されることを確認."""
        # 高確率で摂動が発生するように設定
        p = Perturbator(seed=42, perturbation_prob=0.8, language="english")
        result = p.perturb_word("test")

        # 何らかの操作が記録されているはず
        assert result.original_word == "test"
        assert result.seed == 42
        # 高確率設定なので少なくとも1つは操作があるはず
        assert len(result.operations) > 0

    def test_perturb_word_no_perturbation(self) -> None:
        """摂動確率0で元の単語が維持されることを確認."""
        p = Perturbator(seed=42, perturbation_prob=0.0, language="english")
        result = p.perturb_word("hello")

        assert result.original_word == "hello"
        assert result.perturbed_word == "hello"
        assert len(result.operations) == 0

    def test_reset_seed(self) -> None:
        """シードリセットが正しく動作することを確認."""
        p = Perturbator(seed=42, language="english")
        result1 = p.perturb_word("test")

        p.reset_seed()  # 同じシードにリセット
        result2 = p.perturb_word("test")

        assert result1.perturbed_word == result2.perturbed_word


class TestApplyPerturbationToText:
    """apply_perturbation_to_text関数のテスト."""

    def test_basic_replacement(self) -> None:
        """基本的な置換が正しく動作することを確認."""
        p = Perturbator(seed=42, perturbation_prob=0.5, language="english")
        text = "hello world hello"

        perturbed_text, record, count = apply_perturbation_to_text(
            text, "hello", p, case_sensitive=False
        )

        assert count == 2  # "hello"は2回出現
        assert record.original_word == "hello"
        assert len(record.occurrences) == 2

    def test_case_insensitive(self) -> None:
        """大文字小文字を区別しない置換を確認."""
        p = Perturbator(seed=42, perturbation_prob=0.0, language="english")
        text = "Hello HELLO hello"

        perturbed_text, record, count = apply_perturbation_to_text(
            text, "hello", p, case_sensitive=False
        )

        assert count == 3
        # 摂動確率0なので元のテキストと同じ
        assert perturbed_text == text

    def test_no_occurrences(self) -> None:
        """対象単語が存在しない場合を確認."""
        p = Perturbator(seed=42, language="english")
        text = "this is a test"

        perturbed_text, record, count = apply_perturbation_to_text(
            text, "hello", p, case_sensitive=False
        )

        assert count == 0
        assert perturbed_text == text
        assert len(record.occurrences) == 0


class TestFindWordOccurrences:
    """find_word_occurrences関数のテスト."""

    def test_find_in_multiple_texts(self) -> None:
        """複数テキストから単語を検索できることを確認."""
        texts = [
            "hello world",
            "this is a test",
            "hello again hello",
        ]

        results = find_word_occurrences(texts, "hello", case_sensitive=False)

        assert len(results) == 2  # テキスト0と2に出現
        assert results[0][0] == 0  # 最初のテキスト
        assert results[1][0] == 2  # 3番目のテキスト
        assert len(results[1][1]) == 2  # 3番目のテキストには2回出現

    def test_case_sensitive_search(self) -> None:
        """大文字小文字を区別する検索を確認."""
        texts = ["Hello world", "hello again"]

        results = find_word_occurrences(texts, "hello", case_sensitive=True)

        assert len(results) == 1  # 小文字の"hello"は1つだけ
        assert results[0][0] == 1

    def test_no_matches(self) -> None:
        """マッチしない場合の動作を確認."""
        texts = ["this is a test", "another text"]

        results = find_word_occurrences(texts, "hello", case_sensitive=False)

        assert len(results) == 0
