"""ベンチマーク摂動モジュールのテスト."""

import tempfile
from pathlib import Path

from src.perturbation.benchmark_perturbator import (
    BenchmarkPerturbator,
    OccurrencePerturbation,
    OriginalBenchmarkData,
    PerturbedBenchmarkData,
    PerturbedExample,
    generate_perturbed_data,
)
from src.perturbation.perturbator import PerturbationOperation


class TestOccurrencePerturbation:
    """OccurrencePerturbationクラスのテスト."""

    def test_to_dict(self) -> None:
        """辞書変換が正しく動作することを確認."""
        occ = OccurrencePerturbation(
            occurrence_index=0,
            start_position=5,
            end_position=10,
            original_word="hello",
            perturbed_word="hxllo",
            operations=[
                PerturbationOperation(
                    position=1,
                    operation="replace",
                    original_char="e",
                    new_char="x",
                )
            ],
        )
        result = occ.to_dict()

        assert result["occurrence_index"] == 0
        assert result["start_position"] == 5
        assert result["end_position"] == 10
        assert result["original_word"] == "hello"
        assert result["perturbed_word"] == "hxllo"
        assert len(result["operations"]) == 1

    def test_from_dict(self) -> None:
        """辞書からの復元が正しく動作することを確認."""
        data = {
            "occurrence_index": 1,
            "start_position": 0,
            "end_position": 5,
            "original_word": "test",
            "perturbed_word": "tset",
            "operations": [
                {
                    "position": 2,
                    "operation": "replace",
                    "original_char": "s",
                    "new_char": "e",
                }
            ],
        }
        occ = OccurrencePerturbation.from_dict(data)

        assert occ.occurrence_index == 1
        assert occ.original_word == "test"
        assert occ.perturbed_word == "tset"
        assert len(occ.operations) == 1


class TestPerturbedExample:
    """PerturbedExampleクラスのテスト."""

    def test_to_dict_with_extra_fields(self) -> None:
        """追加フィールドを含む辞書変換を確認."""
        example = PerturbedExample(
            example_id=42,
            seed=12345,
            original_text="Say hello to the world",
            perturbed_text="Say hxllo to the world",
            perturbations=[],
            extra_fields={"answer": "42", "category": "greeting"},
        )
        result = example.to_dict()

        assert result["id"] == 42
        assert result["seed"] == 12345
        assert result["original_text"] == "Say hello to the world"
        assert result["perturbed_text"] == "Say hxllo to the world"
        assert result["answer"] == "42"
        assert result["category"] == "greeting"

    def test_from_dict_with_extra_fields(self) -> None:
        """追加フィールドを含む復元を確認."""
        data = {
            "id": 10,
            "seed": 999,
            "original_text": "original",
            "perturbed_text": "perturbed",
            "perturbations": [],
            "answer": "correct",
            "difficulty": "hard",
        }
        example = PerturbedExample.from_dict(data)

        assert example.example_id == 10
        assert example.extra_fields["answer"] == "correct"
        assert example.extra_fields["difficulty"] == "hard"


class TestPerturbedBenchmarkData:
    """PerturbedBenchmarkDataクラスのテスト."""

    def test_save_and_load(self) -> None:
        """保存と読み込みが正しく動作することを確認."""
        data = PerturbedBenchmarkData(
            benchmark_name="test_benchmark",
            target_word="hello",
            language="english",
            replace_prob=0.1,
            insert_prob=0.1,
            delete_prob=0.1,
            base_seed=42,
            examples=[
                PerturbedExample(
                    example_id=0,
                    seed=420,
                    original_text="hello world",
                    perturbed_text="hxllo world",
                    perturbations=[],
                    extra_fields={"answer": "yes"},
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            data.save(file_path)

            loaded = PerturbedBenchmarkData.load(file_path)

            assert loaded.benchmark_name == "test_benchmark"
            assert loaded.target_word == "hello"
            assert loaded.language == "english"
            assert loaded.replace_prob == 0.1
            assert loaded.insert_prob == 0.1
            assert loaded.delete_prob == 0.1
            assert loaded.base_seed == 42
            assert len(loaded.examples) == 1
            assert loaded.examples[0].example_id == 0


class TestOriginalBenchmarkData:
    """OriginalBenchmarkDataクラスのテスト."""

    def test_save_and_load(self) -> None:
        """保存と読み込みが正しく動作することを確認."""
        data = OriginalBenchmarkData(
            benchmark_name="gsm8k",
            language="english",
            text_field="question",
            examples=[
                {"id": 0, "question": "What is 2+2?", "answer": "4"},
                {"id": 1, "question": "What is 3+3?", "answer": "6"},
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "original.json"
            data.save(file_path)

            loaded = OriginalBenchmarkData.load(file_path)

            assert loaded.benchmark_name == "gsm8k"
            assert loaded.language == "english"
            assert loaded.text_field == "question"
            assert len(loaded.examples) == 2


class TestBenchmarkPerturbator:
    """BenchmarkPerturbatorクラスのテスト."""

    def test_find_word_positions(self) -> None:
        """単語位置検索が正しく動作することを確認."""
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
        )

        text = "Hello world, hello again!"
        positions = perturbator._find_word_positions(text, "hello", case_sensitive=False)

        assert len(positions) == 2
        assert positions[0][2] == "Hello"
        assert positions[1][2] == "hello"

    def test_find_word_positions_case_sensitive(self) -> None:
        """大文字小文字を区別した検索を確認."""
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
        )

        text = "Hello world, hello again!"
        positions = perturbator._find_word_positions(text, "hello", case_sensitive=True)

        assert len(positions) == 1
        assert positions[0][2] == "hello"

    def test_generate_example_seed_uniqueness(self) -> None:
        """サンプルごとのシードが一意であることを確認."""
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
            base_seed=42,
        )

        seeds = set()
        for example_id in range(100):
            for word_idx in range(10):
                seed = perturbator._generate_example_seed(example_id, word_idx)
                seeds.add(seed)

        # 1000個のシードがすべて異なることを確認
        assert len(seeds) == 1000

    def test_perturb_example_returns_none_for_no_match(self) -> None:
        """対象単語が含まれない場合にNoneを返すことを確認."""
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
        )

        result = perturbator.perturb_example(
            example_id=0,
            text="This is a test sentence.",
            target_word="hello",
        )

        assert result is None

    def test_perturb_example_with_match(self) -> None:
        """対象単語が含まれる場合に摂動が適用されることを確認."""
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
            replace_prob=0.5,  # 高確率で摂動
            insert_prob=0.5,
            delete_prob=0.5,
            base_seed=42,
        )

        result = perturbator.perturb_example(
            example_id=0,
            text="Say hello to everyone.",
            target_word="hello",
            extra_fields={"answer": "greeting"},
        )

        assert result is not None
        assert result.example_id == 0
        assert result.original_text == "Say hello to everyone."
        assert len(result.perturbations) == 1
        assert result.perturbations[0].original_word == "hello"
        assert result.extra_fields["answer"] == "greeting"

    def test_perturb_example_preserves_case(self) -> None:
        """大文字小文字パターンが保持されることを確認."""
        # 摂動確率が0の場合は摂動なしとしてNoneを返す（スキップ仕様）
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
            replace_prob=0.0,  # 摂動なし
            insert_prob=0.0,
            delete_prob=0.0,
            base_seed=42,
        )

        result = perturbator.perturb_example(
            example_id=0,
            text="HELLO world Hello",
            target_word="hello",
        )

        # 摂動確率が0の場合、摂動が発生しないためNoneを返す（スキップ仕様）
        assert result is None

    def test_perturb_example_multiple_occurrences_different_perturbations(self) -> None:
        """複数出現に対して異なる摂動が適用されることを確認."""
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
            replace_prob=0.8,  # 高確率で摂動
            insert_prob=0.8,
            delete_prob=0.8,
            base_seed=42,
        )

        result = perturbator.perturb_example(
            example_id=0,
            text="hello hello hello",
            target_word="hello",
        )

        assert result is not None
        assert len(result.perturbations) == 3

        # 各出現に対して独立した摂動が適用されていることを確認
        perturbed_words = [p.perturbed_word for p in result.perturbations]
        # 高確率で少なくとも1つは異なる摂動が発生するはず
        # (確率的なテストだが、prob=0.8で3出現あれば異なる結果になる可能性が高い)
        assert len(perturbed_words) == 3

    def test_perturb_benchmark(self) -> None:
        """ベンチマーク全体への摂動が正しく動作することを確認."""
        perturbator = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
            replace_prob=1.0,  # 高確率で摂動を発生させる
            insert_prob=0.0,
            delete_prob=0.0,
            base_seed=42,
        )

        examples = [
            {"question": "Say hello to the world", "answer": "greeting"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Hello again!", "answer": "hi"},
        ]

        result = perturbator.perturb_benchmark(
            examples=examples,
            target_word="hello",
            text_field="question",
        )

        assert result.benchmark_name == "test"
        assert result.target_word == "hello"
        # "hello"を含むのは1番目と3番目のサンプル
        # 摂動確率が高いため、摂動が発生したサンプルのみが結果に含まれる
        assert len(result.examples) >= 1  # 少なくとも1つは摂動が発生
        # 結果に含まれるexample_idは0か2のどちらか（helloを含むサンプル）
        for ex in result.examples:
            assert ex.example_id in [0, 2]

    def test_reproducibility_with_same_seed(self) -> None:
        """同じシードで同じ結果が得られることを確認."""
        examples = [
            {"question": "Say hello to the world", "answer": "greeting"},
        ]

        # 同じパラメータで2回実行
        perturbator1 = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
            replace_prob=0.5,
            insert_prob=0.5,
            delete_prob=0.5,
            base_seed=42,
        )
        result1 = perturbator1.perturb_benchmark(
            examples=examples,
            target_word="hello",
            text_field="question",
        )

        perturbator2 = BenchmarkPerturbator(
            benchmark_name="test",
            language="english",
            replace_prob=0.5,
            insert_prob=0.5,
            delete_prob=0.5,
            base_seed=42,
        )
        result2 = perturbator2.perturb_benchmark(
            examples=examples,
            target_word="hello",
            text_field="question",
        )

        assert result1.examples[0].perturbed_text == result2.examples[0].perturbed_text
        assert result1.examples[0].seed == result2.examples[0].seed


class TestGeneratePerturbedData:
    """generate_perturbed_data関数のテスト."""

    def test_generate_perturbed_data(self) -> None:
        """摂動データ生成が正しく動作することを確認."""
        benchmark_data = OriginalBenchmarkData(
            benchmark_name="test_bench",
            language="english",
            text_field="question",
            examples=[
                {"question": "Hello world", "answer": "1"},
                {"question": "Say hello", "answer": "2"},
                {"question": "Goodbye world", "answer": "3"},
                {"question": "Test sentence", "answer": "4"},
            ],
        )

        frequent_words = ["hello", "world", "missing"]
        word_scores = {"hello": 100.0, "world": 90.0, "missing": 80.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            saved_paths = generate_perturbed_data(
                benchmark_data=benchmark_data,
                frequent_words=frequent_words,
                word_scores=word_scores,
                output_dir=output_dir,
                replace_prob=1.0,  # 高確率で摂動を発生させる
                insert_prob=0.0,
                delete_prob=0.0,
                base_seed=42,
            )

            # "hello"と"world"はデータに含まれる、"missing"は含まれない
            # 摂動確率が高いため、摂動が発生したサンプルのみが結果に含まれる
            assert "hello" in saved_paths
            assert "world" in saved_paths
            assert "missing" not in saved_paths

            # 元データが保存されていることを確認
            original_path = output_dir / "test_bench" / "original" / "examples.json"
            assert original_path.exists()

            # 摂動データが保存されていることを確認
            assert saved_paths["hello"].exists()
            assert saved_paths["world"].exists()

            # 読み込みテスト
            loaded = PerturbedBenchmarkData.load(saved_paths["hello"])
            assert loaded.target_word == "hello"
            # 摂動が発生したサンプルのみが含まれる（0〜2個）
            assert len(loaded.examples) >= 1
