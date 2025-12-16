"""evaluatorモジュールのテスト."""

from src.evaluation.evaluator import (
    extract_answer_bbh,
    extract_answer_mmlu,
    extract_number_from_text,
    normalize_choice_answer,
    normalize_gsm8k_answer,
)


class TestExtractAnswerMMLU:
    """MMLU回答抽出のテスト.

    lm-eval-harness公式の三層フォールバック戦略に準拠:
    1. "The answer is X" パターン
    2. "Answer: X" パターン
    3. 選択肢先頭の "X." パターン
    """

    def test_the_answer_is_with_period(self) -> None:
        """'The answer is X.' パターンのテスト."""
        assert extract_answer_mmlu("The answer is D.") == "D"
        assert extract_answer_mmlu("The answer is A.") == "A"

    def test_the_answer_is_without_period(self) -> None:
        """'The answer is X' パターン（ピリオドなし）のテスト."""
        assert extract_answer_mmlu("The answer is A") == "A"
        assert extract_answer_mmlu("The answer is B") == "B"

    def test_the_answer_is_with_parentheses(self) -> None:
        """'The answer is (X)' パターンのテスト."""
        assert extract_answer_mmlu("The answer is (A).") == "A"
        assert extract_answer_mmlu("The answer is (B)") == "B"

    def test_answer_colon_pattern(self) -> None:
        """'Answer: X' パターンのテスト."""
        assert extract_answer_mmlu("Answer: B") == "B"
        assert extract_answer_mmlu("answer: C") == "C"
        assert extract_answer_mmlu("Answer:A") == "A"

    def test_choice_prefix_pattern(self) -> None:
        """選択肢先頭の 'X.' パターンのテスト.

        モデルが選択肢の内容を直接引用するケース.
        """
        # 問題のケース: assistant\n\nB. is attached to...
        text = "assistant\n\nB. is attached to the condylar process and elevates the mandible."
        assert extract_answer_mmlu(text) == "B"

        # 他のケース
        assert extract_answer_mmlu("assistant\n\nC. pulmonary artery to the aorta.") == "C"
        assert (
            extract_answer_mmlu(
                "\nA. deposit bone and differentiate from periosteal mesenchymal cells."
            )
            == "A"
        )
        assert extract_answer_mmlu("D. resorb bone and differentiate from monocytes.") == "D"

    def test_single_letter_pattern(self) -> None:
        """単独の選択肢文字のテスト."""
        assert extract_answer_mmlu("(A)") == "A"
        assert extract_answer_mmlu("B") == "B"
        assert extract_answer_mmlu("\nC\n") == "C"

    def test_case_insensitive(self) -> None:
        """大文字小文字を区別しないテスト."""
        assert extract_answer_mmlu("The answer is a.") == "A"
        assert extract_answer_mmlu("the answer is b") == "B"
        assert extract_answer_mmlu("answer: c") == "C"

    def test_invalid_response(self) -> None:
        """無効な回答のテスト."""
        assert extract_answer_mmlu("I don't know the answer.") == "[invalid]"
        assert extract_answer_mmlu("The lateral pterygoid muscle is important.") == "[invalid]"
        assert extract_answer_mmlu("") == "[invalid]"


class TestExtractAnswerBBH:
    """BBH回答抽出のテスト."""

    def test_standard_pattern(self) -> None:
        """標準的なパターンのテスト."""
        assert extract_answer_bbh("So the answer is True.") == "True"
        assert extract_answer_bbh("So the answer is False.") == "False"
        assert extract_answer_bbh("the answer is yes.") == "yes"

    def test_invalid_response(self) -> None:
        """無効な回答のテスト."""
        assert extract_answer_bbh("I'm not sure about this.") == "[invalid]"
        assert extract_answer_bbh("") == "[invalid]"


class TestExtractNumberFromText:
    """GSM8K数値抽出のテスト."""

    def test_the_answer_is_pattern(self) -> None:
        """'The answer is X.' パターンのテスト."""
        assert extract_number_from_text("The answer is 42.") == "42"
        assert extract_number_from_text("The answer is 3.14.") == "3.14"
        assert extract_number_from_text("The answer is $100.") == "100"

    def test_with_commas(self) -> None:
        """カンマ区切り数値のテスト."""
        assert extract_number_from_text("The answer is 1,234.") == "1234"
        assert extract_number_from_text("The answer is 1,234,567.") == "1234567"

    def test_flexible_extract(self) -> None:
        """フレキシブル抽出のテスト."""
        assert extract_number_from_text("After calculating, I got 123") == "123"
        assert extract_number_from_text("So the total is $45.50") == "45.50"

    def test_no_number(self) -> None:
        """数値がない場合のテスト."""
        assert extract_number_from_text("I don't know") is None
        assert extract_number_from_text("") is None


class TestNormalizeChoiceAnswer:
    """多肢選択回答の正規化テスト."""

    def test_parentheses(self) -> None:
        """括弧付きのテスト."""
        assert normalize_choice_answer("(A)") == "A"
        assert normalize_choice_answer("(B)") == "B"

    def test_single_letter(self) -> None:
        """単一文字のテスト."""
        assert normalize_choice_answer("A") == "A"
        assert normalize_choice_answer("b") == "B"

    def test_with_whitespace(self) -> None:
        """空白付きのテスト."""
        assert normalize_choice_answer("  A  ") == "A"
        assert normalize_choice_answer(" (B) ") == "B"


class TestNormalizeGsm8kAnswer:
    """GSM8K回答正規化のテスト（lm-eval-harness公式方式）.

    公式設定:
    - regexes_to_ignore: ["#### ", ",", "\\$", "\\."]
    - ignore_case: true
    """

    def test_remove_hash_prefix(self) -> None:
        """#### プレフィックス除去のテスト."""
        assert normalize_gsm8k_answer("#### 42") == "42"
        assert normalize_gsm8k_answer("#### 1234") == "1234"

    def test_remove_comma(self) -> None:
        """カンマ除去のテスト."""
        assert normalize_gsm8k_answer("1,234") == "1234"
        assert normalize_gsm8k_answer("1,234,567") == "1234567"

    def test_remove_dollar(self) -> None:
        """$除去のテスト."""
        assert normalize_gsm8k_answer("$100") == "100"
        assert normalize_gsm8k_answer("$1,234") == "1234"

    def test_remove_trailing_period(self) -> None:
        """末尾ピリオド除去のテスト."""
        assert normalize_gsm8k_answer("42.") == "42"
        assert normalize_gsm8k_answer("3.14.") == "3.14"

    def test_lowercase(self) -> None:
        """小文字化のテスト."""
        assert normalize_gsm8k_answer("ABC") == "abc"
        assert normalize_gsm8k_answer("Hello") == "hello"

    def test_combined(self) -> None:
        """複合的なテスト."""
        assert normalize_gsm8k_answer("#### $1,234.") == "1234"
        assert normalize_gsm8k_answer("#### 42.") == "42"

    def test_none_input(self) -> None:
        """None入力のテスト."""
        assert normalize_gsm8k_answer(None) == ""

    def test_exact_match_comparison(self) -> None:
        """exact_match比較のテスト（公式方式）."""
        # 同じ値は一致
        assert normalize_gsm8k_answer("42") == normalize_gsm8k_answer("42")
        assert normalize_gsm8k_answer("#### 42") == normalize_gsm8k_answer("42")
        assert normalize_gsm8k_answer("$42") == normalize_gsm8k_answer("42")
        assert normalize_gsm8k_answer("42.") == normalize_gsm8k_answer("42")

        # 異なる値は不一致
        assert normalize_gsm8k_answer("42") != normalize_gsm8k_answer("43")
        assert normalize_gsm8k_answer("42.0") != normalize_gsm8k_answer("42")
