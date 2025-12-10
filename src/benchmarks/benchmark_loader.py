"""ベンチマークデータローダー.

各種ベンチマーク (GSM8K, BBH, MMLU, llm-jp-eval) のデータをロードする.

使用例:
    # 単一ベンチマークをロード
    data = load_benchmark("gsm8k", split="test")

    # 利用可能なベンチマーク一覧
    benchmarks = get_available_benchmarks("english")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset

from src.perturbation.benchmark_perturbator import OriginalBenchmarkData
from src.utils.logger import logger


@dataclass
class BenchmarkConfig:
    """ベンチマーク設定."""

    name: str
    language: str
    dataset_name: str
    dataset_config: str | None = None
    split: str = "test"
    text_field: str = "question"
    answer_field: str = "answer"
    extra_fields: list[str] = field(default_factory=list)


class BenchmarkLoader(ABC):
    """ベンチマークローダーの基底クラス."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """初期化.

        Args:
            config: ベンチマーク設定
        """
        self.config = config

    @abstractmethod
    def load(self, max_samples: int | None = None) -> OriginalBenchmarkData:
        """ベンチマークデータをロードする.

        Args:
            max_samples: 最大サンプル数 (Noneの場合は全件)

        Returns:
            OriginalBenchmarkData: ベンチマークデータ
        """
        pass

    def _create_example(self, idx: int, item: dict[str, Any]) -> dict[str, Any]:
        """データセットアイテムからサンプルを作成.

        Args:
            idx: サンプルID
            item: データセットアイテム

        Returns:
            サンプル辞書
        """
        example = {
            "id": idx,
            self.config.text_field: item[self.config.text_field],
            self.config.answer_field: str(item[self.config.answer_field]),
        }

        # 追加フィールドをコピー
        for field_name in self.config.extra_fields:
            if field_name in item:
                example[field_name] = item[field_name]

        return example


class HuggingFaceLoader(BenchmarkLoader):
    """Hugging Face Datasets用ローダー."""

    def load(self, max_samples: int | None = None) -> OriginalBenchmarkData:
        """Hugging Faceからベンチマークデータをロード.

        Args:
            max_samples: 最大サンプル数

        Returns:
            OriginalBenchmarkData
        """
        logger.info(
            f"ベンチマークをロード: {self.config.name} "
            f"(dataset={self.config.dataset_name}, config={self.config.dataset_config})"
        )

        # データセットをロード
        if self.config.dataset_config:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.split,
            )
        else:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.split,
            )

        # サンプル数を制限
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        # サンプルを作成
        examples = []
        for idx, item in enumerate(dataset):
            example = self._create_example(idx, item)
            examples.append(example)

        logger.info(f"ロード完了: {len(examples)}件")

        return OriginalBenchmarkData(
            benchmark_name=self.config.name,
            language=self.config.language,
            text_field=self.config.text_field,
            examples=examples,
        )


class GSM8KLoader(HuggingFaceLoader):
    """GSM8K用ローダー."""

    def _create_example(self, idx: int, item: dict[str, Any]) -> dict[str, Any]:
        """GSM8Kのサンプルを作成.

        GSM8Kの回答は "#### 数値" の形式なので、数値部分を抽出する.
        """
        answer = item["answer"]
        # "#### 数値" から数値部分を抽出
        if "####" in answer:
            answer = answer.split("####")[-1].strip()

        return {
            "id": idx,
            "question": item["question"],
            "answer": answer,
        }


class BBHLoader(BenchmarkLoader):
    """BIG-Bench Hard (BBH)用ローダー.

    BBHは複数のサブタスクで構成されているため、特別な処理が必要.
    """

    # BBHのサブタスク一覧
    SUBTASKS = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]

    def load(self, max_samples: int | None = None) -> OriginalBenchmarkData:
        """BBHの全サブタスクをロード.

        Args:
            max_samples: サブタスクごとの最大サンプル数

        Returns:
            OriginalBenchmarkData
        """
        logger.info(f"BBHをロード: {len(self.SUBTASKS)}サブタスク")

        all_examples = []
        example_id = 0

        for subtask in self.SUBTASKS:
            try:
                dataset = load_dataset(
                    "lukaemon/bbh",
                    subtask,
                    split="test",
                )

                # サンプル数を制限（サブタスクごと）
                if max_samples is not None and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))

                for item in dataset:
                    example = {
                        "id": example_id,
                        "question": item["input"],
                        "answer": item["target"],
                        "subtask": subtask,
                    }
                    all_examples.append(example)
                    example_id += 1

                logger.debug(f"  {subtask}: {len(dataset)}件")

            except Exception as e:
                logger.warning(f"BBHサブタスク '{subtask}' のロードに失敗: {e}")
                continue

        logger.info(f"BBHロード完了: {len(all_examples)}件")

        return OriginalBenchmarkData(
            benchmark_name="bbh",
            language="english",
            text_field="question",
            examples=all_examples,
        )


class MMLULoader(BenchmarkLoader):
    """MMLU用ローダー.

    MMLUは複数の科目で構成されている.
    """

    # MMLUの科目一覧（主要なもの）
    SUBJECTS = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def load(self, max_samples: int | None = None) -> OriginalBenchmarkData:
        """MMLUの全科目をロード.

        Args:
            max_samples: 科目ごとの最大サンプル数

        Returns:
            OriginalBenchmarkData
        """
        logger.info(f"MMLUをロード: {len(self.SUBJECTS)}科目")

        all_examples = []
        example_id = 0

        for subject in self.SUBJECTS:
            try:
                dataset = load_dataset(
                    "cais/mmlu",
                    subject,
                    split="test",
                )

                # サンプル数を制限（科目ごと）
                if max_samples is not None and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))

                for item in dataset:
                    # MMLUの質問形式: question + choices
                    raw_question = item["question"]
                    choices = item["choices"]
                    choices_text = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
                    question = f"{raw_question}\n\n{choices_text}"

                    # 質問文の終了位置を記録（選択肢を除外するため）
                    question_end_position = len(raw_question)

                    # 正解のインデックスをA/B/C/Dに変換
                    answer = chr(65 + item["answer"])

                    example = {
                        "id": example_id,
                        "question": question,
                        "answer": answer,
                        "subject": subject,
                        "question_end_position": question_end_position,
                    }
                    all_examples.append(example)
                    example_id += 1

                logger.debug(f"  {subject}: {len(dataset)}件")

            except Exception as e:
                logger.warning(f"MMLU科目 '{subject}' のロードに失敗: {e}")
                continue

        logger.info(f"MMLUロード完了: {len(all_examples)}件")

        return OriginalBenchmarkData(
            benchmark_name="mmlu",
            language="english",
            text_field="question",
            examples=all_examples,
            search_end_field="question_end_position",
        )


class JapaneseTaskLoader(BenchmarkLoader):
    """日本語タスク用ローダー (zenless-lab/llm-jp-eval使用).

    対象タスク: jamp, jnli, niilc, jsquad, jcommonsenseqa
    """

    DATASET_NAME = "zenless-lab/llm-jp-eval"

    def load(self, max_samples: int | None = None) -> OriginalBenchmarkData:
        """日本語タスクをロード.

        Args:
            max_samples: 最大サンプル数

        Returns:
            OriginalBenchmarkData
        """
        task_name = self.config.dataset_config
        logger.info(f"日本語タスク '{task_name}' をロード (zenless-lab/llm-jp-eval)")

        # データセットをロード
        dataset = load_dataset(
            self.DATASET_NAME,
            task_name,
            split="test",
        )

        # サンプル数を制限
        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        examples = []
        for idx, item in enumerate(dataset):
            # 入力テキストを取得
            input_text = item["input"]
            output_text = item["output"]

            # JCommonsenseQAの場合は選択肢を除外する位置を設定
            # 選択肢は "A: ..., B: ..., C: ..." の形式で含まれる場合がある
            question_end_position = len(input_text)

            # JCommonsenseQAの選択肢パターンを検出
            if task_name == "jcommonsenseqa":
                # 選択肢の開始パターンを検索
                # 例: "質問文 A: 選択肢1 B: 選択肢2 ..."
                import re

                choice_pattern = re.search(r"\s*[A-E][:：]", input_text)
                if choice_pattern:
                    question_end_position = choice_pattern.start()

            example = {
                "id": idx,
                "question": input_text,
                "answer": output_text,
                "task": task_name,
                "question_end_position": question_end_position,
            }
            examples.append(example)

        logger.info(f"日本語タスク '{task_name}' ロード完了: {len(examples)}件")

        return OriginalBenchmarkData(
            benchmark_name=task_name,
            language="japanese",
            text_field="question",
            examples=examples,
            search_end_field="question_end_position",
        )


# ベンチマーク登録
BENCHMARK_REGISTRY: dict[str, type[BenchmarkLoader]] = {
    # 英語
    "gsm8k": GSM8KLoader,
    "bbh": BBHLoader,
    "mmlu": MMLULoader,
    # 日本語（5タスク個別登録）
    "jamp": JapaneseTaskLoader,
    "jnli": JapaneseTaskLoader,
    "niilc": JapaneseTaskLoader,
    "jsquad": JapaneseTaskLoader,
    "jcommonsenseqa": JapaneseTaskLoader,
}

# ベンチマーク設定
BENCHMARK_CONFIGS: dict[str, BenchmarkConfig] = {
    "gsm8k": BenchmarkConfig(
        name="gsm8k",
        language="english",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        split="test",
        text_field="question",
        answer_field="answer",
    ),
    "bbh": BenchmarkConfig(
        name="bbh",
        language="english",
        dataset_name="lukaemon/bbh",
        dataset_config=None,
        split="test",
        text_field="question",
        answer_field="answer",
    ),
    "mmlu": BenchmarkConfig(
        name="mmlu",
        language="english",
        dataset_name="cais/mmlu",
        dataset_config=None,
        split="test",
        text_field="question",
        answer_field="answer",
    ),
    # 日本語タスク（5タスク個別設定）
    "jamp": BenchmarkConfig(
        name="jamp",
        language="japanese",
        dataset_name="zenless-lab/llm-jp-eval",
        dataset_config="jamp",
        split="test",
        text_field="question",
        answer_field="answer",
    ),
    "jnli": BenchmarkConfig(
        name="jnli",
        language="japanese",
        dataset_name="zenless-lab/llm-jp-eval",
        dataset_config="jnli",
        split="test",
        text_field="question",
        answer_field="answer",
    ),
    "niilc": BenchmarkConfig(
        name="niilc",
        language="japanese",
        dataset_name="zenless-lab/llm-jp-eval",
        dataset_config="niilc",
        split="test",
        text_field="question",
        answer_field="answer",
    ),
    "jsquad": BenchmarkConfig(
        name="jsquad",
        language="japanese",
        dataset_name="zenless-lab/llm-jp-eval",
        dataset_config="jsquad",
        split="test",
        text_field="question",
        answer_field="answer",
    ),
    "jcommonsenseqa": BenchmarkConfig(
        name="jcommonsenseqa",
        language="japanese",
        dataset_name="zenless-lab/llm-jp-eval",
        dataset_config="jcommonsenseqa",
        split="test",
        text_field="question",
        answer_field="answer",
    ),
}


def load_benchmark(
    name: str,
    max_samples: int | None = None,
) -> OriginalBenchmarkData:
    """ベンチマークをロードする.

    Args:
        name: ベンチマーク名 (gsm8k, bbh, mmlu, llm_jp_eval)
        max_samples: 最大サンプル数

    Returns:
        OriginalBenchmarkData

    Raises:
        ValueError: 未知のベンチマーク名
    """
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"未知のベンチマーク: {name}. 利用可能: {available}")

    config = BENCHMARK_CONFIGS[name]
    loader_class = BENCHMARK_REGISTRY[name]
    loader = loader_class(config)

    return loader.load(max_samples=max_samples)


def get_available_benchmarks(language: str | None = None) -> list[str]:
    """利用可能なベンチマーク一覧を取得.

    Args:
        language: 言語でフィルタ (None の場合は全て)

    Returns:
        ベンチマーク名のリスト
    """
    if language is None:
        return list(BENCHMARK_REGISTRY.keys())

    return [name for name, config in BENCHMARK_CONFIGS.items() if config.language == language]
