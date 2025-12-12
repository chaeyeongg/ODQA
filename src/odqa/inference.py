"""
ODQA inference 코드
odqa_pipeline.py에서
1) sparse retrieval + reader 평가(ODQA) 또는
2) retrieval 없이 순수 MRC 평가
를 수행합니다.
"""

import logging
import sys

from datasets import load_from_disk
from transformers import HfArgumentParser, TrainingArguments, set_seed

from .arguments import DataTrainingArguments, ModelArguments, RetrievalArguments
from .odqa_pipeline import ODQAPipeline

logger = logging.getLogger(__name__)


def main() -> None:

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, RetrievalArguments, TrainingArguments)
    )
    model_args, data_args, retrieval_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)

    # seed 고정 (default : 42)
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    pipeline = ODQAPipeline(
        model_args=model_args,
        data_args=data_args,
        retrieval_args=retrieval_args,
        training_args=training_args,
    )

    # ODQA 평가 또는 MRC-only 평가
    if training_args.do_eval or training_args.do_predict:
        pipeline.evaluate_odqa(datasets)

if __name__ == "__main__":
    main()
