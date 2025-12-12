"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
ODQAPipeline 을 사용하여
- (옵션) sparse retrieval + reader 평가(ODQA)
- 또는 retrieval 없이 순수 MRC 평가
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
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

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

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
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
