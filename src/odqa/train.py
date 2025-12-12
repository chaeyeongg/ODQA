import logging
import sys

from datasets import load_from_disk
from transformers import HfArgumentParser, TrainingArguments, set_seed

from .arguments import DataTrainingArguments, ModelArguments, RetrievalArguments
from .odqa_pipeline import ODQAPipeline


logger = logging.getLogger(__name__)


def main() -> None:
    """
    MRC 학습 / 평가 엔트리포인트.
    - 모델/데이터/리트리벌/트레이닝 인자를 파싱한 뒤
    - ODQAPipeline(train_mrc)을 호출하여 학습/평가를 수행합니다.
    """
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, RetrievalArguments, TrainingArguments)
    )
    model_args, data_args, retrieval_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # seed 고정
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    pipeline = ODQAPipeline(
        model_args=model_args,
        data_args=data_args,
        retrieval_args=retrieval_args,
        training_args=training_args,
    )

    if training_args.do_train or training_args.do_eval:
        pipeline.train_mrc(datasets)


if __name__ == "__main__":
    main()
