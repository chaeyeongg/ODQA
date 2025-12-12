# odqa_pipeline.py

from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import os

from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    EarlyStoppingCallback,
)

from .arguments import (
    ModelArguments,
    DataTrainingArguments,
    RetrievalArguments,
)
from .retrieval import (
  SparseRetrieval,
  DenseRetrieval,
  BM25Retrieval,
  HybridRetrieval,
  RerankRetrieval
)

from .trainer_qa import QuestionAnsweringTrainer

from .utils_qa import check_no_error, postprocess_qa_predictions

from mecab import MeCab


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# 2. ODQA 파이프라인 클래스
#    - retriever + reader(MRC) + postprocess를 하나로 캡슐화
# --------------------------------------------------------------------
class ODQAPipeline:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        retrieval_args: RetrievalArguments,
        training_args: TrainingArguments,
    ) -> None:
        # 1) 모델/토크나이저 로딩
        self.model_args = model_args
        self.data_args = data_args
        self.retrieval_args = retrieval_args
        self.training_args = training_args

        self.config = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name or model_args.model_name_or_path,
            use_fast=True,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.config,
        )

        self.mecab = MeCab()

        def mecab_tokenizer(text):
            return self.mecab.morphs(text) # case 1) 형태소 단위로 분리
            # return self.mecab.nouns(text) # case 2) 명사만 분리

        # 2) 리트리버 준비 (sparse / dense 선택) - eval_retrieval이 True일 때만 초기화
        if data_args.eval_retrieval:
            if retrieval_args.retrieval_type == "dense":
                dense_model_name = (
                    retrieval_args.dense_model_name_or_path
                )
                self.retriever = DenseRetrieval(
                    model_name_or_path=dense_model_name,
                    data_path=retrieval_args.data_path,
                    context_path=retrieval_args.context_path,
                )
            elif retrieval_args.retrieval_type == 'sparse':
                # 기본값: TF-IDF sparse retrieval
                self.retriever = SparseRetrieval(
                    # tokenize_fn=self.tokenizer.tokenize, # 기본 토크나이저 (모델 tokenizer)
                    tokenize_fn=mecab_tokenizer, # MeCab 토크나이저
                    data_path=retrieval_args.data_path,
                    context_path=retrieval_args.context_path,
                )
            elif retrieval_args.retrieval_type == 'hybrid':
                logger.info("Initializing Hybrid Retriever (BM25 + Dense)...")
                
                # 1. BM25 초기화
                bm25 = BM25Retrieval(
                    # tokenize_fn=self.tokenizer.tokenize,
                    tokenize_fn=mecab_tokenizer,
                    data_path=retrieval_args.data_path,
                    context_path=retrieval_args.context_path,
                )
                bm25.get_sparse_embedding() # 인덱스 빌드
                
                # 2. Dense 초기화
                dense = DenseRetrieval(
                    model_name_or_path=retrieval_args.dense_model_name_or_path,
                    data_path=retrieval_args.data_path,
                    context_path=retrieval_args.context_path,
                )
                dense.build_index() # 인덱스 빌드 (캐시 로드)
                
                # 3. Hybrid로 결합
                self.retriever = HybridRetrieval(sparse_retriever=bm25, dense_retriever=dense)


            else: #bm25
                self.retriever = BM25Retrieval(
                    # tokenize_fn=self.tokenizer.tokenize,
                    tokenize_fn=mecab_tokenizer,
                    data_path=retrieval_args.data_path,
                    context_path=retrieval_args.context_path,
                )
        else:
            self.retriever = None

        # Reranker 적용
        if retrieval_args.use_reranker:
            logger.info(f"Applying Reranker: {retrieval_args.reranker_model_name}")
            self.retriever = RerankRetrieval(
                base_retriever=self.retriever, #  Retriever Base로 넣음
                model_name_or_path=retrieval_args.reranker_model_name
            )

        # 3) Trainer (reader)는 필요 시점에 구성
        self.trainer: Optional[QuestionAnsweringTrainer] = None

        # check_no_error 결과를 저장해 재사용
        self.last_checkpoint: Optional[str] = None
        self.max_seq_length: Optional[int] = None

    # ----------------------------------------------------------------
    # 3. Dataset 준비 (MRC 학습/평가용)
    # ----------------------------------------------------------------
    def prepare_mrc_datasets(
        self,
        datasets: DatasetDict,
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """
        기존 train.py / inference.py 의 run_mrc 전처리 공통 부분을 모아온 메서드.
        - train_dataset: feature 가 포함된 학습용 Dataset (또는 None)
        - eval_dataset:  feature 가 포함된 평가용 Dataset (또는 None)
        - eval_examples: 전처리 되지 않은 원본 평가용 Dataset (또는 None)
        """
        # train / eval 공통 컬럼 설정
        if self.training_args.do_train and "train" in datasets:
            column_names = datasets["train"].column_names
        else:
            column_names = datasets["validation"].column_names

        question_column_name = (
            "question" if "question" in column_names else column_names[0]
        )
        context_column_name = (
            "context" if "context" in column_names else column_names[1]
        )
        answer_column_name = (
            "answers" if "answers" in column_names else column_names[2]
        )

        # Padding 방향 설정
        pad_on_right = self.tokenizer.padding_side == "right"

        # 오류 및 max_seq_length 확인 (한 번만 수행해서 저장)
        self.last_checkpoint, self.max_seq_length = check_no_error(
            self.data_args, self.training_args, datasets, self.tokenizer
        )
        max_seq_length = self.max_seq_length

        # -------------------------------
        # Train 전처리
        # -------------------------------
        def prepare_train_features(examples):
            tokenized_examples = self.tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=self.data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized_examples.pop("offset_mapping")

            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(self.tokenizer.cls_token_id)

                sequence_ids = tokenized_examples.sequence_ids(i)

                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]

                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    token_start_index = 0
                    while sequence_ids[token_start_index] != (
                        1 if pad_on_right else 0
                    ):
                        token_start_index += 1

                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (
                        1 if pad_on_right else 0
                    ):
                        token_end_index -= 1

                    if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                    ):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        while (
                            token_start_index < len(offsets)
                            and offsets[token_start_index][0] <= start_char
                        ):
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(
                            token_start_index - 1
                        )
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)

            return tokenized_examples

        train_dataset: Optional[Dataset] = None
        if self.training_args.do_train and "train" in datasets:
            train_dataset = datasets["train"]
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        # -------------------------------
        # Validation / Inference 전처리
        # -------------------------------
        def prepare_validation_features(examples):
            tokenized_examples = self.tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=self.data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )

            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(
                    examples["id"][sample_index]
                )

                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]
            return tokenized_examples

        eval_dataset: Optional[Dataset] = None
        eval_examples: Optional[Dataset] = None
        if self.training_args.do_eval or self.training_args.do_predict:
            eval_examples = datasets["validation"]
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        return train_dataset, eval_dataset, eval_examples

    # ----------------------------------------------------------------
    # 4. Trainer 생성
    # ----------------------------------------------------------------
    def create_trainer(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        eval_examples: Optional[Dataset],
    ) -> QuestionAnsweringTrainer:
        """
        QuestionAnsweringTrainer를 생성하고, post-processing / metric을 설정.
        """
        from transformers import DataCollatorWithPadding, EvalPrediction
        import evaluate as hf_evaluate

        data_collator = DataCollatorWithPadding(
            self.tokenizer,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        squad_metric = hf_evaluate.load("squad")

        def post_processing_function(examples, features, predictions, training_args):
            return postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=self.data_args.max_answer_length,
                output_dir=training_args.output_dir,
            )

        def compute_metrics(p) -> Dict:
            # p는 post_processing_function이 반환한 OrderedDict입니다
            # EvalPrediction 객체가 아닙니다
            if isinstance(p, dict):
                # OrderedDict를 SQuAD 형식으로 변환
                formatted_predictions = [
                    {"id": k, "prediction_text": v} for k, v in p.items()
                ]
                # eval_examples에서 정답지 가져오기
                references = []
                for ex in eval_examples:
                    answers = ex.get("answers")
                    # answers가 None이거나 비어있는 경우 (test dataset)
                    if answers is None:
                        # SQuAD 형식의 빈 답변으로 설정
                        references.append({
                            "id": ex["id"], 
                            "answers": {"text": [], "answer_start": []}
                        })
                    else:
                        references.append({
                            "id": ex["id"], 
                            "answers": answers
                        })
                
                # 모든 references가 빈 답변인 경우 (test dataset) metric 계산 건너뜀
                if all(ref["answers"]["text"] == [] for ref in references):
                    logger.warning("정답이 없는 test dataset입니다. Metric 계산을 건너뜁니다.")
                    return {}
                
                return squad_metric.compute(
                    predictions=formatted_predictions, 
                    references=references
                )
            else:
                # EvalPrediction 객체인 경우 (기존 방식)
                return squad_metric.compute(
                    predictions=p.predictions, 
                    references=p.label_ids
                )
        
        self.trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics, 
            post_process_function=post_processing_function,           
        )

        return self.trainer

    # ----------------------------------------------------------------
    # 5. ODQA용 Retrieval
    # ----------------------------------------------------------------
    def retrieve_for_odqa(
        self,
        queries: Union[str, List[str], Dataset],
    ) -> Any:
        """
        ODQA용 retrieval 인터페이스.
        - str 한 개 → top-k passage & scores 반환
        - HF Dataset → question / context가 붙은 테이블 반환
        """
        # retriever가 None이면 retrieval 생략 (eval_retrieval=False 경우)
        if self.retriever is None:
            if isinstance(queries, str):
                # 단일 질문인 경우 빈 결과 반환
                return ([], [])
            elif isinstance(queries, Dataset):
                # Dataset인 경우 그대로 반환 (retrieval 없이)
                return queries

        # retriever가 공통으로 제공하는 build_index / retrieve 사용
        # (SparseRetrieval은 내부적으로 TF-IDF, DenseRetrieval은 transformer encoder 사용)
        if getattr(self.retriever, "p_embedding", None) is None:
            self.retriever.build_index()
        return self.retriever.retrieve(queries, topk=self.retrieval_args.top_k)

    # ----------------------------------------------------------------
    # 6. ODQA End-to-End 인퍼런스 (질문 → 답변 텍스트)
    # ----------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        단일 질문에 대해:
        1) top-k passage retrieval
        2) reader 모델로 span 예측
        3) 후처리된 최종 답변 반환
        """
        top_k = top_k or self.retrieval_args.top_k

        # 1) 리트리버로 상위 passage들 가져오기
        doc_scores, passages = self.retrieve_for_odqa(question)  # (scores, [contexts])

        # 2) HF Dataset 형태로 변환 (id/question/context)
        from datasets import Dataset as HFDataset

        qa_dataset = HFDataset.from_dict(
            {
                "id": [f"q-0-{i}" for i in range(len(passages))],
                "question": [question] * len(passages),
                "context": passages,
            }
        )
        datasets = DatasetDict({"validation": qa_dataset})

        # 3) MRC용 전처리 & Trainer 생성
        train_ds, eval_ds, eval_examples = self.prepare_mrc_datasets(datasets)
        trainer = self.create_trainer(
            train_dataset=None,
            eval_dataset=eval_ds,
            eval_examples=eval_examples,
        )

        # 4) 예측 수행 (trainer.predict → postprocess_qa_predictions 적용됨)
        predictions = trainer.predict(
            test_dataset=eval_ds, test_examples=eval_examples
        )
        # predictions 는 post_processing_function의 반환값 형식

        # 5) 단일 질문에 대한 최상위 답변만 추려서 반환 형식 통일
        #    (기본 predictions는 {id: text} dict 리스트/구조 등일 수 있음)
        return {
            "question": question,
            "answers": predictions,  # 후속 단계에서 필요한 형태에 맞게 가공 가능
            "passages": passages,
            "scores": doc_scores,
        }

    # ----------------------------------------------------------------
    # 7. MRC 학습 / ODQA 평가 (선택)
    # ----------------------------------------------------------------
    def train_mrc(self, datasets: DatasetDict) -> None:
        """
        MRC 학습 전용.
        - 기존 train.py의 run_mrc(do_train 부분)를 이 메서드로 옮겨오는 형태.
        """
        train_dataset, eval_dataset, eval_examples = self.prepare_mrc_datasets(
            datasets
        )

        self.create_trainer(
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            eval_examples=eval_examples if self.training_args.do_eval else None,
        )

        # Training
        if self.training_args.do_train and train_dataset is not None:
            if self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            elif os.path.isdir(self.model_args.model_name_or_path):
                checkpoint = self.model_args.model_name_or_path
            else:
                checkpoint = None

            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            self.trainer.save_model()

            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

            output_train_file = os.path.join(
                self.training_args.output_dir, "train_results.txt"
            )
            with open(output_train_file, "w", encoding="utf-8") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info("  %s = %s", key, value)
                    writer.write(f"{key} = {value}\n")

        # Evaluation
        if self.training_args.do_eval and eval_dataset is not None:
            logger.info("*** Evaluate (MRC) ***")
            metrics = self.trainer.evaluate()
            metrics["eval_samples"] = len(eval_dataset)

            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

    def evaluate_odqa(self, datasets: DatasetDict) -> Dict[str, float]:
        """
        ODQA 평가용 (질문 + 정답 + wiki context 있는 validation 셋 기준).
        - 각 질문에 대해 retrieve → reader → metric 계산까지.
        """
        from datasets import Dataset as HFDataset, Features, Sequence, Value

        # 1) Sparse retrieval 실행 (validation 질문 기준)
        if not self.data_args.eval_retrieval:
            # retrieval 없이 순수 MRC 평가만 수행
            logger.info("eval_retrieval=False, retrieval 없이 MRC 평가만 진행합니다.")
            train_dataset, eval_dataset, eval_examples = self.prepare_mrc_datasets(
                datasets
            )
            self.create_trainer(
                train_dataset=None,
                eval_dataset=eval_dataset,
                eval_examples=eval_examples,
            )
            logger.info("*** Evaluate (MRC-only) ***")
            metrics = self.trainer.evaluate()
            metrics["eval_samples"] = len(eval_dataset) if eval_dataset is not None else 0
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)
            return metrics

        logger.info("Running retrieval for ODQA evaluation...")
        
        # 인덱스 생성 후 validation 질문 기준으로 검색 수행
        # (HybridRetrieval인 경우 build_index 메서드를 호출하도록 구현되어 있어야 함)
        if getattr(self.retriever, "build_index", None):
             self.retriever.build_index()
        elif getattr(self.retriever, "p_embedding", None) is None:
             # Sparse/Dense 등 개별 리트리버의 경우
             # (단, SparseRetrieval의 경우 get_sparse_embedding 이름일 수 있음)
             if hasattr(self.retriever, "get_sparse_embedding"):
                 self.retriever.get_sparse_embedding()
             elif hasattr(self.retriever, "build_index"):
                 self.retriever.build_index()

        # =========================================================
        # Hybrid Retrieval일 경우 alpha 값 전달
        # =========================================================
        if self.retrieval_args.retrieval_type == "hybrid":
            logger.info(f"Using Hybrid Retrieval with alpha={self.retrieval_args.alpha}")
            df = self.retriever.retrieve(
                datasets["validation"], 
                topk=self.retrieval_args.top_k,
                alpha=self.retrieval_args.alpha  # <--- alpha 인자 추가
            )
        else:
            df = self.retriever.retrieve(
                datasets["validation"], 
                topk=self.retrieval_args.top_k
            )
        # =========================================================

        
        if "original_context" in df.columns:
            df = df.drop(columns=["original_context"])

        # 2) DataFrame → HF Dataset 변환 (정답 포함)
        features = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        odqa_dataset = HFDataset.from_pandas(df, features=features)
        odqa_datasets = DatasetDict({"validation": odqa_dataset})

        # 3) MRC 전처리 및 평가
        train_dataset, eval_dataset, eval_examples = self.prepare_mrc_datasets(
            odqa_datasets
        )
        self.create_trainer(
            train_dataset=None,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
        )

        # 정답이 있는지 확인 (test dataset은 answers가 없음)
        has_answers = (
            eval_examples is not None 
            and "answers" in eval_examples.column_names
            and any(
                ex.get("answers") is not None 
                and isinstance(ex.get("answers"), dict)
                and len(ex.get("answers", {}).get("text", [])) > 0
                for ex in eval_examples
            )
        )
        
        if not has_answers:
            # 정답이 없는 경우 (do_predict) - predict만 수행
            logger.info("*** Predict (ODQA) - 정답이 없어 metric 계산을 건너뜁니다 ***")
            predictions = self.trainer.predict(
                test_dataset=eval_dataset,
                test_examples=eval_examples,
            )
            logger.info("*** Predict 완료 - predictions.json이 저장되었습니다 ***")
            return {"status": "predictions_saved", "num_samples": len(eval_dataset) if eval_dataset is not None else 0}
        
        # 정답이 있는 경우 (do_eval) - evaluate 수행
        logger.info("*** Evaluate (ODQA) ***")
        metrics = self.trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset) if eval_dataset is not None else 0
        self.trainer.log_metrics("test", metrics)
        self.trainer.save_metrics("test", metrics)
        return metrics