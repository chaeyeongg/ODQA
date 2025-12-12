"""
train_dense.py 실행을 위한 hard_negatives.json 생성용 코드

BM25로 train dataset의 Question별 문서 후보를 가져와서 GT와 유사하지 않은 문서를 Hard Negative로 저장합니다.
"""

import json
import os
import logging
import sys
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, DatasetDict
from transformers import AutoTokenizer, HfArgumentParser

from .retrieval import BM25Retrieval
from .arguments import MiningArguments

# logger 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    # 1. Arguments 파싱
    parser = HfArgumentParser((MiningArguments,))
    # 명령줄 인자를 dataclass로 변환
    mining_args, = parser.parse_args_into_dataclasses()

    logger.info(f"Mining Arguments: {mining_args}")

    # 2. 데이터셋 로드
    logger.info(f"Loading dataset from {mining_args.dataset_name}...")
    try:
        dataset = load_from_disk(mining_args.dataset_name)
        if "train" not in dataset:
            dataset = DatasetDict({"train": dataset})
    except Exception as e:
        logger.info(f"Load from disk failed ({e}). Trying load_dataset...")
        dataset = load_dataset(mining_args.dataset_name)

    train_dataset = dataset["train"]
    
    # 3. 토크나이저 로드 (BM25 토큰화용)
    tokenizer = AutoTokenizer.from_pretrained(mining_args.model_name_or_path, use_fast=True)

    # 4. BM25 Retriever 초기화 및 빌드
    retriever = BM25Retrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=mining_args.data_path,
        context_path=mining_args.context_path,
    )
    retriever.get_sparse_embedding()

    # 5. Mining 시작
    logger.info("Starting Hard Negative Mining...")
    
    mined_data = []
    search_k = mining_args.top_k_mining  # dataclass 필드 사용

    for example in tqdm(train_dataset, desc="Mining"):
            query = example["question"]
            original_context = example["context"]  # Ground Truth
            
            doc_scores, doc_indices = retriever.get_relevant_doc(query, k=search_k)
            
            # 인덱스(int)를 사용하여 실제 문서 내용(str)을 가져옵니다.
            retrieved_contexts = [retriever.contexts[i] for i in doc_indices]

            hard_negatives = []
            positive_ctx = None

            # Positive vs Hard Negative 분류
            for ctx in retrieved_contexts:
                # 정답 문서와 내용이 겹치면 Positive로 간주
                if original_context in ctx or ctx in original_context:
                    positive_ctx = ctx
                else:
                    # 정답이 아닌데 상위권에 뜬 문서 -> Hard Negative
                    hard_negatives.append(ctx)

            # 데이터 저장
            if positive_ctx and len(hard_negatives) > 0:
                mined_data.append({
                    "query": query,
                    "positive": positive_ctx,
                    "negatives": hard_negatives[:mining_args.num_negatives]
                })


    # 8. 파일 저장
    output_path = os.path.join(mining_args.output_dir, "hard_negatives.json")
    os.makedirs(mining_args.output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mined_data, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Mining complete! Saved {len(mined_data)} samples to {output_path}")

if __name__ == "__main__":
    main()