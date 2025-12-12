import json
import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer, 
    InputExample, 
    losses, 
    models, 
    evaluation
)
from transformers import HfArgumentParser
from .arguments import DenseTrainArguments 

# logger 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    # 1. Arguments 파싱
    parser = HfArgumentParser((DenseTrainArguments,))
    args, = parser.parse_args_into_dataclasses()

    logger.info(f"Arguments: {args}")

    # 2. 모델 로드 (SentenceTransformer로 래핑)
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path)
    model.max_seq_length = args.max_seq_length

    # 3. 데이터셋 로드 및 변환
    logger.info(f"Loading training data from {args.train_data_path}...")
    with open(args.train_data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # BGE 모델용 쿼리 접두어 (Instruction)
    query_instruction = "Represent this sentence for searching relevant passages: "
    
    train_samples = []
    for item in train_data:
        query = item['query']
        positive = item['positive']
        negatives = item['negatives']

        # 쿼리 앞에 접두어 붙이기
        if args.use_instruction:
            query = query_instruction + query

        # [Query, Positive, Hard_Negative_1, Hard_Negative_2, ...]
        texts = [query, positive] + negatives[:2] # 2개까지만 사용
        train_samples.append(InputExample(texts=texts))

    logger.info(f"Total training samples: {len(train_samples)}")

    # 4. DataLoader 설정
    train_dataloader = DataLoader(
        train_samples, 
        shuffle=True, 
        batch_size=args.batch_size
    )

    # 5. Loss Function 설정 (MultipleNegativesRankingLoss)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 6. 학습 설정 (Warmup 등)
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1) # 10% warmup

    # 7. 학습 시작
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.learning_rate},
        output_path=args.output_dir,
        show_progress_bar=True,
        save_best_model=True,
    )
    
    logger.info(f"Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()