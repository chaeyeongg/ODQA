# 📚 ODQA: Open-Domain Question Answering Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Transformers](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/docs/transformers/index)

**ODQA(Open-Domain Question Answering)**는 개방형 도메인 질문 답변 시스템으로, 검색(Retrieval)과 기계 독해(Machine Reading Comprehension)를 결합한 파이프라인입니다.

## ✨ Features

- 🔍 **다양한 검색 방식**: Sparse(TF-IDF), Dense, BM25, Hybrid 검색 지원
- 📖 **고성능 MRC 모델**: KLUE-RoBERTa 기반 Reader 모델
- 🔄 **유연한 파이프라인**: 모듈별 독립 실행 및 결합 가능
- 🎯 **쉬운 실험**: run_experiments.sh 터미널 실행을 통해 간단히 실행 가능
- 📊 **WandB 통합**: 하이퍼파라미터 튜닝 및 실험 추적
- 🏗️ **모듈화 설계**: 재사용 가능한 컴포넌트 구조

## 🚀 Quick Start

### 설치

```bash
# Clone repository
git https://github.com/chaeyeongg/ODQA.git
cd ODQA

# Install dependencies
pip install -r requirements.txt

```

### 데이터 준비

```bash
# 데이터 폴더 생성 및 데이터셋 다운로드
mkdir -p data
# 데이터셋을 data/ 폴더에 배치해주세요
# - train_dataset/
# - test_dataset/
# - wikipedia_documents.json
```

### 실행

```bash
# 전체 파이프라인 실행 (학습 → 평가 → 추론)
./run_experiments.sh

# 하이퍼파라미터 튜닝
./run_experiments.sh tune

# ODQA Eval 실행
python .src/odqa/inference.py \
    --model_name_or_path ./outputs/train_result \
    --dataset_name ./data/train_dataset \
    --output_dir ./outputs/eval_odqa \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --data_path ./data \
    --context_path wikipedia_documents.json \
    --retrieval_type hybrid \
    --use_reranker True \
    --top_k 3 \
    --eval_retrieval True
    
# ODQA Test 실행
python .src/odqa/inference.py \
    --output_dir ./outputs/test_predict \
    --model_name_or_path ./outputs/train_result \
    --dataset_name ./data/test_dataset \
    --do_predict \
    --retrieval_type hybrid \
    --use_reranker True \
    --top_k 3 \
    --data_path ./data \
    --context_path wikipedia_documents.json


# Soft weighted Ensemble 실행
python ./src/odqa/ensemble.py \
    --model_dirs ./data/ensemble_data/nbest_predictions_bigbird.json  ./data/ensemble_data/nbest_predictions_roberta.json ./data/ensemble_data/nbest_predictions_robert_v2.json \
    --weights 0.5 1.5 2.0 \
    --output_dir ./outputs/ensemble_weighted \
    --strategy soft

# Hard weighted Ensemble 실행
python ./src/odqa/ensemble.py \
    --model_dirs ./data/ensemble_data/predictions_bigbird.csv ./data/ensemble_data/predictions_ensemble.csv \
    --weights 0.5 1.5 \
    --output_dir ./outputs/ensemble_hard \
    --strategy hard

```

## 📂 Project Structure

```bash
.
├── src/odqa/               # ODQA 모듈 폴더
│   ├── odqa_pipeline.py    # main class (Retriever + Reranker + Reader)
│   ├── retrieval.py        # Retrieval 모듈
│   ├── train.py            # MRC 모델 학습 스크립트
│   ├── inference.py        # 모델 평가 및 추론 스크립트
│   ├── mining.py           # train_dense.py 실행을 위한 학습 파일 생성용 모듈
│   ├── train_dense.py      # dense model fine-tuning 모듈
│   ├── trainer_qa.py       # QA Task용 Custom Trainer
│   ├── utils_qa.py         # 데이터 전처리 및 후처리 유틸리티
│   ├── ensemble.py         # Predictions 앙상블 모듈
│   ├── README.md           # ODQA 모듈 README
│   └── arguments.py        # Arguments 설정 파일
│
├── data/                   # 데이터 폴더 (대회에서 제공한 datasets를 넣어주세요.)
│   ├── mined_data/         # mining.py 실행 시 자동 생성
│   └── ensemble_data/      # 앙상블 할 predictions 파일을 넣어주세요.
│
├── outputs/                # train.py, inference.py 실행 시 자동 생성
├── README.md               # 이 파일
├── requirements.txt        # ODQA 모듈 실행을 위한 필수 라이브러리
├── sweep.yaml              # 하이퍼파라미터 인자값 설정 스크립트
└── run_experiment.sh       # 전체 실험 자동 실행 스크립트
```

## 🔧 Configuration

### 주요 모델 및 설정

- **Reader Model**: `klue/roberta-large` (기본)
- **Retrieval Type**: `bm25`, `sparse`(TF-IDF), `dense`, `hybrid`
- **Top-K**: 검색할 문서 개수 (기본: 10)

### Arguments 설정

```python
# 모델 설정
model_args = ModelArguments(
    model_name_or_path="klue/roberta-large"
)

# 데이터 설정
data_args = DataTrainingArguments(
    dataset_name="./data/train_dataset",
    context_path="wikipedia_documents.json"
)

# 검색 설정
retrieval_args = RetrievalArguments(
    retrieval_type="hybrid",
    top_k=10,
    use_reranker=True
)
```

## 🎯 Usage Examples

### 1. 파이프라인 직접 사용

```python
from odqa import ODQAPipeline
from odqa import ModelArguments, DataTrainingArguments, RetrievalArguments

# 설정
model_args = ModelArguments(model_name_or_path="klue/roberta-large")
data_args = DataTrainingArguments(dataset_name="./data/train_dataset")
retrieval_args = RetrievalArguments(retrieval_type="hybrid", top_k=3)

# 파이프라인 생성 및 실행
pipeline = ODQAPipeline(model_args, data_args, retrieval_args)
results = pipeline.predict(test_dataset)
```

### 2. 개별 모듈 사용

```python
from odqa.retrieval import BM25Retrieval

# BM25 검색기 초기화
retriever = BM25Retrieval(tokenize_fn=tokenizer.tokenize)
retriever.build_faiss(contexts=context_documents)

# 검색 실행
scores, docs = retriever.retrieve(query, topk=10)
```

## 🧪 Experiments

### 평가 메트릭

- **EM (Exact Match)**: 정답과 완전히 일치하는 비율
- **F1 Score**: 토큰 수준 정밀도와 재현율의 조화 평균

### WandB 하이퍼파라미터 튜닝

```bash
# WandB 로그인 (토큰 필요)
wandb login YOUR_WANDB_TOKEN

# 튜닝 실행
./run_experiments.sh tune
```

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 1.9
- Transformers >= 4.21
- Datasets >= 2.0
- WandB (선택사항)

## 🙏 Acknowledgments

- [KLUE](https://klue-benchmark.com/) - Korean Language Understanding Evaluation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [WandB](https://wandb.ai/) - Experiment tracking
- Baseline code from ODQA competition
