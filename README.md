# 📚 ODQA: Open-Domain Question Answering Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Transformers](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/docs/transformers/index)

**ODQA(Open-Domain Question Answering)**는 개방형 도메인 질문 답변 시스템으로, 검색(Retrieval)과 기계 독해(Machine Reading Comprehension)를 결합한 파이프라인입니다.

## ✨ Features

- 🔍 **다양한 검색 방식**: Sparse(TF-IDF), Dense, BM25, Hybrid 검색 지원
- 📖 **고성능 MRC 모델**: KLUE-RoBERTa 기반 Reader 모델
- 🔄 **유연한 파이프라인**: 모듈별 독립 실행 및 결합 가능
- 🎯 **쉬운 실험**: Jupyter notebook 및 스크립트 기반 실행
- 📊 **WandB 통합**: 하이퍼파라미터 튜닝 및 실험 추적
- 🏗️ **모듈화 설계**: 재사용 가능한 컴포넌트 구조

## 🚀 Quick Start

### 설치

```bash
# Clone repository
git clone https://github.com/your-username/odqa-pipeline.git
cd odqa-pipeline

# Install dependencies
pip install -r requirements.txt

# 또는 개발 모드로 설치
pip install -e .
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

# 개별 모듈 실행
python src/odqa/train.py --help
python src/odqa/inference.py --help
```

## 📂 Project Structure

```
odqa-pipeline/
├── src/odqa/                    # ODQA 패키지
│   ├── __init__.py             # 패키지 초기화
│   ├── odqa_pipeline.py        # 메인 ODQA 파이프라인 클래스
│   ├── retrieval.py            # 검색 모듈 (Sparse, Dense, BM25, Hybrid)
│   ├── train.py                # MRC 모델 학습 스크립트
│   ├── inference.py            # 모델 평가 및 추론 스크립트
│   ├── trainer_qa.py           # QA 태스크용 커스텀 트레이너
│   ├── utils_qa.py             # 데이터 전처리 유틸리티
│   ├── arguments.py            # 실행 인자 설정
│   ├── mining.py               # 데이터 마이닝 스크립트
│   ├── train_dense.py          # Dense Retriever 학습 스크립트
│   ├── ensemble.py             # 앙상블 예측 스크립트
│   ├── test_pipeline.ipynb     # 파이프라인 테스트 노트북
│   ├── EDA.ipynb              # 데이터 탐색 노트북
│   └── README.md               # 모듈별 설명
├── data/                       # 데이터 파일들 (.gitkeep)
├── outputs/                    # 학습 결과 및 예측 파일 (.gitkeep)
├── requirements.txt            # Python 의존성
├── setup.py                    # 패키지 설정
├── run_experiments.sh          # 전체 실험 실행 스크립트
├── sweep.yaml                  # WandB 하이퍼파라미터 튜닝 설정
├── LICENSE                     # MIT 라이선스
└── README.md                   # 이 파일
```

## 🔧 Configuration

### 주요 모델 및 설정

- **Reader Model**: `klue/roberta-large` (기본)
  - `klue/bert-base`, `monologg/koelectra-base-v3-finetuned-korquad` 등 지원
- **Retrieval Type**: `bm25`, `sparse`, `dense`, `hybrid`
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

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [KLUE](https://klue-benchmark.com/) - Korean Language Understanding Evaluation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [WandB](https://wandb.ai/) - Experiment tracking
- Baseline code from ODQA competition
