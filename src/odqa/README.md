# 📚 Open-Domain Question Answering (ODQA) Project

**ODQA(Open-Domain Question Answering) 파이프라인** 입니다.

---

대회 측에서 제공한 baseline에서 구현된 기술은 최대한 반영하되,
모듈을 이해하기 쉽도록 재구성했습니다.

```diff
- data 경로는 아래 프로젝트 구조를 참고해주세요.

  1. **./run_experiment.sh** 을 터미널에서 실행하여, train-eval-inference 전체 과정을 한 번에 실행할 수 있습니다.
  2. **./run_experiment.sh** 의 내용을 변경하여, 쉽게 Arguments를 변경할 수 있습니다.
  3. **./run_experiment.sh tune** 을 터미널에서 실행하여, **./sweep.yaml** 의 내용으로 하이퍼파라미터 튜닝을 진행할 수 있습니다.

---

## 🚀 Retrieval, Reranker, Reader Model

**Sparse Retrieval (TF-IDF, BM25, Dense, Hybrid)**를 통해 관련 문서를 검색하고,**Reranker**에서 한 번 더 문서의 점수를 정교하게 계산하여 추출된 최종 문서를 **Pre-trained Language Model** 기반의 Reader 모델에게 전달합니다.
이후 여러 Reader Model의 predictions를 Hard voting 혹은 Soft voting하여 최종 답변을 결정합니다.

---

## 📂 프로젝트 구조 (Directory Structure)

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
│   └── arguments.py        # Arguments 설정 파일
│
├── data/                   # 데이터 폴더 (대회에서 제공한 datasets를 넣어주세요.)
│   ├── mined_data/         # mining.py 실행 시 자동 생성
│   └── ensemble_data/      # 앙상블 할 predictions 파일을 넣어주세요.
│
├── outputs/                # train.py, inference.py 실행 시 자동 생성
├── README.md               # ODQA 모듈 README
├── requirements.txt        # ODQA 모듈 실행을 위한 필수 라이브러리
├── sweep.yaml              # 하이퍼파라미터 인자값 설정 스크립트
└── run_experiment.sh       # 전체 실험 자동 실행 스크립트
