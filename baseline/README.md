# 📚 Open-Domain Question Answering (ODQA) Project

**ODQA(Open-Domain Question Answering) 파이프라인** 입니다.

---

대회 측에서 제공한 baseline에서 구현된 기술은 변경하지 않고,
모듈을 이해하기 쉽도록 재구성했습니다.

```diff
- 구현된 코드 상, data 경로는 run_experiment.py 기준 ../data 입니다. 참고해주세요!!

  1. **test_pipeline.ipynb** 에서 셀을 하나씩 실행하며, odqa_pipeline.py의 로직을 이해할 수 있습니다.
  2. **./run_experiment.sh** 을 터미널에서 실행하여, train-eval-inference 전체 과정을 한 번에 실행할 수 있습니다.
  3. **./run_experiment.sh** 의 내용을 변경하여, 쉽게 Arguments를 변경할 수 있습니다.

---

## 🚀 Retrieval, Reader Model

**Sparse Retrieval (TF-IDF)**를 통해 관련 문서를 검색하고,
**Pre-trained Language Model (klue/BERT-base)** 기반의 Reader 모델을 사용하여 정답을 추출합니다.

---

## 📂 프로젝트 구조 (Directory Structure)

```bash
.
├── baseline/               # ODQA 핵심 모듈 폴더
│   ├── test_pipeline.ipynb # 간편하게 pipline을 실험해볼 수 있는 Jupyter notebook
│   ├── odqa_pipeline.py    # ODQA 메인 클래스 (Retriever + Reader 연결)
│   ├── retrieval.py        # Sparse Retrieval (TF-IDF) 모듈
│   ├── train.py            # MRC 모델 학습 스크립트
│   ├── inference.py        # 모델 평가 및 추론 스크립트
│   ├── trainer_qa.py       # QA Task용 Custom Trainer
│   ├── utils_qa.py         # 데이터 전처리 및 후처리 유틸리티
│   └── arguments.py        # 실행 인자(Arguments) 설정 파일
├── outputs/                # 학습 결과 및 예측 파일 저장소 -> 코드 실행 후 자동 생성
└── run_experiment.sh       # 전체 실험 자동 실행 스크립트 (Entry Point)
