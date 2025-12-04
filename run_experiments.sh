#!/bin/bash
# ODQA 실험 실행 스크립트

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ODQA 실험 스크립트 ===${NC}"

# 기본 경로 설정
DATA_DIR="../data"
OUTPUT_DIR="./outputs"
MODEL_NAME="klue/bert-base"

# ====================================================
# 1. MRC 학습 (Train)
# ====================================================
echo -e "\n${YELLOW}[1/4] MRC 학습 시작...${NC}"
python baseline/train.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATA_DIR}/train_dataset \
    --output_dir ${OUTPUT_DIR}/train_result \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 100 \
    --overwrite_output_dir

if [ $? -ne 0 ]; then
    echo -e "${RED}MRC 학습 실패!${NC}"
    exit 1
fi
echo -e "${GREEN}MRC 학습 완료!${NC}"


# ====================================================
# 2. 순수 MRC 평가 (Validation - Retrieval 제외)
# ====================================================
echo -e "\n${YELLOW}[2/4] 순수 MRC 평가 시작...${NC}"
# 학습된 모델 경로는 ${OUTPUT_DIR}/train_result 입니다.
python baseline/inference.py \
    --model_name_or_path ${OUTPUT_DIR}/train_result \
    --dataset_name ${DATA_DIR}/train_dataset \
    --output_dir ${OUTPUT_DIR}/eval_mrc \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --eval_retrieval False

if [ $? -ne 0 ]; then
    echo -e "${RED}순수 MRC 평가 실패!${NC}"
    exit 1
fi
echo -e "${GREEN}순수 MRC 평가 완료!${NC}"


# ====================================================
# 3. ODQA 평가 (Validation - Retrieval 포함)
# ====================================================
echo -e "\n${YELLOW}[3/4] ODQA 평가 시작 (Retrieval + MRC)...${NC}"
python baseline/inference.py \
    --model_name_or_path ${OUTPUT_DIR}/train_result \
    --dataset_name ${DATA_DIR}/train_dataset \
    --output_dir ${OUTPUT_DIR}/eval_odqa \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --data_path ${DATA_DIR} \
    --context_path wikipedia_documents.json \
    --retrieval_type sparse \
    --top_k 10 \
    --eval_retrieval True

if [ $? -ne 0 ]; then
    echo -e "${RED}ODQA 평가 실패!${NC}"
    exit 1
fi
echo -e "${GREEN}ODQA 평가 완료!${NC}"


# ====================================================
# 4. 최종 테스트 추론 (Test Inference)
# ====================================================
echo -e "\n${YELLOW}[4/4] 최종 테스트 데이터 추론 시작...${NC}"

# 변수 사용 및 에러 처리 추가
python baseline/inference.py \
    --output_dir ${OUTPUT_DIR}/test_predict \
    --model_name_or_path ${OUTPUT_DIR}/train_result \
    --dataset_name ${DATA_DIR}/test_dataset \
    --do_predict \
    --retrieval_type sparse \
    --top_k 10 \
    --data_path ${DATA_DIR} \
    --context_path wikipedia_documents.json

if [ $? -ne 0 ]; then
    echo -e "${RED}최종 추론 실패!${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== 모든 실험 완료! ===${NC}"
echo -e "최종 결과 파일: ${OUTPUT_DIR}/test_predict/predictions.json"