#!/bin/bash
# ODQA 실험 실행 스크립트

# 사용법: 
# ./run_experiment.sh       -> 기존과 동일하게 일반 학습 및 평가 진행
# ./run_experiment.sh tune  -> WandB Sweep 하이퍼파라미터 튜닝 진행

MODE=$1 # 첫 번째 인자로 모드 설정 (기본값: train)

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== ODQA 실험 스크립트 ===${NC}"

# 기본 경로 설정
DATA_DIR="./data"
OUTPUT_DIR="./outputs"
MODEL_NAME="klue/roberta-large"


# ====================================================
# [MODE: TUNE] 하이퍼파라미터 튜닝 (WandB Sweep)
# ====================================================
if [ "$MODE" == "tune" ]; then
    echo -e "\n${YELLOW}[WandB] 하이퍼파라미터 튜닝을 시작합니다...${NC}"
    
    # 1. WandB 로그인 확인 (생략 가능)
    wandb login "59da02d0bb99f316314a33f87b74cb7f09c95984"
    
    # 2. Sweep 초기화
    if [ ! -f "sweep.yaml" ]; then
        echo -e "${RED}Error: sweep.yaml 파일이 없습니다.${NC}"
        exit 1
    fi

    # 3. Sweep ID 생성 (수정된 로직)
    echo "Creating Sweep..."
    
    # 출력 결과를 파일에 임시 저장하여 분석
    wandb sweep sweep.yaml > sweep_output.log 2>&1
    
    # 로그 파일에서 'wandb agent'로 시작하는 명령어 라인을 찾음
    # 예: "Run sweep agent with: wandb agent entity/project/sweep_id"
    AGENT_COMMAND=$(grep "wandb agent" sweep_output.log | tail -n 1)
    
    # 명령어에서 ID 부분(마지막 단어)만 추출
    SWEEP_ID=$(echo $AGENT_COMMAND | awk '{print $NF}')

    if [ -z "$SWEEP_ID" ] || [ "$SWEEP_ID" == "sweep" ]; then
        echo -e "${RED}Sweep ID 자동 추출 실패!${NC}"
        echo -e "아래 로그를 보고 수동으로 실행해주세요:"
        cat sweep_output.log
        rm sweep_output.log
        exit 1
    fi

    echo -e "${GREEN}Sweep ID Generated: ${SWEEP_ID}${NC}"
    
    # 임시 파일 삭제
    rm sweep_output.log

    echo "Agent를 실행합니다. (Count: 15)"
    
    # 4. Agent 실행
    # SWEEP_ID가 'wandb agent ...' 전체 명령어를 포함할 수도 있으므로 안전하게 처리
    if [[ "$SWEEP_ID" == wandb* ]]; then
        $SWEEP_ID --count 15
    else
        wandb agent $SWEEP_ID --count 15
    fi

    echo -e "${GREEN}튜닝 완료! WandB 대시보드에서 최적의 파라미터를 확인하세요.${NC}"
    exit 0
fi

# ====================================================
# [MODE: DEFAULT] 기존 파이프라인 (학습 -> 평가 -> 추론)
# ====================================================

# 1. MRC 학습 (Train)
echo -e "\n${YELLOW}[1/4] MRC 학습 시작...${NC}"
python baseline/train.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name ${DATA_DIR}/train_dataset \
    --data_path ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}/train_result \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 100 \
    --overwrite_output_dir \
    --overwrite_cache \
    --report_to wandb \
    --fp16 \
    --run_name baseline_run

if [ $? -ne 0 ]; then
    echo -e "${RED}MRC 학습 실패!${NC}"
    exit 1
fi
echo -e "${GREEN}MRC 학습 완료!${NC}"


# 2. 순수 MRC 평가
echo -e "\n${YELLOW}[2/4] 순수 MRC 평가 시작...${NC}"
python baseline/inference.py \
    --model_name_or_path ${OUTPUT_DIR}/train_result \
    --dataset_name ${DATA_DIR}/train_dataset \
    --output_dir ${OUTPUT_DIR}/eval_mrc \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --data_path ${DATA_DIR} \
    --eval_retrieval False

if [ $? -ne 0 ]; then
    echo -e "${RED}순수 MRC 평가 실패!${NC}"
    exit 1
fi
echo -e "${GREEN}순수 MRC 평가 완료!${NC}"


# 3. ODQA 평가
echo -e "\n${YELLOW}[3/4] ODQA 평가 시작 (Retrieval + MRC)...${NC}"
python baseline/inference.py \
    --model_name_or_path ${OUTPUT_DIR}/train_result \
    --dataset_name ${DATA_DIR}/train_dataset \
    --output_dir ${OUTPUT_DIR}/eval_odqa \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --data_path ${DATA_DIR} \
    --context_path wikipedia_documents.json \
    --retrieval_type dense \
    --top_k 10 \
    --eval_retrieval True

if [ $? -ne 0 ]; then
    echo -e "${RED}ODQA 평가 실패!${NC}"
    exit 1
fi
echo -e "${GREEN}ODQA 평가 완료!${NC}"


# 4. 최종 테스트 추론
echo -e "\n${YELLOW}[4/4] 최종 테스트 데이터 추론 시작...${NC}"
python baseline/inference.py \
    --output_dir ${OUTPUT_DIR}/test_predict \
    --model_name_or_path ${OUTPUT_DIR}/train_result \
    --dataset_name ${DATA_DIR}/test_dataset \
    --do_predict \
    --retrieval_type dense \
    --top_k 10 \
    --data_path ${DATA_DIR} \
    --context_path wikipedia_documents.json

if [ $? -ne 0 ]; then
    echo -e "${RED}최종 추론 실패!${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== 모든 실험 완료! ===${NC}"
echo -e "최종 결과 파일: ${OUTPUT_DIR}/test_predict/predictions.json"