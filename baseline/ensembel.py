import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def main(args):
    # 1. 앙상블할 모델들의 결과 디렉토리 리스트
    model_dirs = args.model_dirs
    
    print(f"Ensembling {len(model_dirs)} models...")

    # 모든 모델의 nbest 예측 결과 로드
    nbest_preds_list = []
    for d in model_dirs:
        nbest_path = os.path.join(d, "nbest_predictions.json")
        if not os.path.exists(nbest_path):
            print(f"Warning: {nbest_path} does not exist. Skipping.")
            continue
            
        print(f"Loading {nbest_path}...")
        with open(nbest_path, "r", encoding="utf-8") as f:
            nbest_preds_list.append(json.load(f))

    if not nbest_preds_list:
        print("No valid nbest_predictions.json files found.")
        return

    # 첫 번째 모델의 키(Question ID)를 기준으로 순회
    keys = list(nbest_preds_list[0].keys())
    
    final_predictions = {}
    
    # 2. Soft Voting 수행
    for key in tqdm(keys, desc="Ensembling"):
        candidate_scores = defaultdict(float)
        
        # 각 모델의 예측 결과 순회
        for model_preds in nbest_preds_list:
            if key not in model_preds:
                continue
                
            # 해당 모델이 제안한 Top-K 후보들
            candidates = model_preds[key]
            
            for candidate in candidates:
                text = candidate["text"].strip()
                prob = candidate["probability"]
                
                # 같은 텍스트에 대한 확률을 누적 합산
                # 만약 특정 모델을 신뢰한다면 여기서 weight를 곱할 수 있음 (prob * weight)
                if text: # 빈 문자열 제외
                    candidate_scores[text] += prob

        # 3. 합산 점수가 가장 높은 답변 선택
        if not candidate_scores:
            best_answer = "" # 후보가 없으면 빈 문자열
        else:
            best_answer = max(candidate_scores, key=candidate_scores.get)
            
        final_predictions[key] = best_answer

    # 4. 결과 저장 (predictions.json)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "predictions.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_predictions, f, indent=4, ensure_ascii=False)
        
    print(f"Ensemble complete! Saved to {output_path}")

    # CSV 형태로도 저장 (리더보드 제출용)
    # id, prediction 형태라고 가정
    # final_predictions 딕셔너리를 DataFrame으로 변환
    # predictions_submit.csv 포맷에 맞게 수정 필요할 수 있음
    output_csv_path = os.path.join(args.output_dir, "predictions_ensemble.csv")
    df = pd.DataFrame(list(final_predictions.items()), columns=['id', 'prediction'])
    df.to_csv(output_csv_path, index=False)
    print(f"Saved CSV to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 앙상블할 폴더 경로들을 띄어쓰기로 구분해서 입력 받음
    parser.add_argument(
        "--model_dirs", 
        nargs="+", 
        required=True, 
        help="List of model output directories containing nbest_predictions.json"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs/ensemble_result", 
        help="Directory to save ensemble result"
    )
    
    args = parser.parse_args()
    main(args)