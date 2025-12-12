import json
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

from transformers import HfArgumentParser

from arguments import EnsembleArguments

def custom_read_csv(filepath):
    """
    Pandas read_csv가 실패할 때 사용하는 안전한 CSV 로더.
    """
    preds = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # 헤더 건너뛰기
        if i == 0 and ('id' in line.lower() or 'prediction' in line.lower()):
            continue
            
        # 1. 탭으로 먼저 시도
        parts = line.split('\t', 1)
        if len(parts) < 2:
            # 2. 탭이 없으면 쉼표로 시도
            parts = line.split(',', 1)
        
        if len(parts) >= 2:
            q_id = parts[0].strip()
            text = parts[1].strip().strip('"')
            preds[q_id] = text
            
    return preds

def load_predictions(paths, strategy="hard"):
    """
    파일 경로들을 받아 예측 결과들을 로드
    """
    loaded_data = [] 
    
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: File not found {p}. Skipping.")
            continue
            
        print(f"Loading: {p}")
        
        # 1. JSON 로드
        if p.endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                loaded_data.append(json.load(f))
                
        # 2. CSV 로드
        elif p.endswith(".csv") or p.endswith(".txt"):
            if strategy == "soft":
                print(f"Error: CSV/TXT file ({p}) cannot be used for Soft Voting. Skipping.")
                continue
            
            try:
                # 1차 시도: Pandas
                df = pd.read_csv(p, sep=None, engine='python', on_bad_lines='warn')
                
                if len(df.columns) >= 2:
                    preds = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                else:
                    raise ValueError("Columns < 2")
                
                preds = {k: str(v).strip() if not pd.isna(v) else "" for k, v in preds.items()}
                loaded_data.append(preds)
                
            except Exception as e:
                print(f"Pandas load failed ({e}), trying custom loader...")
                # 2차 시도: 커스텀 로더
                try:
                    preds = custom_read_csv(p)
                    loaded_data.append(preds)
                    print(f"-> Successfully loaded {len(preds)} rows with custom loader.")
                except Exception as e2:
                    print(f"Error reading file {p}: {e2}")

    return loaded_data

def soft_voting(model_dirs, weights):
    """
    Soft Voting: (확률 * 가중치) 합산
    """
    nbest_preds_list = load_predictions(model_dirs, strategy="soft")
    if not nbest_preds_list:
        raise ValueError("No valid nbest files found!")

    # 데이터 개수와 가중치 개수 검증
    if len(nbest_preds_list) != len(weights):
        print(f"Warning: Loaded models ({len(nbest_preds_list)}) != Weights ({len(weights)})")
        # 개수 안 맞으면 앞쪽부터 맞추거나 1.0 처리
        weights = weights[:len(nbest_preds_list)]

    keys = list(nbest_preds_list[0].keys())
    final_predictions = {}
    
    for key in tqdm(keys, desc="Weighted Soft Voting"):
        candidate_scores = defaultdict(float)
        
        # enumerate를 사용해 해당 모델의 weight를 가져옴
        for i, model_preds in enumerate(nbest_preds_list):
            if key not in model_preds: continue
            
            w = weights[i] # 현재 모델의 가중치
            
            for candidate in model_preds[key]:
                text = candidate["text"].strip()
                prob = candidate["probability"]
                if text:
                    # 확률에 가중치를 곱해서 더함
                    candidate_scores[text] += prob * w
                    
        if not candidate_scores:
            best_answer = ""
        else:
            best_answer = max(candidate_scores, key=candidate_scores.get)
        final_predictions[key] = best_answer
        
    return final_predictions

def hard_voting(model_dirs, weights):
    """
    Hard Voting: pandas DataFrame 기반 가중치 적용
    """
    preds_list = load_predictions(model_dirs, strategy="hard")

    if not preds_list:
        raise ValueError("No valid prediction files found!")

    # 데이터 개수와 가중치 개수 검증
    if len(preds_list) != len(weights):
        print(f"Warning: Loaded models ({len(preds_list)}) != Weights ({len(weights)})")
        weights = weights[:len(preds_list)]

    # ---- 모든 모델 결과를 id 기준 병합 (DataFrame 방식) ----
    dfs = []
    for i, preds in enumerate(preds_list):
        df = pd.DataFrame(list(preds.items()), columns=['id', f'model_{i}'])
        dfs.append(df)

    # id 기준으로 모든 모델 결과 병합
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on='id', how='outer')

    # ---- 가중치 적용 Hard Voting 함수 ----
    def weighted_hard_vote(row):
        vote_scores = defaultdict(float)

        # 각 모델의 예측에 가중치 적용
        for i, model_col in enumerate(merged.columns[1:]):  # id 컬럼 제외
            pred = row[model_col]
            if pd.notna(pred) and pred.strip():  # 유효한 예측값만 처리
                vote_scores[pred.strip()] += weights[i]

        if not vote_scores:
            return ""

        # 가장 높은 가중치를 받은 답변 선택
        return max(vote_scores, key=vote_scores.get)

    # 각 행에 hard voting 적용
    merged["prediction"] = merged.apply(weighted_hard_vote, axis=1)

    # 결과를 dictionary로 변환
    final_predictions = dict(zip(merged['id'], merged['prediction']))

    return final_predictions

def main(args: EnsembleArguments):
    # 가중치 설정 (입력 없으면 모두 1.0)
    if args.weights:
        weights = [float(w) for w in args.weights]
    else:
        weights = [1.0] * len(args.model_dirs)
        
    print(f"Ensemble Weights: {weights}")

    # Soft or Hard Voting Start
    if args.strategy == "soft":
        print("=== Starting Weighted Soft Voting ===")
        final_result = soft_voting(args.model_dirs, weights)
    else:
        print("=== Starting Weighted Hard Voting ===")
        final_result = hard_voting(args.model_dirs, weights)
    
    # Outputs path
    os.makedirs(args.output_dir, exist_ok=True)
    
    # JSON 저장
    json_output_path = os.path.join(args.output_dir, "predictions.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON to {json_output_path}")
    
    # CSV 저장 (제출용)
    csv_output_path = os.path.join(args.output_dir, "predictions_submit.csv")
    df = pd.DataFrame(list(final_result.items()), columns=['id', 'prediction'])
    # Header 제외, 탭으로 Seperate
    df.to_csv(csv_output_path, index=False, header=False, sep='\t')
    
    print(f"Saved CSV to {csv_output_path} (No Header, Tab Separated)")
    print("=== Ensemble Complete! ===")

if __name__ == "__main__":
    parser = HfArgumentParser(EnsembleArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)