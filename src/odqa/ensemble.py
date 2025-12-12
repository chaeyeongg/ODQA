import argparse
import json
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

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
        # 개수 안 맞으면 앞쪽부터 맞추거나 1.0 처리 (여기선 에러 대신 안전하게 slice)
        weights = weights[:len(nbest_preds_list)]

    keys = list(nbest_preds_list[0].keys())
    final_predictions = {}
    
    for key in tqdm(keys, desc="Weighted Soft Voting"):
        candidate_scores = defaultdict(float)
        
        # [수정됨] enumerate를 사용해 해당 모델의 weight를 가져옴
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
    Hard Voting: (투표 수 * 가중치) 합산
    """
    preds_list = load_predictions(model_dirs, strategy="hard")
    
    if not preds_list:
        raise ValueError("No valid prediction files found!")

    if len(preds_list) != len(weights):
        print(f"Warning: Loaded models ({len(preds_list)}) != Weights ({len(weights)})")
        weights = weights[:len(preds_list)]

    all_ids = set().union(*[d.keys() for d in preds_list])
    final_predictions = {}
    
    for q_id in tqdm(all_ids, desc="Weighted Hard Voting"):
        vote_scores = defaultdict(float)
        
        # [수정됨] enumerate로 가중치 적용
        for i, model_preds in enumerate(preds_list):
            w = weights[i]
            if q_id in model_preds:
                pred = model_preds[q_id]
                if pred:
                    # 한 표를 던질 때 가중치만큼 점수가 올라감 (1표가 아니라 1.5표 등)
                    vote_scores[pred] += 1.0 * w
        
        if not vote_scores:
            final_predictions[q_id] = ""
            continue
            
        # 가장 높은 점수를 얻은 답변 선택
        best_answer = max(vote_scores, key=vote_scores.get)
        final_predictions[q_id] = best_answer
        
    return final_predictions

def main(args):
    # 가중치 설정 (입력 없으면 모두 1.0)
    if args.weights:
        weights = [float(w) for w in args.weights]
    else:
        weights = [1.0] * len(args.model_dirs)
        
    print(f"Ensemble Weights: {weights}")

    if args.strategy == "soft":
        print("=== Starting Weighted Soft Voting ===")
        final_result = soft_voting(args.model_dirs, weights)
    else:
        print("=== Starting Weighted Hard Voting ===")
        final_result = hard_voting(args.model_dirs, weights)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # JSON 저장
    json_output_path = os.path.join(args.output_dir, "predictions.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON to {json_output_path}")
    
    # CSV 저장 (제출용)
    csv_output_path = os.path.join(args.output_dir, "predictions_submit.csv")
    df = pd.DataFrame(list(final_result.items()), columns=['id', 'prediction'])
    # Header 제외, 탭 분리
    df.to_csv(csv_output_path, index=False, header=False, sep='\t')
    
    print(f"Saved CSV to {csv_output_path} (No Header, Tab Separated)")
    print("=== Ensemble Complete! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_dirs", 
        nargs="+", 
        required=True, 
        help="Input files path (e.g., outputs/nbest1.json outputs/pred2.csv)"
    )
    
    # [추가됨] 가중치 인자
    parser.add_argument(
        "--weights", 
        nargs="+", 
        help="Weights for each model (e.g. 1.0 1.5 1.0). Must match the order of --model_dirs"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs/ensemble_result", 
        help="Output directory"
    )
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="hard", 
        choices=["soft", "hard"],
        help="Ensemble strategy"
    )
    
    args = parser.parse_args()
    main(args)