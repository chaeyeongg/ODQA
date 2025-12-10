import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

import torch
from transformers import AutoModel, AutoTokenizer
from rank_bm25 import BM25Okapi
import pickle
from sentence_transformers import CrossEncoder


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """
        초기화: 데이터 로드 및 TfidfVectorizer 설정
        """
        self.data_path = data_path
        
        # 1. 문서(Context) 데이터 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # 중복 제거 후 리스트로 변환
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # 2. TF-IDF Vectorizer 선언 (아직 학습 전)
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.p_embedding = None  # get_sparse_embedding() 호출 시 생성됨

    def get_sparse_embedding(self) -> None:
        """
        BM25 인덱스 생성 (캐싱 적용)
        """
        # 1. 캐시 파일 경로 설정
        # 데이터 개수를 파일명에 포함시켜 데이터 변경 시 충돌 방지
        cache_dir = os.path.join(self.data_path, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 파일명: bm25_tokenized_문서개수.bin
        cache_file = os.path.join(cache_dir, f"bm25_tokenized_{len(self.contexts)}.bin")

        tokenized_corpus = []

        # 2. 캐시 확인 및 로드
        if os.path.isfile(cache_file):
            print(f"✅ Loading BM25 tokenized corpus from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                tokenized_corpus = pickle.load(f)
        else:
            # 3. 없으면 생성 (토큰화 수행)
            print("🚀 Tokenizing all contexts for BM25 (This may take a while)...")
            tokenized_corpus = [self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing")]
            
            # 4. 저장
            print(f"💾 Saving BM25 tokenized corpus to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(tokenized_corpus, f)
        
        print("Build BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index build complete.")
        
    def build_index(self) -> None:
        """
        공통 인터페이스를 위한 alias. 내부적으로 TF-IDF 임베딩을 구성합니다.
        """
        self.get_sparse_embedding()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        핵심 2: 검색 수행 (단일 질문 or 데이터셋)
        ODQAPipeline과의 호환성을 위해 입력 타입에 따른 분기 처리는 유지.
        """
        assert self.p_embedding is not None, "get_sparse_embedding()을 먼저 실행해야 합니다."

        # Case A: 단일 질문 (str) -> ODQAPipeline.answer() 에서 사용
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            # 점수 리스트와, 실제 문서(text) 리스트를 반환
            return (doc_scores, [self.contexts[i] for i in doc_indices])

        # Case B: 데이터셋 (Dataset) -> ODQAPipeline.evaluate_odqa() 에서 사용
        elif isinstance(query_or_dataset, Dataset):
            total = []
            # 대량 검색 수행
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )
            
            # 결과 포매팅
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]), # 상위 문서들을 합침
                }
                # 정답이 있는 경우(Validation) 포함
                if "context" in example and "answers" in example:
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        단일 쿼리 검색 로직: (Query Vector) x (Passage Vectors Transposed)
        """
        # 1. 쿼리 벡터화
        query_vec = self.tfidfv.transform([query])
        
        # 2. 내적(Dot Product)을 통한 유사도 계산
        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        # 3. 점수 정렬 및 상위 k개 추출
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_scores = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        
        return doc_scores, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        다수 쿼리 검색 로직 (행렬 연산으로 한 번에 처리)
        """
        # 1. 쿼리 리스트 벡터화
        query_vec = self.tfidfv.transform(queries)
        
        # 2. 행렬 곱 (Queries x Passages)
        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        doc_scores = []
        doc_indices = []
        
        # 3. 각 쿼리별 상위 k개 추출
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
            
        return doc_scores, doc_indices


import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Union, Tuple
from datasets import Dataset

class DenseRetrieval:
    def __init__(
        self,
        model_name_or_path: str,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name_or_path # 모델 이름 저장 (캐시 파일명용)

        # 1. 문서 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # 2. 모델 로드
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        if self.device == "cuda":
            self.model.half() # FP16 적용
        self.model.eval()

        self.p_embedding: Optional[torch.Tensor] = None

    def _encode_texts(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        encoded_list = []
        instruction = "Represent this sentence for searching relevant passages: "
        
        target_texts = [instruction + t for t in texts] if is_query else texts

        for start in range(0, len(target_texts), self.batch_size):
            batch = target_texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                cls_emb = torch.nn.functional.normalize(cls_emb.float(), p=2, dim=1)
            
            encoded_list.append(cls_emb.cpu())
            
        return torch.cat(encoded_list, dim=0)

    def build_index(self) -> None:
        """
        캐시 파일이 존재하면 로드하고, 없으면 생성 후 저장합니다.
        """
        # # 1. 캐시 파일 경로 생성 (모델 이름 + 데이터 개수로 유니크하게 만듦)
        # model_tag = self.model_name.replace("/", "_") # 경로 문자 제거
        # cache_dir = os.path.join(self.data_path, "cache")
        # os.makedirs(cache_dir, exist_ok=True)
        
        # cache_file = os.path.join(cache_dir, f"dense_emb_{model_tag}_{len(self.contexts)}.pt")

        # 2. 캐시 확인 및 로드
        # if os.path.isfile(cache_file):
        if True:
            cache_file ="/data/ephemeral/home/ODQA/data/cache/dense_emb__data_ephemeral_home_ODQA_outputs_dense_retriever_finetuned_56737.pt"

            print(f"✅ Loading dense embedding from cache: {cache_file}")
            self.p_embedding = torch.load(cache_file)
        # else:
        #     # 3. 없으면 생성 (오래 걸림)
        #     print("🚀 Building dense passage embedding (This may take a while)...")
        #     self.p_embedding = self._encode_texts(self.contexts, is_query=False)
            
        #     # 4. 저장
        #     print(f"💾 Saving dense embedding to cache: {cache_file}")
        #     torch.save(self.p_embedding, cache_file)
            
        print(f"Dense embedding shape: {self.p_embedding.shape}")

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        return self._encode_texts(queries, is_query=True)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.p_embedding is not None, "build_index()를 먼저 실행해야 합니다."

        # (기존 retrieve 로직과 동일)
        if isinstance(query_or_dataset, str):
            q_emb = self._encode_queries([query_or_dataset])
            if self.p_embedding.dtype != q_emb.dtype:
                q_emb = q_emb.to(self.p_embedding.dtype)
            
            scores = torch.matmul(q_emb, self.p_embedding.T).squeeze(0)
            scores_np = scores.numpy()
            sorted_idx = np.argsort(scores_np)[::-1][:topk]
            doc_scores = scores_np[sorted_idx].tolist()
            return (doc_scores, [self.contexts[i] for i in sorted_idx])

        elif isinstance(query_or_dataset, Dataset):
            questions = list(query_or_dataset["question"])
            q_embs = self._encode_queries(questions)
            if self.p_embedding.dtype != q_embs.dtype:
                q_embs = q_embs.to(self.p_embedding.dtype)

            scores = torch.matmul(q_embs, self.p_embedding.T).numpy()
            total = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval")):
                sorted_idx = np.argsort(scores[idx])[::-1][:topk]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in sorted_idx]),
                }
                if "context" in example and "answers" in example:
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

# -------------------------------------------------------------------------
# BM25 Retrieval 클래스
# -------------------------------------------------------------------------

class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """
        초기화: 데이터 로드
        """
        self.data_path = data_path
        self.tokenize_fn = tokenize_fn
        
        # 1. 문서(Context) 데이터 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # 중복 제거 후 리스트로 변환
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        self.bm25 = None  # get_sparse_embedding() 호출 시 생성됨

    def get_sparse_embedding(self) -> None:
        """
        BM25 인덱스 생성 (캐싱 적용)
        """
        # 1. 캐시 파일 경로 설정
        # 데이터 개수를 파일명에 포함시켜 데이터 변경 시 충돌 방지
        cache_dir = os.path.join(self.data_path, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 파일명: bm25_tokenized_문서개수.bin
        cache_file = os.path.join(cache_dir, f"bm25_tokenized_{len(self.contexts)}.bin")

        tokenized_corpus = []

        # 2. 캐시 확인 및 로드
        if os.path.isfile(cache_file):
            print(f"✅ Loading BM25 tokenized corpus from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                tokenized_corpus = pickle.load(f)
        else:
            # 3. 없으면 생성 (토큰화 수행)
            print("🚀 Tokenizing all contexts for BM25 (This may take a while)...")
            tokenized_corpus = [self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing")]
            
            # 4. 저장
            print(f"💾 Saving BM25 tokenized corpus to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(tokenized_corpus, f)
        
        print("Build BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index build complete.")
        
    def build_index(self) -> None:
        """
        공통 인터페이스
        """
        self.get_sparse_embedding()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        검색 수행 (단일 질문 or 데이터셋)
        """
        assert self.bm25 is not None, "get_sparse_embedding()을 먼저 실행해야 합니다."

        # Case A: 단일 질문 (str)
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            return (doc_scores, [self.contexts[i] for i in doc_indices])

        # Case B: 데이터셋 (Dataset)
        elif isinstance(query_or_dataset, Dataset):
            total = []
            
            # 대량 검색 수행 (BM25는 행렬 연산이 아니므로 반복문 처리)
            # 속도 향상을 위해 병렬 처리를 고려할 수 있으나, 여기선 기본 반복문 사용
            with tqdm(total=len(query_or_dataset), desc="BM25 retrieval") as pbar:
                for idx, example in enumerate(query_or_dataset):
                    # 1. 쿼리별 검색
                    doc_scores, doc_indices = self.get_relevant_doc(example["question"], k=topk)
                    
                    # 2. 결과 저장
                    tmp = {
                        "question": example["question"],
                        "id": example["id"],
                        "context": " ".join([self.contexts[pid] for pid in doc_indices]),
                    }
                    if "context" in example and "answers" in example:
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    
                    total.append(tmp)
                    pbar.update(1)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        단일 쿼리에 대한 BM25 점수 계산 및 상위 k개 추출
        """
        # 1. 쿼리 토큰화
        tokenized_query = self.tokenize_fn(query)
        
        # 2. 점수 계산 (get_scores는 전체 문서에 대한 점수 리스트 반환)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 3. 정렬 및 상위 k개 추출
        # scores는 numpy array가 아니므로 변환 필요
        scores_np = np.array(scores)
        sorted_result = np.argsort(scores_np)[::-1]
        
        doc_scores = scores_np[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

        return doc_scores, doc_indices


class HybridRetrieval:
    def __init__(self, sparse_retriever, dense_retriever):
        """
        Hybrid Retrieval 초기화
        :param sparse_retriever: BM25Retrieval 인스턴스
        :param dense_retriever: DenseRetrieval 인스턴스
        """
        self.sparse = sparse_retriever
        self.dense = dense_retriever
        
        # 문서(Context) 리스트가 두 리트리버 간에 동일한지 확인 (안전장치)
        assert len(self.sparse.contexts) == len(self.dense.contexts), "문서 개수가 서로 다릅니다."

    # [추가할 코드] 파이프라인 인터페이스 호환용
    def build_index(self) -> None:
        """
        ODQAPipeline에서 호출하는 인터페이스 맞춤용 메서드.
        내부 리트리버들이 아직 빌드되지 않았다면 빌드를 수행합니다.
        """
        # 1. Sparse(BM25) 확인
        if getattr(self.sparse, "bm25", None) is None:
            self.sparse.get_sparse_embedding()
            
        # 2. Dense 확인
        if getattr(self.dense, "p_embedding", None) is None:
            self.dense.build_index()

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        점수 정규화 (Min-Max Scaling)
        BM25(0~무한대)와 Dense(-1~1)의 점수 범위를 0~1 사이로 맞춥니다.
        """
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10, alpha: float = 0.5
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Hybrid 검색 수행
        :param alpha: Dense 점수 반영 비율 (0.0 ~ 1.0)
                      alpha=1.0: Dense만 사용 / alpha=0.0: BM25만 사용
        """
        # Case A: 단일 질문 (str)
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk, alpha=alpha)
            return (doc_scores, [self.sparse.contexts[i] for i in doc_indices])

        # Case B: 데이터셋 (Dataset)
        elif isinstance(query_or_dataset, Dataset):
            total = []
            # 진행상황 표시
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Hybrid retrieval")):
                query = example["question"]
                doc_scores, doc_indices = self.get_relevant_doc(query, k=topk, alpha=alpha)
                
                tmp = {
                    "question": query,
                    "id": example["id"],
                    "context": " ".join([self.sparse.contexts[pid] for pid in doc_indices]),
                }
                if "context" in example and "answers" in example:
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: int = 10, alpha: float = 0.5) -> Tuple[List, List]:
        """
        단일 쿼리에 대해 BM25와 Dense 점수를 가중 합산하여 상위 k개 반환
        """
        # 1. Sparse 점수 계산 (BM25)
        # BM25Okapi는 get_scores로 전체 문서에 대한 점수를 줍니다.
        tokenized_query = self.sparse.tokenize_fn(query)
        sparse_scores = np.array(self.sparse.bm25.get_scores(tokenized_query))

        # 2. Dense 점수 계산 (Dot Product)
        # DenseRetrieval 내부 로직 활용 (쿼리 인코딩 -> 내적)
        dense_q_emb = self.dense._encode_queries([query])
        
        # 타입 불일치 방지 (FP16/FP32)
        if self.dense.p_embedding.dtype != dense_q_emb.dtype:
            dense_q_emb = dense_q_emb.to(self.dense.p_embedding.dtype)
            
        # 전체 문서에 대한 코사인 유사도(내적) 구하기
        # squeeze(0)으로 [1, num_docs] -> [num_docs] 형태로 변환
        dense_scores = torch.matmul(dense_q_emb, self.dense.p_embedding.T).squeeze(0).cpu().numpy()

        # 3. 점수 정규화 (Normalization)
        # 두 점수의 스케일이 다르므로 0~1로 맞춤
        norm_sparse = self.normalize_scores(sparse_scores)
        norm_dense = self.normalize_scores(dense_scores)

        # 4. 가중 합산 (Weighted Sum)
        hybrid_scores = alpha * norm_dense + (1 - alpha) * norm_sparse

        # 5. 정렬 및 Top-K 추출
        sorted_indices = np.argsort(hybrid_scores)[::-1][:k]
        top_scores = hybrid_scores[sorted_indices].tolist()
        top_indices = sorted_indices.tolist()

        return top_scores, top_indices

class RerankRetrieval:
    def __init__(self, base_retriever, model_name_or_path: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
        """
        Reranker 초기화
        """
        self.base_retriever = base_retriever # 1차 검색 BM25, Dense, Hybrid 등
        self.device = device

        print(f"Loading Reranker model: {model_name_or_path}...")
        self.cross_encoder = CrossEncoder(model_name_or_path, device=device)

        if self.device == "cuda":
            self.cross_encoder.model.half() # FP16

    def build_index(self):
        # Base retriever의 인덱스 빌드 호출
        if hasattr(self.base_retriever, "build_index"):
            self.base_retriever.build_index()
        elif hasattr(self.base_retriever, "get_sparse_embedding"):
            self.base_retriever.get_sparse_embedding()

    def retrieve(self, query_or_dataset, topk=10, **kwargs):            
            """
            1차 검색 후 Reranking 수행
            """
            # 후보 문서는 topk * 5개 정도
            # candidate_k = min(topk * 5, 100)
            candidate_k = 50

            # Case A: 단일 질문 (str)
            if isinstance(query_or_dataset, str):
                # 1. Base 검색 (문서 리스트를 받아옴)
                if hasattr(self.base_retriever, "retrieve"):
                    # Dense or Hybrid or BM25
                    base_result = self.base_retriever.retrieve(query_or_dataset, topk=candidate_k)
                    # base_result가 (scores, contexts) 튜플인지 확인
                    if isinstance(base_result, tuple):
                        doc_contexts = base_result[1]
                    elif isinstance(base_result, pd.DataFrame): 
                        # 혹시 DataFrame으로 온다면 처리 (단일 쿼리는 보통 tuple임)
                        return base_result 
                
                return self._rerank_single(query_or_dataset, doc_contexts, topk)
            
            # Case B: 데이터셋 (Dataset or List)
            elif isinstance(query_or_dataset, (Dataset, pd.DataFrame)):
                return self._rerank_bulk(query_or_dataset, topk, candidate_k)

    def _rerank_bulk(self, dataset, topk, candidate_k, **kwargs):
        total = []
        
        queries = dataset["question"] if isinstance(dataset, pd.DataFrame) else list(dataset["question"])
        ids = dataset["id"] if isinstance(dataset, pd.DataFrame) else list(dataset["id"])
        
        cols = dataset.columns if isinstance(dataset, pd.DataFrame) else dataset.column_names

        # 정답 및 원본 컨텍스트 보존 -> Validation을 위해
        has_answers = "answers" in cols
        answers = dataset["answers"] if has_answers else None
        if has_answers and not isinstance(answers, list): answers = list(answers)
        
        org_ctx_col =  None

        if "original_context" in cols:
            org_ctx_col = "original_context"
        elif "context" in cols:
            org_ctx_col = "context"
        
        original_contexts = list(dataset[org_ctx_col]) if org_ctx_col else None

        for i, query in enumerate(tqdm(queries, desc="Reranking")):
            # 1. Base Retriever 호출 시 **kwargs (alpha 등) 전달
            if hasattr(self.base_retriever, "get_relevant_doc"):
                # [수정] alpha 값 전달
                _, doc_indices = self.base_retriever.get_relevant_doc(query, k=candidate_k, **kwargs)
                
                if hasattr(self.base_retriever, "sparse"): # Hybrid
                    docs = [self.base_retriever.sparse.contexts[idx] for idx in doc_indices]
                else: # BM25
                    docs = [self.base_retriever.contexts[idx] for idx in doc_indices]
            
            elif hasattr(self.base_retriever, "retrieve"):
                # [수정] alpha 값 전달
                _, docs = self.base_retriever.retrieve(query, topk=candidate_k, **kwargs)

            # 2. Cross-Encoder로 점수 계산
            # (Query, Document) 쌍 만들기
            pairs = [[query, doc] for doc in docs]
            
            # 점수 예측 (Batch 처리가 내부적으로 됨)
            scores = self.cross_encoder.predict(pairs, batch_size=8)
            
            # 3. 점수 높은 순으로 정렬 및 Top-K 자르기
            sorted_idx = np.argsort(scores)[::-1][:topk]
            top_docs = [docs[k] for k in sorted_idx]
            
            # 4. 결과 저장
            tmp = {
                "question": query,
                "id": ids[i],
                "context": " ".join(top_docs) # 모델 입력용으로 합침
            }
            
            if has_answers:
                tmp["answers"] = answers[i]

            if original_contexts:
                tmp["original_context"] = original_contexts[i]
                
            total.append(tmp)

        return pd.DataFrame(total)