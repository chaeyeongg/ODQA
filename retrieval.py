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
        핵심 1: 문서들을 TF-IDF 임베딩 벡터로 변환 (Fit & Transform)
        pickle 저장/로딩 로직을 제거하고, 매번 계산하도록 단순화함.
        """
        print("Build passage embedding...")
        self.p_embedding = self.tfidfv.fit_transform(self.contexts)
        print(f"Embedding shape: {self.p_embedding.shape}")
        print("Embedding complete.")

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


class DenseRetrieval:
    """
    Dense retrieval using a transformer encoder and cosine similarity.
    - 문서(context)와 쿼리를 동일한 인코더로 임베딩하고,
      내적(코사인 유사도)을 기준으로 상위 top-k를 검색합니다.
    """

    def __init__(
        self,
        model_name_or_path: str,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
        max_length: int = 256,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size

        # 1. 문서(Context) 데이터 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.ids = list(range(len(self.contexts)))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # 2. Dense encoder 준비
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

        self.p_embedding: Optional[torch.Tensor] = None  # build_index()로 생성

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        텍스트 리스트를 [CLS] 임베딩으로 변환하고 L2 정규화합니다.
        """
        encoded_list = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
            encoded_list.append(cls_emb.cpu())
        return torch.cat(encoded_list, dim=0)  # [N, hidden]

    def build_index(self) -> None:
        """
        문서(Context)들에 대한 dense 임베딩 인덱스를 구성합니다.
        """
        print("Build dense passage embedding...")
        self.p_embedding = self._encode_texts(self.contexts)  # [num_ctx, dim]
        print(f"Embedding shape: {self.p_embedding.shape}")
        print("Dense embedding complete.")

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        return self._encode_texts(queries)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.p_embedding is not None, "build_index()를 먼저 실행해야 합니다."

        # Case A: 단일 질문 (str)
        if isinstance(query_or_dataset, str):
            q_emb = self._encode_queries([query_or_dataset])  # [1, dim]
            scores = torch.matmul(q_emb, self.p_embedding.T).squeeze(0)  # [num_ctx]
            scores_np = scores.numpy()
            sorted_idx = np.argsort(scores_np)[::-1][:topk]
            doc_scores = scores_np[sorted_idx].tolist()
            return (doc_scores, [self.contexts[i] for i in sorted_idx])

        # Case B: 데이터셋 (Dataset)
        elif isinstance(query_or_dataset, Dataset):
            questions = list(query_or_dataset["question"])
            q_embs = self._encode_queries(questions)  # [num_q, dim]
            scores = torch.matmul(q_embs, self.p_embedding.T).numpy()  # [num_q, num_ctx]

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