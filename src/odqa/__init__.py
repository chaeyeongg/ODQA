"""
ODQA: Open-Domain Question Answering Pipeline

This package provides a complete ODQA pipeline that combines
retrieval and machine reading comprehension models.
"""

from .odqa_pipeline import ODQAPipeline
from .retrieval import SparseRetrieval, DenseRetrieval, BM25Retrieval, HybridRetrieval, RerankRetrieval
from .arguments import ModelArguments, DataTrainingArguments, RetrievalArguments

__version__ = "1.0.0"
__all__ = [
    "ODQAPipeline",
    "SparseRetrieval",
    "DenseRetrieval",
    "BM25Retrieval",
    "HybridRetrieval",
    "RerankRetrieval",
    "ModelArguments",
    "DataTrainingArguments",
    "RetrievalArguments",
]
