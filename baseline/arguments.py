from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="./data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )


@dataclass
class RetrievalArguments:
    """
    Arguments related to passage retrieval for ODQA.
    """

    data_path: str = field(
        default="./data",
        metadata={"help": "Path to directory where wikipedia_documents.json is stored."},
    )
    context_path: str = field(
        default="wikipedia_documents.json",
        metadata={"help": "File name of wikipedia passages json."},
    )
    top_k: int = field(
        default=10,
        metadata={"help": "Top-k passages to retrieve for each question in ODQA."},
    )
    use_faiss: bool = field(
        default=False,
        metadata={"help": "Whether to use FAISS index for retrieval (if built)."},
    )
    num_clusters: int = field(
        default=64,
        metadata={
            "help": "Number of clusters to use when building FAISS index (IVF)."
        },
    )
    retrieval_type: str = field(
        default="sparse",
        metadata={
            "help": "Retrieval method to use: 'sparse' or 'dense' or 'bm25' or  'hybrid. "
            "Additional methods can be added as needed."
        },
    )
    dense_model_name_or_path: Optional[str] = field(
        default="dragonkue/BGE-m3-ko",
        metadata={
            "help": "Model name or path to use for dense retrieval. "
            "If None and retrieval_type='dense', falls back to model_name_or_path."
        },
    )