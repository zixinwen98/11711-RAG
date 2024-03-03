import io
import os
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import transformers
import torch

@dataclass
class InferenceArguments:
    "containing args used only in inference"
    max_length: int = field(default=1024, metadata={"help": "max sequence length of generation"})

@dataclass
class ModelArguments:
    "containing args for retriever model and generation model"

    qa_model_name_or_path: str = field(default='microsoft/phi-2')
    tokenizer_path: Optional[str] = field(default='mistralai/Mistral-7B-Instruct-v0.2')
    qa_model_dtype : torch.dtype = field(default=torch.bfloat16)
    qa_model_device: str = field(default='cuda:1')
    doc_encoder_model_device: str = field(default='cuda:0')
    doc_encoder_model_name_or_path: str = field(default='facebook/contriever') #WhereIsAI/UAE-Large-V1
    query_model_max_length: int = field(
        default=512,
        metadata={"help": "Query model maximum sequence length."},
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Key model maximum sequence length."},
    )
    vector_db_name_or_path: str = field(default='FAISS')
    apply_lora: Optional[bool] = field(default=False)
    use_reranker: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    "training arguments"
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    local_rank: Optional[int] = field(default=None, metadata={'help': 'deepspeed config'})
    config_file: Optional[str] = field(default=None)


@dataclass 
class DataArguments:
    "containing args used in data loading"
    train_data_path: Optional[str]= field(default=None,
                                   metadata={'help':'path to a .json file with list of qa pairs stored in dict'})
    eval_data_path: Optional[str]= field(default=None,
                                   metadata={'help':'path to a .json file with list of qa pairs stored in dict'})
    document_path: Optional[str] = field(default='/zfsauton2/home/yifuc/11711-RAG/data/cmu', 
                               metadata={'help':'path to the document folder of .txt/.json files'})
    test_data_path: Optional[str]= field(default='/zfsauton2/home/yifuc/11711-RAG/data/questions.json',
                                   metadata={'help':'path to a .json file with list of qa pairs stored in dict'})
    chunk_size: Optional[str]= field(default=250,
                                   metadata={'help':'number of tokens to chunk the document'})
    overlap: Optional[str]= field(default=50,
                                   metadata={'help':'number of tokens to overlapping between chunks'})
    retriever_topk: Optional[str]= field(default=5,
                                   metadata={'help':'number of related text retrieved'})
    result_path: Optional[str]= field(default='/zfsauton2/home/yifuc/11711-RAG/result/',
                                   metadata={'help':'folder to save various result'})