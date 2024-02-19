import io
import os
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import transformers

@dataclass
class InferenceArguments:
    "containing args used only in inference"
    max_length: int = field(default=512, metadata={"help": "max sequence length of generation"})

@dataclass
class ModelArguments:
    "containing args for retriever model and generation model"
    qa_model_name_or_path: str = field(default='microsoft/phi-2')
    doc_encoder_model_name_or_path: str = field(default='facebook/contriever')
    query_model_max_length: int = field(
        default=512,
        metadata={"help": "Query model maximum sequence length."},
    )
    doc_model_max_length: int = field(
        default=512,
        metadata={"help": "Key model maximum sequence length."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    "training arguments"
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


@dataclass 
class DataArguments:
    "containing args used in data loading"
    document_path: Optional[str] = field(default='/zfsauton2/home/yifuc/11711-RAG/data/databrick_qa/documents', 
                               metadata={'help':'path to the document folder of .txt files'})
    test_question_path: Optional[str]= field(default='/zfsauton2/home/yifuc/11711-RAG/data/databrick_qa/questions.txt',
                                   metadata={'help':'path to a .txt file with one question per line'})
    test_answer_path: Optional[str]= field(default='/zfsauton2/home/yifuc/11711-RAG/data/databrick_qa/answers.txt',
                                   metadata={'help':'path to a .txt file with multiple answers (seperated by ;) per line'})
    test_data_path: Optional[str]= field(default='/zfsauton2/home/yifuc/11711-RAG/data/databrick_qa/databrick_test.json',
                                   metadata={'help':'path to a .json file with list of qa pairs stored in dict'})
    chunk_size: Optional[str]= field(default=1000,
                                   metadata={'help':'number of tokens to chunk the document'})
    overlap: Optional[str]= field(default=150,
                                   metadata={'help':'number of tokens to overlapping between chunks'})
    retriever_topk: Optional[str]= field(default=5,
                                   metadata={'help':'number of related text retrieved'})
    result_path: Optional[str]= field(default='/zfsauton2/home/yifuc/11711-RAG/result/',
                                   metadata={'help':'folder to save various result'})