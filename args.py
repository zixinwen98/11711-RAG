import io
import os
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import transformers

@dataclass
class InferenceArguments:
    "containing args used only in inference"
    max_length: int = field(default=256, metadata={"help": "max sequence length of generation"})

@dataclass
class ModelArguments:
    "containing args for retriever model and generation model"
    qa_model_name_or_path: str = field(default='google/flan-t5-base')
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
    document_path: Optional[str] = field(default='/data', 
                               metadata={'help':'path to the document folder of .txt files'})
    test_question_path: Optional[str]= field(default=None,
                                   metadata={'help':'path to a .txt file with one question per line'})
    test_answer_path: Optional[str]= field(default=None,
                                   metadata={'help':'path to a .txt file with multiple answers (seperated by ;) per line'})