import io
import os
import json
import copy
import random
import logging
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import transformers
from transformers import (
    Trainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorWithPadding,
    default_data_collator,
)

### Import from local files
from args import * 
from dataset import FactualQuestionAnsweringDataset, RetrievalDataCollator

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "alpaca": (
        "Below is an question, paired with an document to supply important information. "
        "Write an response that appropriately answers the question.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "microsoft/phi-2": "Background Information: {documents}\nInstruct: {question}\n Output:",
}

def add_special_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": DEFAULT_EOS_TOKEN})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": DEFAULT_BOS_TOKEN})
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token": DEFAULT_UNK_TOKEN})

def train():
    model_args, data_args, training_args = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses()


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.qa_model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.qa_model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=tokenizer.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    add_special_token(tokenizer)
    

    train_dataset = FactualQuestionAnsweringDataset(data_path=data_args.qa_train_data_path, tokenizer=tokenizer)

    data_collator = RetrievalDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == '__main__':
    train()