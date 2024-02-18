import torch
import torch.nn as nn

import transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    HFArgumentParser,
)

# Load from local files

from model import RetrieverModel
from args import ModelArguments, TrainingArguments


def main():
    
    model_args, training_args = HFArgumentParser((ModelArguments,TrainingArguments))

    retriever_model = RetrieverModel() # TODO: specify parameters

    train_dataset = RetrievalDataset() # TODO: define train dataset in a different file


    retrieval_trainer = Trainer(
        model=retriever_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        documents=documents, # TODO: define documents
        train_group_size = training_args.train_group_size, #TODO: training the retrival encoder
        data_collator=DataCollator( #TODO: implement data collator
            tokenizer=tokenizer,
            query_max_length=model_args.query_max_length,
            key_max_length=model_args.key_max_length,
        )
    )