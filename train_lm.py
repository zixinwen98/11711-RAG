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
import wandb
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
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

### Import from local files
from args import * 
from dataset import FactualQuestionAnsweringDataset, FactualQADataCollator

#os.environ["WANDB_PROJECT"] = "11711-RAG"  # name your W&B project
#os.environ["WANDB_LOG_MODEL"] = "checkpoint"

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
    "microsoft/phi-2": "Background Information: {context}\nInstruct: {question}\n Output: ",
    "mistralai/Mistral-7B-Instruct-v0.2": "<s>[INST]Please answer a question by information in context. Below is the context and the question.\n Context: {context}\nQuestion: {question}\n Output: [/INST]"
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

def main():
    model_args, data_args, training_args = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses()
    if training_args.report_to == "wandb":
        wandb.init(project="11711-RAG", config=training_args, reinit=True)
        wandb.login()

    #accelerator = setup_accelerator()
    #deepspeed_states = AcceleratorState().deepspeed_plugin
    #deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'] = opt.batch_size
    #deepspeed_states.deepspeed_config['checkpoint'] = {'use_node_local_storage': True}

    # logging.basicConfig(
    #         format='%(asctime)s - ' + f'Rank: {accelerator.process_index}' + ' - %(levelname)s - %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S',
    #         level=logging.INFO
    #         )
    logger = logging.getLogger(__name__)

    # random.seed(opt.seed)
    # np.random.seed(opt.seed)
    # torch.manual_seed(opt.seed)
    # torch.cuda.manual_seed(opt.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.qa_model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.qa_model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    add_special_token(tokenizer)
    

    train_dataset = FactualQuestionAnsweringDataset(data_path=data_args.train_data_path, tokenizer=tokenizer, prompt_template=PROMPT_DICT[f'{model_args.qa_model_name_or_path}'])
    eval_dataset = FactualQuestionAnsweringDataset(data_path=data_args.eval_data_path, tokenizer=tokenizer, prompt_template=PROMPT_DICT[f'{model_args.qa_model_name_or_path}'])

    data_collator = FactualQADataCollator(tokenizer=tokenizer)

    if model_args.apply_lora:
        peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir+model_args.qa_model_name_or_path)
    #model.save_pretrained(training_args.output_dir+model_args.qa_model_name_or_path)
    
    

if __name__ == '__main__':
    main()