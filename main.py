import torch
import torch.nn as nn

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    HFArgumentParser,
    GenerationConfig,
)

# Load From Local

from model import RetrieverModel
from args import *


def prompt_formatting(question, documents, model_name):
    documents = ''.join(documents) if isinstance(documents, list) else documents
    if 'phi-2' in model_name:
        return f"Background Information: {documents}\nInstruct: {question}\n Output:"
    else: 
        raise NotImplementedError

def retrieval_augmented_answer(question, related_docs, model, tokenizer, generation_config, model_args):
    
    inputs_with_doc = prompt_formatting(question, related_docs, model_name=model_args.qa_moel_name_or_path)
    inputs_with_doc = tokenizer(inputs_with_doc, return_tensors="pt", return_attention_mask=False)

    answers = model.generate(**inputs_with_doc, generation_config=generation_config)
    answers = tokenizer.batch_decode(answers)
    return answers

def main():
    data_args, inference_args, model_args = HFArgumentParser((DataArguments, InferenceArguments, ModelArguments)).parse_args_into_dataclasses()

    retriever_model = RetrieverModel(model_args, data_args) # TODO: Use HF models or define in a different file

    tokenizer = AutoTokenizer.from_pretrained(model_args.qa_model_name_or_path, trust_remote_code=True) # TODO: Add tokenizer

    qa_model = AutoModelForCausalLM.from_pretrained(model_args.qa_model_name_or_path, trust_remote_code=True)

    generation_config = GenerationConfig(
        max_length=inference_args.max_length, temperature=0.01, top_p=0.95, repetition_penalty=1.1,
        do_sample=True, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )

    # Documents
    database = retriever_model.create_vector_store() #TODO: implement the database, assume to be list of strings

    # question
    input_question = str(input("Enter your question after the colon, enter 'quit' to quit the program: ")) #TODO
    while input_question != 'quit':
        related_documents = retriever_model.retrieve(input_question, database)

        #inputs_with_doc = prompt_formating(input_question, related_documents)

        answer = retrieval_augmented_answer(input_question, related_documents, 
                                            model=qa_model, 
                                            tokenizer=tokenizer, 
                                            generation_config=generation_config, 
                                            model_args=model_args)
        print(f"answer is: {answer}")
        input_question = str(input("Enter your question after the colon, enter 'quit' to quit the program: ")) #TODO





    

