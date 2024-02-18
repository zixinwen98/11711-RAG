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
import numpy as np

# Load From Local
from model import RetrieverModel
from args import *
from main import retrieval_augmented_answer, prompt_formatting

def compute_metrics(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1, rec

def main():
    data_args, inference_args, model_args = HFArgumentParser((DataArguments, InferenceArguments, ModelArguments))

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

    # load question and answer pairs
    with open(data_args.test_question_path, 'r') as file:
        questions = file.readlines()
    
    with open(data_args.test_answer_path, 'r') as file:
        answers = file.readlines()
        answers = [answer.split(';') for answer in answers]

    assert(len(questions) == len(answers), 'length of questions and answers must be the same')

    #TODO: let's check whether we can vectorize this 
    for idx, question in enumerate(questions):
        related_documents = retriever_model.retrieve(question, database)
        model_answer = retrieval_augmented_answer(question, related_documents, 
                                            model=qa_model, 
                                            tokenizer=tokenizer, 
                                            generation_config=generation_config, 
                                            model_args=model_args)
        exact_match = False
        f1 = []
        recall = []
        for answer in answers[idx]:
            if answer.lower() == model_answer.lower():
                exact_match = True
            f, r = compute_metrics(model_answer, answer)
            f1.append(f)
            recall.append(r)
        print('----------------------------------------')
        print(f'question is: {question}')
        print(f"answer is: {model_answer}")
        print(f"the predicted answer exactly match one of the references: {exact_match}")
        print(f'max f1 among reference answer is {max(f1)}, min is {min(f1)}, average is {np.mean(f1)}')
        print(f'max f1 among reference answer is {max(recall)}, min is {min(recall)}, average is {np.mean(recall)}')
        print('----------------------------------------')



