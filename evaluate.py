import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    GenerationConfig,
)
import numpy as np
from tqdm import tqdm 

# Load From Local
from model import RetrieverModel
from args import *
from main import retrieval_augmented_answer, prompt_formatting

def compute_metrics(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens), int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1, rec

def main():
    data_args, inference_args, model_args = HfArgumentParser((DataArguments, InferenceArguments, ModelArguments)).parse_args_into_dataclasses()

    retriever_model = RetrieverModel(model_args, data_args) # TODO: Use HF models or define in a different file
    tokenizer = AutoTokenizer.from_pretrained(model_args.qa_model_name_or_path, trust_remote_code=True) # TODO: Add tokenizer

    qa_model = AutoModelForCausalLM.from_pretrained(model_args.qa_model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda')
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

    #generate result path
    result_name = data_args.result_path + data_args.test_question_path.split('/')[-2] + '.txt'

    #TODO: let's check whether we can vectorize this 
    f1_all = []
    recall_all = []
    for idx, question in tqdm(enumerate(questions)):
        related_documents = retriever_model.retrieve(question, database)
        model_answer = retrieval_augmented_answer(question, related_documents, 
                                            model=qa_model, 
                                            tokenizer=tokenizer, 
                                            generation_config=generation_config, 
                                            model_args=model_args)
        exact_match = False
        f1 = []
        recall = []
        model_answer = model_answer[0].split('\n')
        model_answer = [m for m in model_answer if 'Output' in m][0][7:]
        
        evaluate_str = ''
        for answer in answers[idx]:
            if answer.lower() == model_answer.lower():
                exact_match = True
            f, r = compute_metrics(model_answer, answer)
            f1.append(round(f, 2))
            recall.append(round(r, 2))

        #append to calculate final test set performance
        f1_all.append(np.mean(f1))
        recall_all.append(np.mean(recall))

        evaluate_str += '----------------------------------------\n'
        evaluate_str += f'question is: {question}\n'
        evaluate_str += f'model answer is: {model_answer}\n'
        evaluate_str += f"actual answer (first reference) is: {answers[idx][0]}\n"
        evaluate_str += f"the predicted answer exactly match one of the references: {exact_match}\n"
        evaluate_str += f'f1 (max, min, avg): {max(f1)}, {min(f1)}, {np.mean(f1)}\n'
        evaluate_str += f'recall (max, min, avg): {max(recall)}, {min(recall)}, {np.mean(recall)}\n'
        evaluate_str += '----------------------------------------\n'

        if idx == 0:
            with open(result_name, 'w') as f:
                f.write(evaluate_str)
        else:
            with open(result_name, 'a') as f:
                f.write(evaluate_str)

    with open(result_name, 'a') as f:
        overall_result = ''
        overall_result += f'f1: {np.mean(f1_all)}\n'
        overall_result += f'recall: {np.mean(recall_all)}\n'
        f.write(overall_result)

if __name__ == "__main__":
    main()

