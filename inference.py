import re
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    GenerationConfig,
    AutoModelForSequenceClassification
)
import numpy as np
from tqdm import tqdm 
import pdb

# Load From Local
from model import RetrieverModel
from args import *
from utils import *


PROMPT_DICT = {
            "phi-2" : ("Background Information: {context}\nInstruct: {question}\n Output:"),
            "alpaca": (
                "Below is an question, paired with an document to supply important information. "
                "Write an response that appropriately answers the question.\n\n"
                "### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:"
            ),
            "microsoft/phi-2": "Background Information: {context}\nInstruct: {question}\n Output:",
            #"mistralai/Mistral-7B-Instruct-v0.2": "<s>[INST]You are asked to answer a question by extracting related facts from a given context.\nIt is very important to keep the generated answer to include only the facts that have appeared in the context. Below is the context and the question.\n Context: {context}\nQuestion: {question}\n Output: [/INST]",
            "mistralai/Mistral-7B-Instruct-v0.2": "<s>[INST]Please answer a question by information in context. Below is the context and the question.\n Context: {context}\nQuestion: {question}\n Output: [/INST]",
            "google/gemma-7b-it": "{context}\nGiven the context above, answer the following question: {question}\n Below is my answer: ",

        }

def strip_punctuation(text):
    pattern = re.compile(r'^[^\w\s]+|[^\w\s]+$')
    return pattern.sub('', text)

def prompt_formatting(question, documents, model_name):
    '''Formats the prompt for the model to generate the answer'''
    if isinstance(documents, list):
        documents = [doc+'\n' for doc in documents]
    documents = ''.join(documents) if isinstance(documents, list) else documents
    if 'phi-2' in model_name:
        return f"Background Information: {documents}\nInstruct: {question}\n Output:"
    elif 'alpaca' in model_name:
        return PROMPT_DICT["alpaca"].format_map({"question":question, "context":documents})
    elif 'mistral' in model_name or 'sfr' in model_name.lower():
        return PROMPT_DICT["mistralai/Mistral-7B-Instruct-v0.2"].format_map({"question":question, "context":documents})
    elif 'gemma' in model_name:
        return PROMPT_DICT["google/gemma-7b-it"].format_map({"question":question, "context":documents})
    else: 
        raise NotImplementedError

def retrieval_augmented_answer(question, related_docs, model, tokenizer, generation_config, model_args, reranker_model, reranker_tokenizer, return_doc=False, rerank=True):
    '''
    Generates an answer to the question using the related documents as context 
    using .generate() api of the model.
    '''
    if rerank:
        #assume we know there will be 6 docs 
        scores = reranker(reranker_model, reranker_tokenizer, question, related_docs)
        rank = scores.argsort(descending=True)[:3]
        related_docs = [related_docs[i] for i in rank]

    inputs_with_doc = prompt_formatting(question, related_docs, model_name=model_args.qa_model_name_or_path)
    inputs_with_doc = tokenizer(inputs_with_doc, return_tensors="pt", return_attention_mask=False).to(model.device)
    answers = model.generate(**inputs_with_doc, pad_token_id=tokenizer.eos_token_id, generation_config=generation_config)
    answers = tokenizer.batch_decode(answers)
    
    if return_doc:
        return answers, related_docs
    return answers


def reranker(reranker_model, tokenizer, question, documents):
    '''
    Reranks the documents based on the question
    '''
    pairs = [[question, doc] for doc in documents]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(reranker_model.device)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores

def main():
    seed_everything()
    data_args, inference_args, model_args = HfArgumentParser((DataArguments, InferenceArguments, ModelArguments)).parse_args_into_dataclasses()

    retriever_model = RetrieverModel(model_args, data_args) 

    #tokenizer
    tokenizer_path = model_args.tokenizer_path if model_args.tokenizer_path is not None else model_args.qa_model_name_or_path
    access_token = "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"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,token=access_token,)
    
    #qa model
    qa_model = AutoModelForCausalLM.from_pretrained(model_args.qa_model_name_or_path,
                                                    token=access_token, 
                                                    trust_remote_code=True, 
                                                    torch_dtype=model_args.qa_model_dtype).to(model_args.qa_model_device)
    
    reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large',trust_remote_code=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large', 
                                                                        trust_remote_code=True).to(model_args.qa_model_device)
    reranker_model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=inference_args.max_length, temperature=0.01, top_p=0.95, repetition_penalty=1.1,
        do_sample=True, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )

    # Documents
    database = retriever_model.create_vector_store() 
    with open('data/questions.txt', 'r') as f:
        questions = f.readlines()
    #remove white spaces and new lines on two sides of questions
    questions = [q.strip() for q in questions]

    #generate result path
    #d
    result_name = data_args.result_path +\
        'heldout-testset' +\
        f'_{model_args.qa_model_name_or_path}' +\
        f'_{model_args.vector_db_name_or_path}' +\
        f'_{model_args.doc_encoder_model_name_or_path.split("/")[-1]}' +\
        f'_{data_args.chunk_size}' +\
        f'_{data_args.overlap}' +\
        f'_{data_args.retriever_topk}' + '.txt'

    answers = []
    qa_model.eval()
    for idx, question in tqdm(enumerate(questions), total= len(questions)):
        
        related_documents = retriever_model.retrieve(question, database)
        #question = query_engineer(question, model_args.doc_encoder_model_name_or_path)
        model_answer = retrieval_augmented_answer(question, related_documents, 
                                            model=qa_model, 
                                            tokenizer=tokenizer, 
                                            generation_config=generation_config, 
                                            model_args=model_args,
                                            reranker_model=reranker_model,
                                            reranker_tokenizer=reranker_tokenizer,
                                            return_doc=False,
                                            rerank=model_args.use_reranker)
        
        model_answer = model_answer[0].split('\n')
        #pdb.set_trace()
        model_answer = [m for m in model_answer if 'Output:' in m][0][7:]

        model_answer = model_answer.replace('[/INST]', '')
        model_answer = model_answer.replace('</s>', '')
        if model_answer.startswith(':'): model_answer = model_answer[1:]
        
        model_answer = model_answer.strip()
        #print(model_answer)
        model_answer = model_answer.split()
        model_answer = [strip_punctuation(pred_token.lower()) for pred_token in model_answer]
        model_answer = ' '.join(model_answer)

        answers.append(model_answer)

    os.makedirs(os.path.dirname(result_name), exist_ok=True)
    with open(result_name, 'w') as f:
        for answer in answers:
            f.write(answer + '\n')
    

if __name__ == "__main__":
    main()

