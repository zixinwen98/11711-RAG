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

# Load From Local

from model import RetrieverModel
from args import *

PROMPT_DICT = {
            "phi-2" : ("Background Information: {documents}\nInstruct: {question}\n Output:"),
            "alpaca": (
                "Below is an question, paired with an document to supply important information. "
                "Write an response that appropriately answers the question.\n\n"
                "### Instruction:\n{question}\n\n### Input:\n{documents}\n\n### Response:"
            ),
            "microsoft/phi-2": "Background Information: {documents}\nInstruct: {question}\n Output:",
        }

def prompt_formatting(question, documents, model_name):
    '''Formats the prompt for the model to generate the answer'''

    documents = ''.join(documents) if isinstance(documents, list) else documents
    if 'phi-2' in model_name:
        return f"Background Information: {documents}\nInstruct: {question}\n Output:"
    elif 'alpaca' in model_name:
        return PROMPT_DICT["alpaca"].format_map({"question":question, "documents":documents})
    else: 
        raise NotImplementedError

def retrieval_augmented_answer(question, related_docs, model, tokenizer, generation_config, model_args, return_doc=False):
    '''
    Generates an answer to the question using the related documents as context 
    using .generate() api of the model.
    '''

    inputs_with_doc = prompt_formatting(question, related_docs, model_name=model_args.qa_model_name_or_path)
    inputs_with_doc = tokenizer(inputs_with_doc, return_tensors="pt", return_attention_mask=False).to(model.device)
    answers = model.generate(**inputs_with_doc, pad_token_id=tokenizer.eos_token_id, generation_config=generation_config)
    answers = tokenizer.batch_decode(answers)
    
    if return_doc:
        return answers, related_docs
    return answers

def main():
    data_args, inference_args, model_args = HfArgumentParser((DataArguments, InferenceArguments, ModelArguments)).parse_args_into_dataclasses()

    retriever_model = RetrieverModel(model_args, data_args) 

    tokenizer = AutoTokenizer.from_pretrained(model_args.qa_model_name_or_path, trust_remote_code=True) 
    qa_model = AutoModelForCausalLM.from_pretrained(model_args.qa_model_name_or_path, 
                                                    trust_remote_code=True,
                                                    torch_dtype=model_args.qa_model_dtype).to(model_args.qa_model_device)

    generation_config = GenerationConfig(
        max_length=inference_args.max_length, temperature=0.01, top_p=0.95,
        do_sample=True, use_cache=True,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )

    # Documents
    database = retriever_model.create_vector_store() 

    # question
    input_question = str(input("Enter your question after the colon, enter 'quit' to quit the program: ")) #TODO
    while input_question != 'quit':
        related_documents = retriever_model.retrieve(input_question, database)[:2]

        #inputs_with_doc = prompt_formating(input_question, related_documents)

        answer = retrieval_augmented_answer(input_question, related_documents, 
                                            model=qa_model, 
                                            tokenizer=tokenizer, 
                                            generation_config=generation_config, 
                                            model_args=model_args)
        print(f"answer is: {answer}")
        input_question = str(input("Enter your question after the colon, enter 'quit' to quit the program: ")) #TODO

if __name__ == "__main__":
    main()





    

