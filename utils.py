from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from os import listdir

def prompt_engineer(context, question):
    '''
    context: list[str]
    question: str
    return: 
    a string that prompts the language model to generate answer based on prompt 
    '''
    pre_prompt = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nGenerate the next agent response by answering the question. Answer it as succinctly as possible. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities in your answer and use the titles to separate between topics or domains. If you cannot answer the question from the given documents, please state that you do not have an answer.\n"""
    prompt = pre_prompt + "CONTEXT:\n\n{context}\n" +"Question : {question}" + "[\INST]"
    llama_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])
    raise NotImplementedError

def load_documents(text_path, chunk_size=1000, chunk_overlap=150):
    '''
    text_path: str, a directory of .txt files
    return: 
    langchain document format (a list of trucated document)
    '''
    loader = DirectoryLoader(text_path, glob="**/*.txt")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs
