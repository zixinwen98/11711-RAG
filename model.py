import torch.nn as nn
from transformers import (AutoTokenizer, 
                          AutoModel, 
                          AutoModelForCausalLM) #TODO: depend on support
from utils import load_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

'''
what should config have 
text_retriver: huggingface card/local 
query_retriver: huggingface card/local 
decoder: huggingface card/local 
data_path: path to .txt path
'''
class RetrieverModel(nn.Module):
    def __init__(self, config, data_config):
        super().__init__()
        self.config = config 
        self.data_config = data_config
        self.text_retriever = config.doc_encoder_model_name_or_path
        self.text_encoder_tokenizer = AutoTokenizer.from_pretrained(self.text_retriever)
        self.text_encoder = AutoModel.from_pretrained(self.text_retriever)

        #retriver module 
        #self.vector_databse = self.create_vector_store()
        self.document_path = data_config.document_path

    def create_vector_store(self):
        '''
        upon initialization, create and store indexes of text embeddings as 
        an model object for future similarity search
        return: 
        langchain vector store object
        '''
        docs = load_documents(self.document_path, 
                              chunk_size=int(self.data_config.chunk_size), 
                              chunk_overlap=int(self.data_config.overlap))
        embeddings = HuggingFaceEmbeddings(
                                        model_name=self.text_retriever,     # Provide the pre-trained model's path
                                        # model_kwargs=model_kwargs, # TODO: create device based on config 
                                        # encode_kwargs=encode_kwargs # TODO: write for normalize embedding 
                                        )
        vector_database = FAISS.from_documents(docs, embeddings)
        return vector_database
    
    def retrieve(self, question:str, database):
        '''
        question: [str]
        database: langchain data base object
        return:
        a list of document string
        '''
        related_documents = database.similarity_search(question, k=int(self.data_config.retriever_topk))
        return [doc.page_content for doc in related_documents]
