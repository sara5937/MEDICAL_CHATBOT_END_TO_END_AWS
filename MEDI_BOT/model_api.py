import os
from dotenv import load_dotenv
import sys

from langchain_community.llms import CTransformers
from langchain.embeddings import SentenceTransformerEmbeddings

from exception import customexception
from logger import logging

load_dotenv()

#OPEN_AI_API_KEY= os.getenv('HUGGINGFACE_API_KEY')



def load_model():
    

    try:
        logging.info("Initialize Embedding model")
        llm_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info("Embedding Model Loaded")

       
        logging.info("Chat Model Loaded")
        Chat_model=CTransformers(model="Model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.5})
        logging.info("Returning Model")
        return Chat_model,llm_embedding
    except Exception as e:
        raise customexception(e,sys)
        