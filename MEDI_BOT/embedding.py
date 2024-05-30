from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone


from MEDI_BOT.data_ingestion import Data_loading
from MEDI_BOT.model_api import load_model
from dotenv import load_dotenv

import sys
from exception import customexception
from logger import logging

from dotenv import load_dotenv

def download_embedding(model,document):
   
    try:
        index_name = "medical-bot"
        logging.info("Creating Chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100)
        document_chunk = text_splitter.split_documents(document)
        logging.info("chunk creation completed...")
        logging.info("Initializing chunk,index,model...")
        vectorstore = Pinecone.from_documents(
        document_chunk,
        index_name=index_name,
        embedding=model
    )
        logging.info("Returing vectorstore...")
        return vectorstore
    except Exception as e:
        raise customexception(e,sys)
