from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader

from logger import logging
from exception import customexception
import sys
import os
from langchain.vectorstores import Pinecone
from pinecone import Pinecone,ServerlessSpec


async def Data_loading():
    #loading the PDF files
    try:

        logging.info("loading API key...")
        load_dotenv()
        logging.info("Saving API...")
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

        pc = Pinecone(api_key=PINECONE_API_KEY)
        logging.info("Creating Index...")
        pc.create_index(
        name="medical-bot",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
        region="us-east-1"
        ))

        logging.info("data loading started...")

        loader = PyPDFLoader("D:/MLOPS_END_PROJECT/LLM_PROJECT/New folder/QA-System-Gemini/data/Medical_book.pdf")
        documents = await loader.aload()  # Await the async method
        logging.info("data loading completed...")
        return documents
    except Exception as e:
        logging.info("exception in loading data...")
        raise customexception(e, sys)
    







