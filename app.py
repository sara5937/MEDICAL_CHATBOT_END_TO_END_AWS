import os
import asyncio
from flask import Flask, render_template, jsonify, request
from MEDI_BOT.model_api import load_model
from MEDI_BOT.prompt import prompt_template
from MEDI_BOT.embedding import download_embedding
from MEDI_BOT.data_ingestion import Data_loading
from MEDI_BOT.retriever import redriever_model
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from logger import logging
from dotenv import load_dotenv






app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

index_name = "medical-bot"

# Ensure environment variables are loaded
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

# Load models
Chat_model, embedding_model = load_model()

try:
    logging.info("Loading Existing index...")
    vectordb = Pinecone.from_existing_index(index_name, embedding_model)
except Exception as e:
    logging.info("Not found index ,...Loading data into database")
    logging.info("Creating Pinecone index with specified dimensions...")
        
    
    documents = asyncio.run(Data_loading())
    vectordb = download_embedding(embedding_model, documents)
logging.info("Setting Prompt template")
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input = msg
        logging.info(f"User input: {input}")
        logging.info("Calling redriever_model")
        qa = redriever_model(vectordb, Chat_model, input, chain_type_kwargs)
        logging.info("Passing User query")
        #result = qa({"query": input})
        result = qa
        logging.info("Getting response from Model")
        #logging.info(f"Response: {result['result']}")
        logging.info(f"Response: {result}")
        #return str(result["result"])
        return str(result)
    except Exception as e:
        logging.error(f"Error during chat processing: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True,use_reloader=False)
