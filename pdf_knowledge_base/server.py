from fastapi import FastAPI, File, UploadFile
from typing import List
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os


app = FastAPI()

docstore = None
chain = None

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global docstore
    global chain
    
    raw_text = ""
    for file in files:
        try:    
            reader = PdfReader(file)
        except Exception as e:
            print(e)
            continue
        text = ""
        for i, page in enumerate(reader.pages):
            text += page.extract_text()
            if text:
                raw_text += text
                
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )       
    texts = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings()
    
    
    docstore = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

@app.post("/query")
async def make_query(query: str) -> str:
    if chain is None:
        raise Exception("Please upload pdfs first")
    
    docs = docstore.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    
    # return response
    return response