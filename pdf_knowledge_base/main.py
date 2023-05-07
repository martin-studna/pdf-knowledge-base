from argparse import ArgumentParser, Namespace
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import uvicorn
from server import app



def main(args: Namespace) -> None:
    load_dotenv()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if "__main__" == __name__:
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--pdf_file_dir', type=str, help='pdf file directory', default="./pdfs")
    parser.add_argument('--query', type=str, help='query', default='What is YOLO?')
    args = parser.parse_args()
    main(args)