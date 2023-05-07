from argparse import ArgumentParser, Namespace
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os


def main(args: Namespace) -> None:
    
    load_dotenv()
    
    raw_text = ""
    for filename in os.listdir(args.pdf_file_dir):
        try:    
            reader = PdfReader(os.path.join(args.pdf_file_dir, filename))
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
    
    docsearch = FAISS.from_texts(texts, embeddings)
                
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    
    docs = docsearch.similarity_search(args.query)
    response = chain.run(input_documents=docs, question=args.query)
    print(response)
        
    


if "__main__" == __name__:
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--pdf_file_dir', type=str, help='pdf file directory', default="./pdfs")
    parser.add_argument('--query', type=str, help='query', default='What is YOLO?')
    args = parser.parse_args()
    main(args)