import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_DIR = "./chroma_db"

def get_embedding_function():
    # Offloads mathematical vectorization to Google Cloud
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def process_and_index_pdf(pdf_path: str) -> int:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embedding_function()
    
    # Reset existing collection to prevent vector mixing across documents
    vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    try:
        vector_db.delete_collection()
    except Exception:
        pass

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    return len(chunks)

if __name__ == "__main__":
    print("Cloud ingestion engine initialized.")