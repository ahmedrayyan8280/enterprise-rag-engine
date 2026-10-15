import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ingest import process_and_index_pdf
from rag_engine import RAGEngine

app = FastAPI(
    title="DocMind Enterprise RAG Engine",
    description="Production Retrieval-Augmented Generation Microservice with Page-Level Source Citations",
    version="1.0.0"
)

rag_service = RAGEngine()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "DocMind RAG Engine",
        "embedding_model": "models/gemini-embedding-2",
        "llm_backbone": "gemini-3.8-flash"
    }

@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        shutil.copyfileobj(file.file, temp_pdf)
        temp_path = temp_pdf.name

    try:
        total_chunks = process_and_index_pdf(temp_path)
    finally:
        os.remove(temp_path)

    return {
        "filename": file.filename,
        "status": "indexed_successfully",
        "total_chunks_created": total_chunks
    }

@app.post("/api/v1/query")
def query_rag(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    response = rag_service.query(question=request.question, k=request.top_k)
    return response