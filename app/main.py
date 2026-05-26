import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag import build_rag_pipeline, ask_question

app = FastAPI(title="RAG Document QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage temporaire du pipeline en mémoire
qa_pipeline = None

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG Document QA API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_pipeline
    
    # Sauvegarder le PDF temporairement
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Construire le pipeline RAG
    qa_pipeline = build_rag_pipeline(temp_path)
    
    # Supprimer le fichier temporaire
    os.remove(temp_path)
    
    return {"message": f"PDF '{file.filename}' processed successfully"}

@app.post("/ask")
async def ask(request: QuestionRequest):
    global qa_pipeline
    
    if qa_pipeline is None:
        return {"error": "No PDF uploaded yet. Please upload a PDF first."}
    
    answer = ask_question(qa_pipeline, request.question)
    return {"answer": answer}