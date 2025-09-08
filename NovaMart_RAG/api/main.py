import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline.pipeline import query_rag

app = FastAPI(
    title="NovaMart RAG API",
    description="Query NovaMart's institutional memory using RAG + LLAMA stack",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Welcome to NovaMart RAG API! Use /query to ask questions."}

@app.post("/query")
def ask_rag(query_request: QueryRequest):
    question = query_request.question
    answer = query_rag(question)
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
