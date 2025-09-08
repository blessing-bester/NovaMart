from fastapi import FastAPI
from pydantic import BaseModel
import os

try:
    from llama_index.llms.openai import OpenAI
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    from llama_index import OpenAI, VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

app = FastAPI(
    title="NovaMart RAG API",
    description="Query NovaMart's institutional memory using RAG + LLAMA stack",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

# Initialize RAG system
EMBEDDINGS_PATH = "./embeddings"
# Use local embedding model to avoid OpenAI API key requirement
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Comment out OpenAI LLM for now - you can add your API key later
# Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

print(f"Looking for embeddings at: {os.path.abspath(EMBEDDINGS_PATH)}")
if os.path.exists(EMBEDDINGS_PATH):
    print(f"Embeddings directory exists. Files: {os.listdir(EMBEDDINGS_PATH)}")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=EMBEDDINGS_PATH)
        index = load_index_from_storage(storage_context)
        print("✅ Index loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        index = None
else:
    print(f"❌ Embeddings directory not found at {EMBEDDINGS_PATH}")
    index = None

def query_rag(question):
    if index is None:
        return "Error: Index not loaded. Please run ingest_data.py first to create embeddings."
    
    try:
        # Use retriever to get relevant documents without LLM
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(question)
        
        if not nodes:
            return "No relevant information found in the documents."
        
        # Combine the retrieved text
        context = "\n\n".join([node.text for node in nodes])
        return f"Based on NovaMart's documents:\n\n{context}"
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

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