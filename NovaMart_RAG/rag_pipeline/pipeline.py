
try:
    from llama_index.llms.openai import OpenAI
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
except ImportError:
    from llama_index import OpenAI, VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage

import os

EMBEDDINGS_PATH = "../embeddings"

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

try:
    storage_context = StorageContext.from_defaults(persist_dir=EMBEDDINGS_PATH)
    index = load_index_from_storage(storage_context)
except (FileNotFoundError, ValueError, Exception) as e:
    print(f"Index not found at {EMBEDDINGS_PATH}. Please run ingest_data.py first.")
    index = None

def query_rag(question):
    if index is None:
        return "Error: Index not loaded. Please run ingest_data.py first to create embeddings."
    
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response.response