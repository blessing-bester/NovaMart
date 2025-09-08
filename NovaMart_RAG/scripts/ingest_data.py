import os
from pathlib import Path

# Handle different llama-index versions
try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    try:
        from llama_index import SimpleDirectoryReader, VectorStoreIndex, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        print(f"Error importing llama-index: {e}")
        print("Please install: pip install llama-index llama-index-embeddings-huggingface")
        exit(1)

# Paths for the data
DATA_DIR = Path("./data")
EMBEDDINGS_DIR = Path("./embeddings")
INDEX_FILE = EMBEDDINGS_DIR / "nova_index.json"

# Ensuring eddings folder exists
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Loading all documents in the data folder
print(f"Loading documents from {DATA_DIR}...")
documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
print(f"Loaded {len(documents)} documents.")

# Set up local embedding model
print("Setting up local embedding model...")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Creating vector store index (embeddings)
print("Generating embeddings using local model...")
index = VectorStoreIndex.from_documents(documents)
# Saving embeddings
index.storage_context.persist(persist_dir=str(EMBEDDINGS_DIR))
print(f"Embeddings saved to {INDEX_FILE}")

print("NovaMart dataset successfully ingested! RAG system is ready for queries.")
