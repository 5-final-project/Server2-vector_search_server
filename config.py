import os
import torch

class Config:
    UPLOAD_DIR = "uploads"
    VECTOR_DB_PATH = "vector_db_chroma_lc" # New path for LangChain Chroma DB
    
    # Embedding Model (Keep the one requested earlier)
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    EMBED_MODEL_KWARGS = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    EMBED_ENCODE_KWARGS = {'normalize_embeddings': True} # Important for cosine similarity

    # LangChain Text Splitter settings
    LC_CHUNK_SIZE = 500  # Target chunk size in characters (adjust as needed)
    LC_CHUNK_OVERLAP = 100 # Overlap in characters

    # ChromaDB settings (LangChain integration)
    CHROMA_COLLECTION_NAME = "lc_document_chunks"

    # Search settings
    SEARCH_K = 1 # Number of results to return
    SIMILARITY_THRESHOLD = 0.1 # Minimum similarity score to consider a result relevant (0.0 to 1.0)

# Ensure directories exist
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)

# Instantiate config
config = Config()
