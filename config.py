import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    VECTOR_DB_BASE_PATH = os.path.join(BASE_DIR, "vector_db_collections") # Base path for all collections
    VECTOR_DB_PATH = os.path.join(VECTOR_DB_BASE_PATH, "vector_db_chroma_lc") # New path for LangChain Chroma DB
    
    # ChromaDB Settings
    # MASTER_COLLECTION_NAME will store all documents from all specific collections.
    MASTER_COLLECTION_NAME = os.getenv("MASTER_COLLECTION_NAME", "master")
    
    # Define the 5 specific collection names based on user's image
    SPECIFIC_COLLECTION_NAMES = [
        "strategy_documents",
        "compliance_documents",
        "operation_documents",
        "it_security_documents",
        "organization_documents"
    ]

    # All collection names that will be created/managed
    ALL_COLLECTION_NAMES = [MASTER_COLLECTION_NAME] + SPECIFIC_COLLECTION_NAMES

    # Text Splitting (can be customized per document type or globally)
    LC_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500)) # Target chunk size in characters (adjust as needed)
    LC_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100)) # Overlap in characters

    # Embedding Model (Keep the one requested earlier)
    EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    EMBED_MODEL_KWARGS = {'device': device}
    EMBED_ENCODE_KWARGS = {'normalize_embeddings': True} # Important for cosine similarity

    # Search settings
    SEARCH_K = int(os.getenv("SEARCH_K", 1)) # Number of results to return
    # For search_with_score, lower distance scores are better (more similar).
    # This threshold is for *distance* score (e.g., cosine distance).
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.1)) # Minimum similarity score to consider a result relevant (0.0 to 1.0)

    def __init__(self):
        # Create necessary directories if they don't exist
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_DB_PATH, exist_ok=True)
        # Each collection will have its own subdirectory under VECTOR_DB_BASE_PATH
        # os.makedirs(self.VECTOR_DB_BASE_PATH, exist_ok=True) # get_vector_store will handle subdirectories

# Instantiate the config to be used by other modules
config = Config()
