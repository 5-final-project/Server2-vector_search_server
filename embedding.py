import logging
from langchain_huggingface import HuggingFaceEmbeddings
from config import config # Import config instance

logger = logging.getLogger(__name__)

def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model using LangChain."""
    try:
        logger.info(f"Loading HuggingFace embedding model: {config.EMBED_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL_NAME,
            model_kwargs=config.EMBED_MODEL_KWARGS,
            encode_kwargs=config.EMBED_ENCODE_KWARGS
        )
        # Test encoding to ensure the model loads correctly
        _ = embeddings.embed_query("Test query") 
        logger.info("HuggingFace embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        logger.exception(f"Failed to load embedding model '{config.EMBED_MODEL_NAME}': {e}")
        # Depending on the application's needs, you might want to exit or raise the exception.
        raise RuntimeError(f"Could not initialize embedding model: {e}")

# Instantiate the embedding model
# This will be imported and used by other modules
embedding_model = get_embedding_model()

# We might need the dimension later for some vector stores, although Chroma usually infers it.
# Let's add a way to get it if needed.
def get_embedding_dimension(embeddings: HuggingFaceEmbeddings) -> int:
    """Gets the embedding dimension from a loaded HuggingFaceEmbeddings model."""
    try:
        # Embed a dummy query and check the vector length
        dummy_embedding = embeddings.embed_query("dimension check")
        return len(dummy_embedding)
    except Exception as e:
        logger.error(f"Could not determine embedding dimension: {e}")
        # Fallback or raise error - let's raise for now
        raise RuntimeError("Failed to determine embedding dimension.")

# Get the dimension (optional, but good practice)
try:
    EMBEDDING_DIM = get_embedding_dimension(embedding_model)
    logger.info(f"Determined embedding dimension: {EMBEDDING_DIM}")
except RuntimeError as e:
    logger.error(str(e))
    # Handle cases where dimension couldn't be determined if necessary
    EMBEDDING_DIM = None # Or a default value if applicable
