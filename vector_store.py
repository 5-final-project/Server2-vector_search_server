import logging
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever # Import VectorStoreRetriever
from langchain_core.documents import Document
from typing import List

from config import config # Import config instance
from embedding import embedding_model # Import the instantiated LangChain embedding model

logger = logging.getLogger(__name__)

def get_vector_store() -> Chroma:
    """Initializes and returns the Chroma vector store using LangChain."""
    try:
        logger.info(f"Initializing Chroma vector store at path: {config.VECTOR_DB_PATH}")
        logger.info(f"Using collection name: {config.CHROMA_COLLECTION_NAME}")
        
        vector_store = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model, # Pass the LangChain embedding model instance
            persist_directory=config.VECTOR_DB_PATH
        )
        logger.info("Chroma vector store initialized successfully.")
        # You might want to add a small test here, e.g., add and query a dummy doc
        # vector_store.add_texts(["test"], [{"source": "init"}], ids=["init_test"])
        # results = vector_store.similarity_search("test", k=1)
        # logger.info(f"Vector store test query returned: {len(results)} result(s)")
        # vector_store.delete(ids=["init_test"]) # Clean up test entry
        return vector_store
    except Exception as e:
        logger.exception(f"Failed to initialize Chroma vector store: {e}")
        raise RuntimeError(f"Could not initialize vector store: {e}")

# Instantiate the vector store
# This will be imported and used by other modules
vector_store: Chroma = get_vector_store()

# --- Optional Helper Functions (can be moved or integrated elsewhere) ---

def add_documents_to_store(docs: List[Document]):
    """Adds a list of LangChain Document objects to the vector store."""
    if not docs:
        logger.warning("No documents provided to add to the vector store.")
        return
    try:
        # Extract texts and metadatas for logging or potential direct use if needed
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        # Generate IDs if not present in metadata (Chroma can auto-generate, but explicit is often better)
        # For simplicity, let Chroma handle ID generation if not provided in metadata.
        
        logger.info(f"Adding {len(docs)} documents to the vector store...")
        vector_store.add_documents(docs)
        # Chroma with persist_directory handles persistence automatically on add/delete
        logger.info(f"Successfully added {len(docs)} documents.")
    except Exception as e:
        logger.exception(f"Error adding documents to vector store: {e}")
        # Decide on error handling: raise, log, etc.

def search_similar_documents(query: str, k: int = config.SEARCH_K, filter_dict: dict = None) -> List[Document]:
    """Performs similarity search in the vector store."""
    try:
        logger.info(f"Performing similarity search for query: '{query}' with k={k}")
        results = vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict # Pass the filter dictionary directly
        )
        logger.info(f"Similarity search returned {len(results)} documents.")
        return results
    except Exception as e:
        logger.exception(f"Error during similarity search: {e}")
        return [] # Return empty list on error

def search_similar_documents_with_score(query: str, k: int = config.SEARCH_K, filter_dict: dict = None) -> List[tuple[Document, float]]:
    """Performs similarity search and returns documents with scores."""
    try:
        logger.info(f"Performing similarity search with score for query: '{query}' with k={k}")
        # Note: LangChain's similarity_search_with_score returns a list of (Document, score) tuples
        # The score is distance (e.g., cosine distance). Lower is better.
        results_with_scores = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        logger.info(f"Similarity search with score returned {len(results_with_scores)} results.")
        # Convert distance score to similarity score (assuming cosine distance where similarity = 1 - distance)
        # This might need adjustment based on the actual distance metric used by the embedding/vector store.
        # Chroma with cosine distance: score = distance, so similarity = 1 - score.
        # Let's assume the score returned IS distance for now.
        # We will convert to similarity in the app layer if needed, or adjust threshold accordingly.
        return results_with_scores # Return list of (Document, distance_score)
    except Exception as e:
        logger.exception(f"Error during similarity search with score: {e}")
        return []

def get_retriever(k: int = config.SEARCH_K, filter_dict: dict = None) -> VectorStoreRetriever:
    """Gets a retriever object from the vector store."""
    logger.info(f"Creating retriever with k={k}")
    search_kwargs = {'k': k}
    if filter_dict:
        search_kwargs['filter'] = filter_dict
        
    try:
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        logger.info("Retriever created successfully.")
        return retriever
    except Exception as e:
        logger.exception(f"Error creating retriever: {e}")
        # Handle error appropriately
        raise RuntimeError(f"Could not create retriever: {e}")
