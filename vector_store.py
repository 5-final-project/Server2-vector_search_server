import logging
import os
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.documents import Document
from typing import List, Dict, Optional, Any

from config import config
from embedding import embedding_model

logger = logging.getLogger(__name__)

# Global dictionary to hold all vector store instances, keyed by collection name
vector_stores: Dict[str, Chroma] = {}

def initialize_vector_stores():
    """Initializes all Chroma vector stores defined in config and populates the global vector_stores dictionary."""
    global vector_stores
    if vector_stores: # Avoid re-initialization if already done
        logger.info("Vector stores already initialized.")
        return

    logger.info("Initializing all vector stores...")
    for collection_name in config.ALL_COLLECTION_NAMES:
        persist_directory = os.path.join(config.VECTOR_DB_BASE_PATH, collection_name)
        os.makedirs(persist_directory, exist_ok=True) # Ensure directory exists
        
        try:
            logger.info(f"Initializing Chroma vector store for collection: '{collection_name}' at path: {persist_directory}")
            store = Chroma(
                collection_name=collection_name, # Use the actual collection name for Chroma's internal naming
                embedding_function=embedding_model,
                persist_directory=persist_directory
            )
            vector_stores[collection_name] = store
            logger.info(f"Successfully initialized vector store for collection: '{collection_name}'.")
        except Exception as e:
            logger.exception(f"Failed to initialize Chroma vector store for collection '{collection_name}': {e}")
            # Decide if one failure should stop all, or allow partial initialization
            # For now, we'll raise an error if any store fails to initialize.
            raise RuntimeError(f"Could not initialize vector store for collection '{collection_name}': {e}")
    logger.info(f"All {len(vector_stores)} vector stores initialized.")

# Call initialization at module load time, or explicitly from app startup
# For simplicity here, calling at module load. Consider app startup hook for FastAPI.
initialize_vector_stores() 

def get_vector_store(collection_name: str) -> Chroma:
    """Retrieves a specific vector store instance by its name."""
    store = vector_stores.get(collection_name)
    if not store:
        logger.error(f"Vector store for collection '{collection_name}' not found or not initialized.")
        raise ValueError(f"Vector store for collection '{collection_name}' not found.")
    return store

def document_exists(collection_name: str, doc_name: str) -> bool:
    """Checks if a document with the given doc_name already exists in the specified collection."""
    try:
        store = get_vector_store(collection_name)
        # Chroma's get method with a where filter can check for metadata.
        # We are looking for any document chunk that has the matching 'doc_name'.
        results = store.get(where={"doc_name": doc_name}, limit=1) # Limit to 1 as we only need existence proof
        return bool(results and results.get('ids'))
    except ValueError: # Collection not found
        return False
    except Exception as e:
        logger.error(f"Error checking document existence for '{doc_name}' in '{collection_name}': {e}")
        return False # Or re-raise depending on desired error handling

def add_documents_to_store(docs: List[Document], target_collection_name: str):
    """Adds a list of LangChain Document objects to the specified target collection AND the master collection."""
    if not docs:
        logger.warning("No documents provided to add.")
        return

    # Add to the target specific collection
    try:
        target_store = get_vector_store(target_collection_name)
        logger.info(f"Adding {len(docs)} documents to collection: '{target_collection_name}'...")
        # For documents added to the target collection, set original_collection to itself
        # This ensures consistency if we later query original_collection from the target collection itself.
        docs_for_target = []
        for doc in docs:
            updated_metadata = doc.metadata.copy()
            updated_metadata['original_collection'] = target_collection_name
            docs_for_target.append(Document(page_content=doc.page_content, metadata=updated_metadata))

        target_store.add_documents(docs_for_target)
        logger.info(f"Successfully added {len(docs_for_target)} documents to collection: '{target_collection_name}'.")
    except Exception as e:
        logger.exception(f"Error adding documents to collection '{target_collection_name}': {e}")
        # Optionally re-raise or handle

    # Prepare documents for the master collection with original_collection metadata
    docs_for_master = []
    for doc in docs: # Iterate over the original docs list
        updated_metadata = doc.metadata.copy()
        updated_metadata['original_collection'] = target_collection_name # The true origin
        docs_for_master.append(Document(page_content=doc.page_content, metadata=updated_metadata))

    # Also add to the master collection, if the target is not already the master collection
    if target_collection_name != config.MASTER_COLLECTION_NAME:
        try:
            master_store = get_vector_store(config.MASTER_COLLECTION_NAME)
            logger.info(f"Adding {len(docs_for_master)} documents to master collection: '{config.MASTER_COLLECTION_NAME}'...")
            master_store.add_documents(docs_for_master)
            logger.info(f"Successfully added {len(docs_for_master)} documents to master collection: '{config.MASTER_COLLECTION_NAME}'.")
        except Exception as e:
            logger.exception(f"Error adding documents to master collection '{config.MASTER_COLLECTION_NAME}': {e}")
            # Optionally re-raise or handle
    elif target_collection_name == config.MASTER_COLLECTION_NAME:
        # If the target IS the master collection, the docs_for_target already handled this.
        # No separate add to master_store is needed because target_store IS master_store.
        # However, ensure that the metadata reflects this correctly for consistency.
        # The docs_for_target loop already set original_collection = MASTER_COLLECTION_NAME.
        pass 

def search_similar_documents(query: str, collection_name: str, k: int = -1, filter_dict: dict = None) -> List[Document]:
    """Performs similarity search in the specified vector store collection."""
    if k == -1: k = config.SEARCH_K # Use default from config if not provided
    try:
        store = get_vector_store(collection_name)
        logger.info(f"Performing similarity search in '{collection_name}' for query: '{query}' with k={k}")
        results = store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        logger.info(f"Similarity search in '{collection_name}' returned {len(results)} documents.")
        return results
    except Exception as e:
        logger.exception(f"Error during similarity search in '{collection_name}': {e}")
        return []

def search_similar_documents_with_score(query: str, collection_name: str, k: int = -1, filter_dict: dict = None) -> List[tuple[Document, float]]:
    """Performs similarity search and returns documents with scores from the specified collection."""
    if k == -1: k = config.SEARCH_K # Use default from config if not provided
    try:
        store = get_vector_store(collection_name)
        logger.info(f"Performing similarity search with score in '{collection_name}' for query: '{query}' with k={k}")
        results_with_scores = store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        logger.info(f"Similarity search with score in '{collection_name}' returned {len(results_with_scores)} results.")
        return results_with_scores
    except Exception as e:
        logger.exception(f"Error during similarity search with score in '{collection_name}': {e}")
        return []

def get_retriever(collection_name: str, k: int = -1, filter_dict: dict = None) -> VectorStoreRetriever:
    """Gets a retriever object from the specified vector store collection."""
    if k == -1: k = config.SEARCH_K # Use default from config if not provided
    logger.info(f"Creating retriever for collection '{collection_name}' with k={k}")
    search_kwargs = {'k': k}
    if filter_dict:
        search_kwargs['filter'] = filter_dict
        
    try:
        store = get_vector_store(collection_name)
        retriever = store.as_retriever(search_kwargs=search_kwargs)
        logger.info(f"Retriever created successfully for collection '{collection_name}'.")
        return retriever
    except Exception as e:
        logger.exception(f"Error creating retriever for '{collection_name}': {e}")
        raise RuntimeError(f"Could not create retriever for '{collection_name}': {e}")

def delete_document_by_id(doc_id: str, collection_name: str) -> bool:
    """Deletes a document by its doc_id.
    - If collection_name is a specific collection, deletes from it and the master collection.
    - If collection_name is the master collection, deletes from master and also from its 'original_collection' (if different).
    """
    logger.info(f"Deletion process started for doc_id '{doc_id}' from initial collection '{collection_name}'.")
    
    deleted_from_initial_target = False
    deleted_from_master = False
    deleted_from_original_specific = False # Used when master is the initial target

    if collection_name == config.MASTER_COLLECTION_NAME:
        # Case 1: Deletion initiated from the MASTER collection
        original_collection_name_from_meta = None
        try:
            master_store = get_vector_store(config.MASTER_COLLECTION_NAME)
            
            # Step 1a: Get metadata from master store to find 'original_collection' BEFORE deleting
            docs_in_master = master_store.get(where={'doc_id': doc_id}, include=["metadatas"])
            if docs_in_master and docs_in_master.get('ids'):
                if docs_in_master['metadatas']:
                    first_metadata = docs_in_master['metadatas'][0]
                    original_collection_name_from_meta = first_metadata.get('original_collection')
                
                # Step 1b: Delete from master store
                master_store.delete(where={'doc_id': doc_id})
                logger.info(f"Document '{doc_id}' deleted from master collection (initial target) '{config.MASTER_COLLECTION_NAME}'.")
                deleted_from_initial_target = True # Master was the initial target
                deleted_from_master = True         # Master itself is now handled
            else:
                logger.info(f"Document '{doc_id}' not found in master collection (initial target) '{config.MASTER_COLLECTION_NAME}'.")
                deleted_from_initial_target = True # Not found, effectively handled for initial target
                deleted_from_master = True         # Not found, effectively handled for master
        except ValueError:
            logger.warning(f"Master collection (initial target) '{config.MASTER_COLLECTION_NAME}' not found for deletion of doc_id '{doc_id}'.")
            deleted_from_initial_target = True
            deleted_from_master = True
        except Exception as e:
            logger.error(f"Error during master collection deletion for doc_id '{doc_id}': {e}")
            # Flags (deleted_from_initial_target, deleted_from_master) remain False

        # Step 1c: If master deletion was successful (or doc not found) and original_collection is known and different, delete from it
        if deleted_from_master and original_collection_name_from_meta and original_collection_name_from_meta != config.MASTER_COLLECTION_NAME:
            try:
                original_specific_store = get_vector_store(original_collection_name_from_meta)
                existing_original_ids = original_specific_store.get(where={'doc_id': doc_id}, include=[])['ids']
                if existing_original_ids:
                    original_specific_store.delete(where={'doc_id': doc_id})
                    logger.info(f"Document '{doc_id}' also deleted from its original specific collection '{original_collection_name_from_meta}'.")
                    deleted_from_original_specific = True
                else:
                    logger.info(f"Document '{doc_id}' not found in its original specific collection '{original_collection_name_from_meta}'.")
                    deleted_from_original_specific = True # Not found, effectively handled
            except ValueError:
                logger.warning(f"Original specific collection '{original_collection_name_from_meta}' (for doc_id '{doc_id}') not found.")
                deleted_from_original_specific = True # Collection not found, effectively handled
            except Exception as e:
                logger.error(f"Error deleting doc_id '{doc_id}' from original specific collection '{original_collection_name_from_meta}': {e}")
                # Flag (deleted_from_original_specific) remains False
        elif deleted_from_master: # Master deletion was okay, but no further action for original_specific needed
            deleted_from_original_specific = True
        # Else, if deleted_from_master is false, deleted_from_original_specific also remains false

    else:
        # Case 2: Deletion initiated from a SPECIFIC collection (collection_name != config.MASTER_COLLECTION_NAME)
        try:
            initial_target_store = get_vector_store(collection_name)
            existing_ids = initial_target_store.get(where={'doc_id': doc_id}, include=[])['ids']
            if existing_ids:
                initial_target_store.delete(where={'doc_id': doc_id})
                logger.info(f"Document '{doc_id}' deleted from initial target collection '{collection_name}'.")
                deleted_from_initial_target = True
            else:
                logger.info(f"Document '{doc_id}' not found in initial target collection '{collection_name}'.")
                deleted_from_initial_target = True # Not found, effectively handled
        except ValueError:
            logger.warning(f"Initial target collection '{collection_name}' not found for deletion of doc_id '{doc_id}'.")
            deleted_from_initial_target = True # Collection not found, effectively handled
        except Exception as e:
            logger.error(f"Error deleting doc_id '{doc_id}' from initial target collection '{collection_name}': {e}")
            # Flag (deleted_from_initial_target) remains False

        # Try to delete from master collection as well if deletion from specific collection was successful
        if deleted_from_initial_target:
            try:
                master_store = get_vector_store(config.MASTER_COLLECTION_NAME)
                existing_master_ids = master_store.get(where={'doc_id': doc_id}, include=[])['ids']
                if existing_master_ids:
                    master_store.delete(where={'doc_id': doc_id})
                    logger.info(f"Document '{doc_id}' also deleted from master collection '{config.MASTER_COLLECTION_NAME}'.")
                    deleted_from_master = True
                else:
                    logger.info(f"Document '{doc_id}' not found in master collection '{config.MASTER_COLLECTION_NAME}'.")
                    deleted_from_master = True # Not found, effectively handled
            except ValueError:
                logger.warning(f"Master collection '{config.MASTER_COLLECTION_NAME}' not found for secondary deletion of doc_id '{doc_id}'.")
                deleted_from_master = True # Master collection not found, effectively handled
            except Exception as e:
                logger.error(f"Error deleting doc_id '{doc_id}' from master collection '{config.MASTER_COLLECTION_NAME}': {e}")
                # Flag (deleted_from_master) remains False
        else: # Deletion from initial target failed or doc not found, so master part is skipped but considered 'handled' for overall success logic if initial target was already 'handled'.
            deleted_from_master = True 
        
        deleted_from_original_specific = True # This path is not applicable when initial target is specific

    # Final success is if all relevant deletions occurred or were not needed (e.g., doc not found).
    final_success = deleted_from_initial_target and deleted_from_master and deleted_from_original_specific
    logger.info(f"Deletion process for doc_id '{doc_id}' from initial collection '{collection_name}' finished. Success: {final_success}")
    return final_success

def get_all_documents_from_collection(collection_name: str) -> List[Dict[str, Any]]:
    """Gets all documents from the specified collection."""
    try:
        store = get_vector_store(collection_name)
        logger.info(f"Getting all documents from collection: '{collection_name}'...")
        results = store.get()
        logger.info(f"Got {len(results)} documents from collection: '{collection_name}'.")
        return results
    except Exception as e:
        logger.exception(f"Error getting all documents from collection '{collection_name}': {e}")
        return []
