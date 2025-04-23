import os
import logging
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document # Import LangChain Document type

# Import refactored modules and components
from config import config
# Import both search functions and other necessary components
from vector_store import (
    vector_store, 
    search_similar_documents, # Original search without score
    search_similar_documents_with_score, # Search with score
    add_documents_to_store 
)
from document_processor import load_and_split_document # Import the processing function
from api_models import (
    KeywordSearchRequest,
    DocumentUploadResponse,
    SearchResponse,
    ListDocumentsResponse,
    DocumentInfo,
    SearchResultItem
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Search System (LangChain Refactored)")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document_endpoint(file: UploadFile = File(...)):
    """
    Uploads a document, loads, splits, and indexes it using LangChain components.
    """
    # Create a unique ID for this document processing job
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}_{file.filename}")
    logger.info(f"Received file upload: {file.filename} (Assigned Doc ID: {doc_id})")

    try:
        # 1. Save the uploaded file temporarily
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.info(f"File saved temporarily to: {file_path}")

        # 2. Load and split the document using LangChain loaders/splitters
        # The function now returns LangChain Document objects (chunks) with metadata
        split_chunks: List[Document] = load_and_split_document(file_path, doc_id=doc_id)

        if not split_chunks:
            logger.warning(f"No chunks generated for file {file.filename}. It might be empty or failed to load/split.")
            # Decide if this is an error or just an empty file case
            raise HTTPException(status_code=400, detail=f"Could not process file {file.filename}. No content found or processing failed.")

        # 3. Add the processed chunks to the vector store
        add_documents_to_store(split_chunks) # Uses the helper in vector_store.py

        logger.info(f"Successfully processed and indexed document: {file.filename} (Doc ID: {doc_id}), added {len(split_chunks)} chunks.")
        
        # 4. Return response
        return DocumentUploadResponse(
            doc_id=doc_id,
            doc_name=file.filename,
            num_chunks_added=len(split_chunks)
        )

    except ValueError as ve:
         # Specific error from load_document for unsupported types
         logger.error(f"Unsupported file type for {file.filename}: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
         # Errors during loading (e.g., missing dependencies) or vector store init
         logger.error(f"Runtime error processing {file.filename}: {re}")
         raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.exception(f"Unexpected error during document upload for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error processing file: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
            except OSError as oe:
                 logger.error(f"Error removing temporary file {file_path}: {oe}")


@app.post("/search", response_model=SearchResponse)
async def search_endpoint_original(request: KeywordSearchRequest):
    """
    Original search: Searches the vector database using LangChain's similarity search 
    without applying a score threshold. Returns top K results regardless of score.
    """
    query = " ".join(request.keywords) if isinstance(request.keywords, list) else request.keywords
    logger.info(f"Received original search request: query='{query}', k={request.k}, filter={request.filter}")
    
    try:
        # Use the original helper function without scores
        results: List[Document] = search_similar_documents(
            query=query, 
            k=request.k, 
            filter_dict=request.filter
        )
        
        # Convert LangChain Document results to our API model (score will be None)
        response_items = [SearchResultItem(page_content=doc.page_content, metadata=doc.metadata) for doc in results]
        
        logger.info(f"Original search completed. Found {len(response_items)} results.")
        return SearchResponse(results=response_items)
        
    except Exception as e:
        logger.exception(f"Error during original search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {e}")


@app.post("/search_score", response_model=SearchResponse)
async def search_endpoint_with_score(request: KeywordSearchRequest):
    """
    Search with score: Searches the vector database, returns results with scores, 
    and filters them based on the SIMILARITY_THRESHOLD defined in config.
    """
    query = " ".join(request.keywords) if isinstance(request.keywords, list) else request.keywords
    logger.info(f"Received search_score request: query='{query}', k={request.k}, filter={request.filter}, threshold={config.SIMILARITY_THRESHOLD}")
    
    try:
        # Use the helper function that returns scores
        results_with_scores: List[tuple[Document, float]] = search_similar_documents_with_score(
            query=query,
            k=request.k, # Fetch initial k results
            filter_dict=request.filter
        )

        # Filter results based on the similarity threshold
        filtered_results = []
        for doc, distance_score in results_with_scores:
            # Convert distance score to similarity score (assuming cosine: 1 - distance)
            similarity_score = 1.0 - distance_score 
            
            if similarity_score >= config.SIMILARITY_THRESHOLD:
                filtered_results.append(
                    SearchResultItem(
                        page_content=doc.page_content,
                        metadata=doc.metadata,
                        score=similarity_score # Include the similarity score
                    )
                )
            else:
                 logger.debug(f"Filtering out result with score {similarity_score:.4f} (below threshold {config.SIMILARITY_THRESHOLD})")

        # Sort the final filtered results by score (highest similarity first)
        filtered_results.sort(key=lambda x: x.score, reverse=True)

        if not filtered_results:
             logger.info(f"Search_score completed for '{query}'. No results met the similarity threshold of {config.SIMILARITY_THRESHOLD}.")
        else:
             logger.info(f"Search_score completed for '{query}'. Found {len(filtered_results)} results meeting the threshold.")
             
        response_items = filtered_results # Already in the correct format
        return SearchResponse(results=response_items)
        
    except Exception as e:
        logger.exception(f"Error during search_score: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during search_score: {e}")


@app.get("/documents", response_model=ListDocumentsResponse)
async def list_documents_endpoint():
    """
    Lists the documents currently indexed by retrieving unique doc_ids and names from metadata.
    Note: This implementation fetches all metadata, which might be inefficient for very large databases.
    """
    logger.info("Received request to list documents.")
    try:
        # Access the underlying Chroma collection to get metadata
        # This bypasses LangChain's standard retriever interface but is necessary for this specific task
        # Ensure the vector_store instance is the Chroma class instance
        if not isinstance(vector_store, Chroma):
             logger.error("Vector store is not a Chroma instance, cannot directly access collection.")
             raise HTTPException(status_code=500, detail="Internal configuration error: Vector store type mismatch.")

        chroma_collection = vector_store.get() # Gets all data {'ids', 'embeddings', 'documents', 'metadatas'}
        
        documents_dict: Dict[str, str] = {} # {doc_id: doc_name}
        if chroma_collection and chroma_collection.get('metadatas'):
            for metadata in chroma_collection['metadatas']:
                doc_id = metadata.get('doc_id')
                doc_name = metadata.get('doc_name')
                # Ensure we have both ID and name, and haven't seen this ID yet
                if doc_id and doc_name and doc_id not in documents_dict:
                    documents_dict[doc_id] = doc_name
        
        document_list = [DocumentInfo(doc_id=id, doc_name=name) for id, name in documents_dict.items()]
        
        logger.info(f"Found {len(document_list)} unique documents by scanning metadata.")
        return ListDocumentsResponse(documents=document_list)

    except Exception as e:
        logger.exception(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error listing documents: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for LangChain-based application...")
    # Ensure necessary environment variables or configurations are set if needed
    # e.g., for specific model downloads or API keys if used later.
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
