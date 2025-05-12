import os
import logging
import uuid
import zipfile # Added for zip file handling
import tempfile # Added for temporary directory management
import unicodedata # Added for Unicode filename normalization
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document # Import LangChain Document type

# Import refactored modules and components
from config import config
# Import both search functions and other necessary components
from vector_store import (
    search_similar_documents, # Original search without score
    search_similar_documents_with_score, # Search with score
    add_documents_to_store,
    delete_document_by_id, # Added for document deletion
    initialize_vector_stores, # Added for initializing stores at startup
    get_vector_store, # Added for retrieving a specific vector store instance
    document_exists # Added for checking if a document already exists
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
from langchain_chroma import Chroma # Chroma import for type checking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Search System (LangChain Refactored)")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/upload-document") # Removed response_model for custom JSONResponse
async def upload_document_endpoint(files: List[UploadFile] = File(...), collection_name: Optional[str] = Query(None, description="The name of the collection to upload the document to. Defaults to the master collection if not provided.")):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    effective_collection_name = collection_name if collection_name else config.MASTER_COLLECTION_NAME
    logger.info(f"Attempting to upload {len(files)} file(s) to collection: {effective_collection_name}")

    # Ensure the target collection exists or can be created (get_vector_store might create it)
    try:
        get_vector_store(effective_collection_name)
    except ValueError as e:
        logger.error(f"Error ensuring collection '{effective_collection_name}' exists before upload: {e}")
        # Depending on config, this might mean we cannot proceed.
        # For now, we assume get_vector_store or add_documents_to_store will create if needed.
        pass 

    processed_files_info = []

    for file in files:
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()
        # Assign a doc_id for the uploaded file itself (zip or single). 
        # For sub-files in a zip, new doc_ids will be generated.
        top_level_doc_id = str(uuid.uuid4()) 

        if file_extension == '.zip':
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_file_path = os.path.join(temp_dir, original_filename)
                    # Save the uploaded zip file to the temporary directory
                    with open(zip_file_path, "wb") as buffer:
                        buffer.write(await file.read())
                    
                    logger.info(f"Extracting zip file: {original_filename} to {temp_dir}")
                    # Instead of zip_ref.extractall(temp_dir), manually extract to handle filenames
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        for member in zip_ref.infolist():
                            try:
                                raw_filename_from_zip = member.filename
                                resolved_filename_str = ""

                                # Check if UTF-8 flag (0x800) is set in general purpose bit flag
                                if not (member.flag_bits & 0x800):
                                    # UTF-8 flag is NOT set. Python's zipfile decodes using CP437 by default.
                                    # Try to "reverse" the CP437 decoding and re-decode as UTF-8.
                                    # This handles cases where macOS/Windows use UTF-8 but don't set the flag.
                                    try:
                                        corrected_filename = raw_filename_from_zip.encode('cp437').decode('utf-8')
                                        resolved_filename_str = corrected_filename
                                        logger.debug(f"ZIP member '{raw_filename_from_zip}' (no UTF-8 flag) re-interpreted from CP437 to UTF-8 as '{resolved_filename_str}'")
                                    except (UnicodeEncodeError, UnicodeDecodeError) as e_recode:
                                        logger.warning(f"Could not re-interpret ZIP member '{raw_filename_from_zip}' as UTF-8 from CP437; falling back to as-is from zipfile. Error: {e_recode}")
                                        resolved_filename_str = raw_filename_from_zip # Fallback to what zipfile gave
                                else:
                                    # UTF-8 flag IS set. Assume zipfile already decoded it as UTF-8.
                                    resolved_filename_str = raw_filename_from_zip
                                    logger.debug(f"ZIP member '{raw_filename_from_zip}' (UTF-8 flag set) used as-is from zipfile.")

                                # Always apply NFC normalization to the resolved string
                                member_filename_normalized = unicodedata.normalize('NFC', resolved_filename_str)
                                if member_filename_normalized != resolved_filename_str:
                                    logger.debug(f"Applied NFC normalization: '{resolved_filename_str}' -> '{member_filename_normalized}'")

                                # Skip macOS specific hidden files/folders, .DS_Store, and directory entries themselves
                                if member_filename_normalized.startswith('__MACOSX/') or \
                                   member_filename_normalized.endswith('.DS_Store') or \
                                   member.is_dir():
                                    logger.debug(f"Skipping ZIP member (post-normalization): {member_filename_normalized}")
                                    continue

                                # Construct full path for the extracted file within temp_dir
                                target_path = os.path.join(temp_dir, member_filename_normalized)
                                
                                # Ensure parent directory exists for the file, especially for nested structures
                                target_parent_dir = os.path.dirname(target_path)
                                if not os.path.exists(target_parent_dir):
                                    os.makedirs(target_parent_dir)
                                
                                # Extract file manually to the target path with the normalized name
                                with zip_ref.open(member) as source_file, open(target_path, "wb") as target_file_stream:
                                    target_file_stream.write(source_file.read())
                                logger.debug(f"Extracted '{member.filename}' to '{target_path}' with normalized name.")

                            except Exception as e_member:
                                logger.error(f"Error processing member {member.filename} from zip {original_filename}: {e_member}")
                                # Optionally, add info about this specific member error to processed_files_info
                                # processed_files_info.append({
                                #     "file_name": f"{original_filename}/{member.filename}", 
                                #     "status": f"Error extracting/normalizing member: {str(e_member)}", 
                                #     "doc_id": top_level_doc_id # Or a new one for the member if desired
                                # })
                                continue # Continue with other members

                    logger.info(f"Successfully extracted and normalized contents of zip file: {original_filename}")
                    
                    extracted_files_count = 0
                    # Walk through the extracted contents (os.walk will now see normalized names)
                    for root, _, extracted_filenames_in_walk in os.walk(temp_dir):
                        for extracted_filename_only in extracted_filenames_in_walk:
                            # The original zip file itself is not in the walk results of temp_dir's contents.
                            # (unless it was named like a regular file inside the zip, which is not standard)
                            
                            full_extracted_file_path = os.path.join(root, extracted_filename_only)
                            # Create a unique doc_name for metadata, showing its origin from the zip
                            # Relative path within the zip, using normalized names from the filesystem
                            relative_path_in_zip = os.path.relpath(full_extracted_file_path, temp_dir)
                            doc_name_for_metadata = relative_path_in_zip # Changed from f"{original_filename}/{relative_path_in_zip}"
                            sub_doc_id = str(uuid.uuid4()) # New ID for each file in zip
                            
                            try:
                                logger.info(f"Processing extracted file: {doc_name_for_metadata}")
                                documents = load_and_split_document(full_extracted_file_path, doc_id=sub_doc_id, original_doc_name=doc_name_for_metadata)
                                
                                if documents:
                                    add_documents_to_store(documents, effective_collection_name)
                                    processed_files_info.append({
                                        "file_name": doc_name_for_metadata, 
                                        "status": "Successfully processed and added from zip", 
                                        "doc_id": sub_doc_id
                                    })
                                    extracted_files_count += 1
                                else:
                                    logger.warning(f"No documents could be loaded from extracted file: {doc_name_for_metadata}")
                                    processed_files_info.append({
                                        "file_name": doc_name_for_metadata, 
                                        "status": "No content loaded or processed from zip component", 
                                        "doc_id": sub_doc_id
                                    })
                            except Exception as e:
                                logger.error(f"Error processing extracted file {doc_name_for_metadata}: {e}")
                                processed_files_info.append({
                                    "file_name": doc_name_for_metadata, 
                                    "status": f"Error processing from zip: {str(e)}", 
                                    "doc_id": sub_doc_id
                                })
                    if extracted_files_count == 0 and not processed_files_info:
                        # Only add this if no other info (e.g. errors for subfiles) has been added for this zip
                        processed_files_info.append({"file_name": original_filename, "status": "Zip file was empty or contained no processable files.", "doc_id": top_level_doc_id}) 
                
            except zipfile.BadZipFile:
                logger.error(f"Uploaded file {original_filename} is not a valid zip file or is corrupted.")
                processed_files_info.append({"file_name": original_filename, "status": "Invalid or corrupted zip file", "doc_id": top_level_doc_id})
            except Exception as e:
                logger.exception(f"An unexpected error occurred while processing zip file {original_filename}: {e}")
                processed_files_info.append({"file_name": original_filename, "status": f"Error processing zip: {str(e)}", "doc_id": top_level_doc_id})
        
        else: # Not a .zip file, process as a single file
            temp_file_path = ""
            try:
                # Use NamedTemporaryFile for single files as well for consistent temp handling
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(await file.read())
                    temp_file_path = temp_file.name
                
                logger.info(f"Processing single file: {original_filename}")
                documents = load_and_split_document(temp_file_path, doc_id=top_level_doc_id, original_doc_name=original_filename)
                
                if documents:
                    add_documents_to_store(documents, effective_collection_name)
                    processed_files_info.append({"file_name": original_filename, "status": "Successfully processed and added", "doc_id": top_level_doc_id})
                else:
                    logger.warning(f"No documents could be loaded from file: {original_filename}")
                    processed_files_info.append({"file_name": original_filename, "status": "No content loaded or processed", "doc_id": top_level_doc_id})
            
            except Exception as e:
                logger.exception(f"Error processing file {original_filename}: {e}")
                processed_files_info.append({"file_name": original_filename, "status": f"Error: {str(e)}", "doc_id": top_level_doc_id})
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path) # Clean up the temporary file

    if not processed_files_info:
        # This might happen if 'files' list was empty (though caught earlier) or all files failed in a way that didn't add to info.
        return JSONResponse(content={"message": "No files were processed or all failed early."}, status_code=400)
    
    return JSONResponse(content={
        "message": "File processing complete.",
        "processed_files": processed_files_info,
        "collection_name": effective_collection_name
    }, status_code=200)


@app.post("/search", response_model=SearchResponse)
async def search_endpoint_original(request: KeywordSearchRequest, collection_name: str = "master"):
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
            filter_dict=request.filter,
            collection_name=collection_name
        )
        
        # Convert LangChain Document results to our API model (score will be None)
        response_items = [SearchResultItem(page_content=doc.page_content, metadata=doc.metadata) for doc in results]
        
        logger.info(f"Original search completed. Found {len(response_items)} results.")
        return SearchResponse(results=response_items)
        
    except Exception as e:
        logger.exception(f"Error during original search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {e}")


@app.post("/search_score", response_model=SearchResponse)
async def search_endpoint_with_score(request: KeywordSearchRequest, collection_name: str = "master"):
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
            filter_dict=request.filter,
            collection_name=collection_name
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


@app.get("/list-documents", response_model=ListDocumentsResponse)
async def list_documents_endpoint(collection_name: str = "master"):
    """Lists the documents currently indexed by retrieving unique doc_ids and names from metadata."""
    logger.info(f"Request to list documents from collection: {collection_name}")
    try:
        retrieved_store = get_vector_store(collection_name)
        if not retrieved_store:
            logger.error(f"Vector store for collection '{collection_name}' not found.")
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")

        # Access the underlying Chroma collection to get metadata
        # This bypasses LangChain's standard retriever interface but is necessary for this specific task
        # Ensure the retrieved_store instance is the Chroma class instance
        if not isinstance(retrieved_store, Chroma):
             logger.error("Retrieved store is not a Chroma instance, cannot directly access collection.")
             raise HTTPException(status_code=500, detail="Internal configuration error: Vector store type mismatch.")

        chroma_collection = retrieved_store.get() # Gets all data {'ids', 'embeddings', 'documents', 'metadatas'} for the specific collection
        
        documents_dict: Dict[str, str] = {} # {doc_id: doc_name}
        if chroma_collection and chroma_collection.get('metadatas'):
            for metadata in chroma_collection['metadatas']:
                doc_id = metadata.get('doc_id')
                doc_name = metadata.get('doc_name')
                # Ensure we have both ID and name, and haven't seen this ID yet
                if doc_id and doc_name and doc_id not in documents_dict:
                    documents_dict[doc_id] = doc_name
        
        document_list = [DocumentInfo(doc_id=id, doc_name=name) for id, name in documents_dict.items()]
        
        logger.info(f"Found {len(document_list)} unique documents in '{collection_name}' by scanning metadata.")
        return ListDocumentsResponse(documents=document_list)

    except HTTPException as http_exc: # Re-raise HTTPExceptions to preserve status code and detail
        raise http_exc
    except Exception as e:
        logger.exception(f"Error listing documents from collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error listing documents: {e}")


@app.delete("/documents/{doc_id}", status_code=200)
async def delete_document_endpoint(doc_id: str, collection_name: Optional[str] = Query(None, description="The name of the collection to delete from. Defaults to the master collection.")):
    """
    Deletes a document and its associated chunks from the specified vector store collection by its `doc_id`.
    If `collection_name` is not provided, it defaults to the master collection.
    """
    effective_collection_name = collection_name if collection_name else config.MASTER_COLLECTION_NAME
    logger.info(f"Attempting to delete document with doc_id: {doc_id} from collection: {effective_collection_name}")

    try:
        # Ensure the collection is loaded/initialized, especially if it might be a new one or master.
        # get_vector_store will attempt to load or create it.
        # This call isn't strictly necessary for deletion if delete_document_by_id also calls it,
        # but can be useful for early feedback or ensuring the collection reference exists.
        get_vector_store(effective_collection_name) 
    except ValueError as e:
        logger.error(f"[Kss]: Error ensuring collection '{effective_collection_name}' exists before deletion: {e}")
        # Depending on desired behavior, might not want to raise here if delete_document_by_id handles it.
        # For now, just log it, as delete_document_by_id will also try to get/create.
        pass # Let delete_document_by_id handle it ultimately

    try:
        success = delete_document_by_id(doc_id, collection_name=effective_collection_name)
        if success:
            logger.info(f"Successfully deleted document with doc_id: {doc_id} from collection: {effective_collection_name}")
            return JSONResponse(content={"message": f"Document {doc_id} deleted successfully from {effective_collection_name}"}, status_code=200)
        else:
            logger.warning(f"Document with doc_id: {doc_id} not found in collection {effective_collection_name} or failed to delete.")
            return JSONResponse(content={"error": f"Document {doc_id} not found in collection {effective_collection_name} or failed to delete"}, status_code=404)
    except Exception as e:
        logger.exception(f"[Kss]: Unexpected error in delete_document_endpoint for doc_id {doc_id}: {e}")
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)


@app.get("/documents_ui", response_class=HTMLResponse)
async def documents_ui_endpoint(request: Request, collection_name: str = Query(default=config.MASTER_COLLECTION_NAME, description="The name of the collection to display documents from.")):
    """
    Provides a UI to view documents and their chunks stored in the vector database.
    Displays documents from the specified collection, defaulting to the master collection.
    """
    logger.info(f"Request for documents UI for collection: {collection_name}")
    try:
        # Ensure the collection exists, or handle gracefully if get_vector_store returns None
        current_store = get_vector_store(collection_name)
        if not current_store:
            logger.warning(f"Collection '{collection_name}' not found. Rendering UI with error.")
            # Render a basic error message within the template structure if possible,
            # or redirect/show a more generic error.
            # For now, pass an empty documents list and let the template handle it.
            return templates.TemplateResponse(
                "documents_ui.html", 
                {
                    "request": request, 
                    "documents": [], 
                    "current_collection": collection_name, 
                    "collections": config.ALL_COLLECTION_NAMES,
                    "error_message": f"Collection '{collection_name}' not found."
                }
            )

        chroma_collection_data = current_store.get(include=["metadatas", "documents"]) # Gets all data for the specific collection
        
        documents_for_ui = []
        if chroma_collection_data and chroma_collection_data.get('ids'):
            # Group chunks by doc_id
            doc_chunks_map: Dict[str, list] = {}
            doc_names_map: Dict[str, str] = {}

            for i, doc_id_chunk in enumerate(chroma_collection_data['ids']):
                # Assuming doc_id_chunk is the actual chunk_id (e.g., "doc_id_0", "doc_id_1")
                # And metadata contains the original doc_id and doc_name
                metadata = chroma_collection_data['metadatas'][i]
                original_doc_id = metadata.get('doc_id', 'Unknown_ID')
                doc_name = metadata.get('doc_name', 'Unknown Document')
                
                if original_doc_id not in doc_chunks_map:
                    doc_chunks_map[original_doc_id] = []
                    doc_names_map[original_doc_id] = doc_name
                
                doc_chunks_map[original_doc_id].append({
                    'chunk_id': doc_id_chunk, # This is the chunk's own ID in Chroma
                    'content': chroma_collection_data['documents'][i],
                    'metadata': metadata
                })
            
            for doc_id_original, chunks in doc_chunks_map.items():
                documents_for_ui.append({
                    'doc_id': doc_id_original,
                    'doc_name': doc_names_map[doc_id_original],
                    'chunks': chunks
                })

        return templates.TemplateResponse(
            "documents_ui.html", 
            {
                "request": request, 
                "documents": documents_for_ui, 
                "current_collection": collection_name,
                "collections": config.ALL_COLLECTION_NAMES # Pass all collection names for the dropdown
            }
        )
    except Exception as e:
        logger.exception(f"Error rendering documents UI for collection '{collection_name}': {e}")
        # Provide a user-friendly error page or message
        # For simplicity, returning a basic HTML response with the error
        error_content = f"<html><body><h1>Internal Server Error</h1><p>Failed to load documents for collection '{collection_name}'. Details: {str(e)}</p></body></html>"
        return HTMLResponse(content=error_content, status_code=500)


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for LangChain-based application...")
    # Ensure necessary environment variables or configurations are set if needed
    # e.g., for specific model downloads or API keys if used later.
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
