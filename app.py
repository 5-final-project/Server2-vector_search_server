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
    document_exists, # Keep for now, or remove if not used elsewhere
    document_exists_globally # Import the new function
)
from document_processor import load_and_split_document # Import the processing function
from api_models import (
    KeywordSearchRequest,
    SearchResponse,
    ListDocumentsResponse,
    DocumentInfo,
    SearchResultItem,
    BatchUploadResponse, # Updated import
    FileUploadStatus # Updated import
)
from langchain_chroma import Chroma # Chroma import for type checking

# S3 Integration dependencies
from dotenv import load_dotenv
import boto3

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- AWS S3 Configuration and Client Initialization ---
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

s3_client = None
try:
    if AWS_ACCESS_KEY and AWS_SECRET_KEY and BUCKET_NAME and AWS_DEFAULT_REGION:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_DEFAULT_REGION
        )
        logger.info("S3 client initialized successfully.")
    else:
        logger.warning("AWS S3 credentials or configuration not fully set. S3 upload disabled.")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    s3_client = None # Ensure client is None if init fails

def _upload_file_to_s3(file_path: str, object_name: str) -> bool:
    """Helper function to upload a file to S3.

    Args:
        file_path (str): The local path to the file to upload.
        object_name (str): The desired object name (key) in S3.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    if not s3_client:
        logger.error("S3 client not available. Cannot upload file.")
        return False
    if not BUCKET_NAME:
        logger.error("S3 BUCKET_NAME not configured. Cannot upload file.")
        return False

    try:
        logger.info(f"Uploading {file_path} to S3 bucket '{BUCKET_NAME}' as '{object_name}'")
        s3_client.upload_file(file_path, BUCKET_NAME, object_name)
        logger.info(f"Successfully uploaded {file_path} to s3://{BUCKET_NAME}/{object_name}")
        return True
    except FileNotFoundError:
        logger.error(f"Local file not found for S3 upload: {file_path}")
        return False
    except Exception as e:
        # Catch potential boto3 exceptions specifically if needed, e.g., ClientError
        logger.error(f"Error uploading '{file_path}' to S3 as '{object_name}': {e}")
        return False

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

@app.post("/upload-document", response_model=BatchUploadResponse) # Use the new Pydantic model
async def upload_document_endpoint(files: List[UploadFile] = File(...), collection_name: Optional[str] = Query(None, description="The name of the collection to upload the document to. Defaults to the master collection if not provided.")):
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    effective_collection_name = collection_name if collection_name else config.MASTER_COLLECTION_NAME
    logger.info(f"Attempting to upload {len(files)} file(s) to collection: {effective_collection_name}")

    try:
        get_vector_store(effective_collection_name) # Ensure collection is accessible
    except ValueError as e:
        logger.error(f"Failed to access collection '{effective_collection_name}': {e}")
        raise HTTPException(status_code=404, detail=f"Collection '{effective_collection_name}' not found or not initialized.")

    processed_files_info: List[FileUploadStatus] = []

    for file in files:
        original_filename = file.filename
        file_status = FileUploadStatus(
            filename=original_filename,
            status="pending",
            vector_db_status="pending",
            s3_status="pending"
        )
        temp_file_path = None

        try: # Outer try block for processing a single file
            temp_file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}_{original_filename}")
            with open(temp_file_path, "wb") as buffer:
                buffer.write(await file.read())
            logger.info(f"File '{original_filename}' saved temporarily to '{temp_file_path}'")

            # Check if document exists globally
            if document_exists_globally(doc_name=original_filename):
                logger.warning(f"Document named '{original_filename}' already exists in one of the collections. Skipping Vector DB processing and S3 upload globally.")
                file_status.status = "skipped"
                file_status.vector_db_status = "skipped"
                file_status.s3_status = "skipped"
                file_status.message = f"Document with name '{original_filename}' already exists globally."
            else:
                # If it's a ZIP file, extract and process its contents
                if original_filename.lower().endswith('.zip'):
                    logger.info(f"'{original_filename}' is a ZIP file. Attempting to extract and process...")
                    file_status.extracted_files = []
                    all_extracted_successful_vdb = True
                    all_extracted_successful_s3 = True

                    with tempfile.TemporaryDirectory() as extract_dir:
                        try:
                            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                                for member in zip_ref.infolist():
                                    original_member_filename = member.filename # For logging in case of failure
                                    try:
                                        # Check for UTF-8 flag (bit 11 of general purpose bit flag)
                                        is_utf8 = member.flag_bits & 0x800
                                        
                                        if is_utf8:
                                            decoded_filename = member.filename
                                            # logger.debug(f"Filename '{original_member_filename}' is UTF-8 encoded.")
                                        else:
                                            # Not flagged as UTF-8, try CP437 -> UTF-8 then CP437 -> EUC-KR
                                            # logger.debug(f"Filename '{original_member_filename}' not flagged as UTF-8. Attempting CP437 decode.")
                                            try:
                                                decoded_filename = member.filename.encode('cp437').decode('utf-8')
                                            except UnicodeDecodeError:
                                                # logger.debug(f"CP437 -> UTF-8 failed for '{original_member_filename}'. Attempting CP437 -> EUC-KR.")
                                                decoded_filename = member.filename.encode('cp437').decode('euc-kr', 'ignore') # 'ignore' errors for broader compatibility
                                        
                                        # Normalize filename (NFC for macOS/Windows compatibility)
                                        member.filename = unicodedata.normalize('NFC', decoded_filename)
                                        # logger.debug(f"Normalized filename: '{member.filename}' from '{original_member_filename}'")

                                    except Exception as e_decode:
                                        # If any decoding/normalization step fails, log and use original filename
                                        logger.warning(f"Could not properly decode or normalize filename '{original_member_filename}': {e_decode}. Using it as is for extraction.")
                                        member.filename = original_member_filename # Fallback to original member filename

                                    zip_ref.extract(member, extract_dir)

                                logger.info(f"Successfully extracted '{original_filename}' to '{extract_dir}'.")

                                for extracted_filename_os in os.listdir(extract_dir):
                                    extracted_filename_relative = extracted_filename_os
                                    # Use only the extracted filename as the display name
                                    extracted_file_display_name = extracted_filename_relative
                                
                                    sub_file_status = FileUploadStatus(
                                        filename=extracted_file_display_name,
                                        status="pending",
                                        vector_db_status="pending",
                                        s3_status="pending"
                                    )

                                    if document_exists_globally(doc_name=extracted_file_display_name):
                                        logger.warning(f"Extracted document '{extracted_file_display_name}' already exists globally. Skipping.")
                                        sub_file_status.status = "skipped"
                                        sub_file_status.vector_db_status = "skipped"
                                        sub_file_status.s3_status = "skipped"
                                        sub_file_status.message = "Document already exists globally."
                                    else:
                                        extracted_doc_id = str(uuid.uuid4())
                                        sub_file_status.doc_id = extracted_doc_id # Assign doc_id to status
                                        documents: List[Document] = []
                                        try:
                                            documents = load_and_split_document(
                                                file_path=os.path.join(extract_dir, extracted_filename_relative),
                                                doc_id=extracted_doc_id,
                                                original_doc_name=extracted_file_display_name # Corrected argument name
                                            )
                                            if documents:
                                                add_documents_to_store(documents, target_collection_name=effective_collection_name)
                                                sub_file_status.vector_db_status = "success"
                                            else:
                                                sub_file_status.vector_db_status = "skipped"
                                                sub_file_status.message = "No content to process."
                                        except Exception as e_proc:
                                            all_extracted_successful_vdb = False
                                            sub_file_status.vector_db_status = "failed"
                                            sub_file_status.message = (sub_file_status.message + f" VDB Error: {str(e_proc)[:100]};" if sub_file_status.message else f"VDB Error: {str(e_proc)[:100]};")
                                            logger.error(f"Error processing extracted file '{extracted_file_display_name}' for VectorDB: {e_proc}")
                                    
                                        # S3 Upload for extracted file
                                        if s3_client:
                                            extracted_s3_object_name = f"{effective_collection_name}/{extracted_filename_relative}"
                                            try:
                                                with open(os.path.join(extract_dir, extracted_filename_relative), "rb") as f_obj_ext:
                                                    s3_client.upload_fileobj(f_obj_ext, BUCKET_NAME, extracted_s3_object_name)
                                                sub_file_status.s3_status = "success"
                                            except Exception as e_s3_ext:
                                                all_extracted_successful_s3 = False
                                                sub_file_status.s3_status = "failed"
                                                sub_file_status.message = (sub_file_status.message + f" S3 Error: {str(e_s3_ext)[:100]};" if sub_file_status.message else f"S3 Error: {str(e_s3_ext)[:100]};")
                                                logger.error(f"Failed to upload extracted file '{extracted_file_display_name}' to S3: {e_s3_ext}")
                                        else:
                                            sub_file_status.s3_status = "skipped"
                                            all_extracted_successful_s3 = False
                                            sub_file_status.message = (sub_file_status.message + " S3 client NA;" if sub_file_status.message else "S3 client NA;")
                                        
                                        # Determine overall status for sub-file
                                        if sub_file_status.vector_db_status == "success" and sub_file_status.s3_status == "success":
                                            sub_file_status.status = "success"
                                        elif sub_file_status.vector_db_status == "skipped" and sub_file_status.s3_status == "skipped":
                                            sub_file_status.status = "skipped"
                                        else:
                                            sub_file_status.status = "partial success" if sub_file_status.vector_db_status == "success" or sub_file_status.s3_status == "success" else "error"

                                    file_status.extracted_files.append(sub_file_status)
                        
                        except zipfile.BadZipFile:
                            logger.error(f"'{original_filename}' is not a valid ZIP file or is corrupted.")
                            file_status.status = "error"
                            file_status.message = "Invalid or corrupted ZIP file."
                            all_extracted_successful_vdb = False
                            all_extracted_successful_s3 = False
                        except Exception as e_zip:
                            logger.error(f"Error processing ZIP file '{original_filename}': {e_zip}")
                            file_status.status = "error"
                            file_status.message = f"ZIP processing error: {str(e_zip)[:100]}"
                            all_extracted_successful_vdb = False
                            all_extracted_successful_s3 = False
                    
                    # Determine overall status for the ZIP file based on its extracted contents and original file S3 upload
                    if not file_status.extracted_files:
                        file_status.status = "error" if "error" in file_status.status else "skipped"
                        file_status.vector_db_status = "failed" if file_status.status == "error" else "skipped"
                    elif all_extracted_successful_vdb and all_extracted_successful_s3:
                        file_status.status = "success"
                        file_status.vector_db_status = "success"
                    else:
                        file_status.status = "partial success"
                        if any(sf.vector_db_status == "success" for sf in file_status.extracted_files):
                            file_status.vector_db_status = "partial success"
                        elif all(sf.vector_db_status == "skipped" for sf in file_status.extracted_files):
                            file_status.vector_db_status = "skipped"
                        else:
                            file_status.vector_db_status = "failed"
                
                # For non-ZIP files
                else: 
                    current_doc_id = str(uuid.uuid4()) # Generate UUID for doc_id
                    file_status.doc_id = current_doc_id # Store doc_id in status
                    
                    # S3 Upload Logic for the single file, using UUID in the key
                    s3_upload_successful = False
                    if s3_client:
                        s3_object_name = f"{effective_collection_name}/{original_filename}"
                        try:
                            # We already have the temp_file_path from earlier
                            with open(temp_file_path, "rb") as f_obj:
                                s3_client.upload_fileobj(f_obj, BUCKET_NAME, s3_object_name)
                            logger.info(f"Single file '{original_filename}' uploaded to S3 bucket '{BUCKET_NAME}' as '{s3_object_name}'.")
                            s3_upload_successful = True
                            file_status.s3_status = "success"
                        except Exception as e_s3:
                            logger.error(f"Failed to upload single file '{original_filename}' to S3: {e_s3}")
                            file_status.s3_status = "failed"
                            file_status.message = (file_status.message + f" S3 upload failed: {e_s3};" if file_status.message else f"S3 upload failed: {e_s3};")
                    else:
                        logger.warning("S3 client not available. Skipping S3 upload for single file.")
                        file_status.s3_status = "skipped"
                        file_status.message = (file_status.message + " S3 client not available;" if file_status.message else "S3 client not available;")

                    # Vector DB Processing for single file
                    documents: List[Document] = []
                    try:
                        logger.info(f"Processing single file '{original_filename}' for Vector DB. Collection: {effective_collection_name}, Doc ID: {current_doc_id}")
                        documents = load_and_split_document(temp_file_path, current_doc_id, original_filename)
                        if documents:
                            add_documents_to_store(documents, target_collection_name=effective_collection_name)
                            file_status.vector_db_status = "success"
                        else:
                            logger.info(f"No documents to process in '{original_filename}'.")
                            file_status.vector_db_status = "skipped"
                            file_status.message = (file_status.message + " VDB: No content;" if file_status.message else "VDB: No content;")
                    except Exception as e:
                        logger.error(f"Error processing file '{original_filename}' for VectorDB: {e}")
                        file_status.vector_db_status = "failed"
                        file_status.message = (file_status.message + f" VDB Error: {str(e)[:100]};" if file_status.message else f"VDB Error: {str(e)[:100]};")
                    
                    # Final status update for single file
                    if file_status.vector_db_status == "success" and s3_upload_successful: 
                        file_status.status = "success"
                    elif file_status.vector_db_status == "failed" or file_status.s3_status == "failed":
                        file_status.status = "failed"
                    else: # Covers s3 skipped, vdb skipped (no content), etc.
                        file_status.status = "partial_failure"

        except HTTPException: # Re-raise HTTPExceptions to be handled by FastAPI
            raise
        except Exception as e_outer:
            logger.error(f"Unexpected error processing file '{original_filename}': {e_outer}", exc_info=True)
            file_status.status = "error"
            file_status.message = f"Unexpected error: {str(e_outer)[:100]}"
            if file_status.vector_db_status == "pending": file_status.vector_db_status = "failed"
            if file_status.s3_status == "pending": file_status.s3_status = "failed"
        finally: 
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Temporary file '{temp_file_path}' removed.")
                except OSError as e_remove:
                    logger.warning(f"Could not remove temporary file '{temp_file_path}': {e_remove}")
            processed_files_info.append(file_status) 

    logger.info(f"Finished processing all {len(files)} files for collection '{effective_collection_name}'.")
    return BatchUploadResponse(processed_files=processed_files_info)


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
    # Ensure necessary vector stores are initialized at startup
    initialize_vector_stores()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
