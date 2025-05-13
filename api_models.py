from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

# --- Request Models ---

class KeywordSearchRequest(BaseModel):
    keywords: Union[List[str], str]
    k: int = 5 # Optional parameter for number of results
    filter: Optional[Dict[str, Any]] = None # Optional metadata filter

# --- Response Models ---

# class DocumentUploadResponse(BaseModel): # Replaced by BatchUploadResponse
#     doc_id: str
#     doc_name: str
#     num_chunks_added: int

class FileUploadStatus(BaseModel):
    """Represents the processing status of a single uploaded file (or a file within a zip)."""
    filename: str
    doc_id: Optional[str] = None # ID assigned to the doc (top-level or sub-doc)
    status: str # e.g., "success", "error", "skipped", "partial success"
    vector_db_status: str # e.g., "success", "failed", "skipped", "pending", "not attempted"
    s3_status: str # e.g., "success", "failed", "skipped", "pending", "not attempted"
    message: Optional[str] = None # Error message or additional info
    extracted_files: Optional[List['FileUploadStatus']] = None # For zip files, list status of contained files

# Optional: If using Python < 3.9, you might need this for forward reference resolution
# FileUploadStatus.update_forward_refs()

class BatchUploadResponse(BaseModel):
    """Response model for the batch upload endpoint.
    Contains a list of statuses for each file processed.
    """
    processed_files: List[FileUploadStatus]

class SearchResultItem(BaseModel):
    """Represents a single search result chunk (LangChain Document) with score."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None # Add score field back

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class DocumentInfo(BaseModel):
    """Basic info about an indexed document."""
    doc_id: str
    doc_name: str
    # Could add chunk count or other info if easily retrievable

class ListDocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
