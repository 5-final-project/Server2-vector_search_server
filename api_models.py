from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

# --- Request Models ---

class KeywordSearchRequest(BaseModel):
    keywords: Union[List[str], str]
    k: int = 5 # Optional parameter for number of results
    filter: Optional[Dict[str, Any]] = None # Optional metadata filter

# --- Response Models ---

class DocumentUploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    num_chunks_added: int

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
