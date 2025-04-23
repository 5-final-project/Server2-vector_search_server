import os
import logging
from typing import List, Optional
import uuid

# LangChain components for loading and splitting
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredFileLoader # Generic loader, might require extra dependencies
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import config # Import config instance
# We don't directly use the vector store or embedder here anymore,
# the main app orchestrates loading, splitting, and adding to the store.

logger = logging.getLogger(__name__)

# --- Text Splitter Initialization ---
# Using RecursiveCharacterTextSplitter as a good default
# Consider KoreanTextSplitter from langchain_experimental for better Korean handling if needed later
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.LC_CHUNK_SIZE,
    chunk_overlap=config.LC_CHUNK_OVERLAP,
    # separators=["\n\n", "\n", " ", ""] # Default separators are usually fine
    length_function=len, # Use character length
    is_separator_regex=False,
)
logger.info(f"Initialized RecursiveCharacterTextSplitter with chunk_size={config.LC_CHUNK_SIZE}, chunk_overlap={config.LC_CHUNK_OVERLAP}")


# --- Document Loading Logic ---
def load_document(file_path: str) -> List[Document]:
    """Loads a document from the given file path using appropriate LangChain loader."""
    logger.info(f"Attempting to load document: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    loader = None

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            # UnstructuredWordDocumentLoader often works well
            loader = UnstructuredWordDocumentLoader(file_path, mode="elements") # Try "elements" mode
        elif file_extension in [".txt", ".md", ".markdown"]:
            loader = TextLoader(file_path, encoding='utf-8') # Specify encoding
        else:
            # Fallback to UnstructuredFileLoader for other types if needed
            # Note: UnstructuredFileLoader might require additional system dependencies
            # like libmagic, poppler, tesseract depending on the file type.
            logger.warning(f"Unsupported file extension '{file_extension}'. Attempting generic loader.")
            # loader = UnstructuredFileLoader(file_path, mode="elements") # Use with caution
            raise ValueError(f"Unsupported file extension: {file_extension}")

        if loader:
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} document sections from {file_path} using {type(loader).__name__}.")
            # Add file source metadata if not already present
            for doc in documents:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = os.path.basename(file_path)
            return documents
        else:
            # This case should ideally be caught by the ValueError above
            logger.error(f"No suitable loader found for file: {file_path}")
            return []

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except ImportError as ie:
         logger.error(f"Missing dependency for loading {file_extension}: {ie}. Please install required packages.")
         raise RuntimeError(f"Missing dependency for {file_extension}: {ie}")
    except Exception as e:
        logger.exception(f"Error loading document {file_path}: {e}")
        # Depending on desired behavior, return empty list or re-raise
        return []


# --- Document Splitting Logic ---
def split_documents(documents: List[Document]) -> List[Document]:
    """Splits a list of documents into smaller chunks using the configured text splitter."""
    if not documents:
        logger.warning("No documents provided for splitting.")
        return []
    
    logger.info(f"Splitting {len(documents)} document sections into smaller chunks...")
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks.")
    return split_docs


# --- Combined Processing Function (Example Usage Pattern) ---
def load_and_split_document(file_path: str, doc_id: Optional[str] = None) -> List[Document]:
    """Loads a document, splits it into chunks, and adds metadata."""
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    
    file_name = os.path.basename(file_path)
    
    # 1. Load document(s) from file
    loaded_docs = load_document(file_path)
    if not loaded_docs:
        return [] # Return empty if loading failed

    # 2. Split the loaded documents into chunks
    split_docs = split_documents(loaded_docs)

    # 3. Add common metadata (doc_id, doc_name) to each chunk
    for i, chunk in enumerate(split_docs):
        chunk.metadata['doc_id'] = doc_id
        chunk.metadata['doc_name'] = file_name
        chunk.metadata['chunk_index'] = i # Add chunk index

    logger.info(f"Processed '{file_name}' (ID: {doc_id}): Loaded {len(loaded_docs)} sections, split into {len(split_docs)} chunks.")
    return split_docs

# Note: The actual adding to the vector store is now handled separately,
# typically in the main application logic (app.py) after calling load_and_split_document.
