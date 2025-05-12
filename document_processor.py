import os
import logging
from typing import List, Optional
import uuid

# LangChain components for loading and splitting
from langchain_community.document_loaders import (
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredFileLoader # Generic loader, might require extra dependencies
)
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from text_spliter import KoreanSentenceSplitter
from langchain_core.documents import Document

from config import config # Import config instance
# We don't directly use the vector store or embedder here anymore,
# the main app orchestrates loading, splitting, and adding to the store.

logger = logging.getLogger(__name__)

# --- Text Splitter Initialization ---
# Using RecursiveCharacterTextSplitter as a good default
# Consider KoreanTextSplitter from langchain_experimental for better Korean handling if needed later
text_splitter = KoreanSentenceSplitter(
    chunk_size=config.LC_CHUNK_SIZE,
    chunk_overlap=config.LC_CHUNK_OVERLAP,
    # separators=["\n\n", "\n", " ", ""] # Default separators are usually fine
    length_function=len, # Use character length
    # is_separator_regex=False, # Removed, as False is the default in the parent class
)
logger.info(f"Initialized KoreanSentenceSplitter with chunk_size={config.LC_CHUNK_SIZE}, chunk_overlap={config.LC_CHUNK_OVERLAP}")


# --- Document Loading Logic ---
def load_document(file_path: str) -> List[Document]:
    """Loads a document from the given file path using appropriate LangChain loader."""
    logger.info(f"Attempting to load document: {file_path}")
    file_extension = os.path.splitext(file_path)[1].lower()
    loader = None

    try:
        if file_extension == ".pdf":
            loader = PDFMinerLoader(file_path)
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
SUPPORTED_FILE_TYPES = { # Added to ensure we can check against it
    ".pdf": PDFMinerLoader,
    ".txt": TextLoader,
    # ".csv": CSVLoader,
    # TODO: Add .doc and .docx with UnstructuredFileLoader if textൃact is installed
    # ".doc": UnstructuredFileLoader, 
    # ".docx": UnstructuredFileLoader,
}

def load_and_split_document(file_path: str, doc_id: str, original_doc_name: str) -> List[Document]: # Added original_doc_name
    """Loads a document from file_path, splits it into chunks, and adds metadata."""
    file_extension = os.path.splitext(file_path)[1].lower()
    loader_class = SUPPORTED_FILE_TYPES.get(file_extension)

    if not loader_class:
        logger.error(f"Unsupported file type: {file_extension} for file: {original_doc_name}")
        # Return empty list or raise specific error
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types are: {', '.join(SUPPORTED_FILE_TYPES.keys())}")

    try:
        loader = loader_class(file_path)
        documents = loader.load() # This typically returns a list of Document objects (often just one for these loaders)
    except Exception as e:
        logger.error(f"Error loading document {original_doc_name} with {loader_class.__name__}: {e}")
        raise RuntimeError(f"Failed to load document {original_doc_name}: {e}")

    if not documents:
        logger.warning(f"No documents loaded from {original_doc_name}. The file might be empty or unreadable.")
        return []

    # Initialize the text splitter
    text_splitter = KoreanSentenceSplitter(
        chunk_size=config.LC_CHUNK_SIZE,
        chunk_overlap=config.LC_CHUNK_OVERLAP,
        length_function=len,
        # is_separator_regex=False,
    )

    all_split_chunks = []
    for doc in documents: # Iterate if loader returns multiple documents (e.g. some CSV loaders)
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(chunks):
            # Create a new Document object for each chunk with enriched metadata
            chunk_metadata = doc.metadata.copy() # Start with existing metadata from the loader
            chunk_metadata.update({
                'doc_id': doc_id,  # Master document ID for all chunks of this file
                'doc_name': original_doc_name, # Original file name
                'chunk_index': i, # Index of this chunk within the document
                'file_path_source': file_path # Optional: path of the source file processed
            })
            all_split_chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    
    logger.info(f"Document '{original_doc_name}' (ID: {doc_id}) split into {len(all_split_chunks)} chunks.")
    return all_split_chunks

# Note: The actual adding to the vector store is now handled separately,
# typically in the main application logic (app.py) after calling load_and_split_document.
