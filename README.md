# LangChain-Based Keyword Document Search System

## Overview

This project implements a keyword-based document search system built using the FastAPI web framework and the LangChain library. Users can upload documents (PDF, DOCX, TXT), which are then processed and stored in a vector database (ChromaDB). The system allows for semantic keyword searching within these documents and provides functionality to filter results based on similarity scores.

## Architecture

The system follows a Retrieval-Augmented Generation (RAG) pattern, leveraging vector embeddings for semantic search.

```mermaid
graph TD
    subgraph "User Interaction"
        User -- API Request --> AppServer[FastAPI Server (app.py)]
    end

    subgraph "Application Layer (Python Modules)"
        AppServer -- Reads Config --> Config(config.py)
        AppServer -- Uses Models --> APIModels(api_models.py)
        AppServer -- Handles Uploads & Search --> SearchEngine(search_engine.py)

        SearchEngine -- Processes Docs --> DocProcessor(document_processor.py)
        SearchEngine -- Interacts with DB --> VectorStore(vector_store.py)

        DocProcessor -- Loads Docs --> LangChainLoaders[LangChain Document Loaders]
        DocProcessor -- Splits Text --> LangChainSplitters[LangChain Text Splitters]

        VectorStore -- Generates Embeddings --> EmbeddingModel(embedding.py)
        VectorStore -- Stores/Retrieves --> ChromaDB[(ChromaDB Vector Store)]

        EmbeddingModel -- Uses --> HFEmbeddings[LangChain HuggingFaceEmbeddings]
        HFEmbeddings -- Loads Model --> HFModel[Hugging Face Model<br>(BM-K/KoSimCSE-Unsup-RoBERTa)]
    end

    subgraph "Data & Storage"
        User -- Uploads Files --> UploadDir[/uploads/]
        ChromaDB -- Persists Data --> VectorDBDir[/vector_db_chroma_lc/]
        ChromaDB -- Stores --> Embeddings & Metadata
    end

    style User fill:#f9f,stroke:#333,stroke-width:2px
    style AppServer fill:#ccf,stroke:#333,stroke-width:2px
    style ChromaDB fill:#cfc,stroke:#333,stroke-width:2px
    style HFModel fill:#ffc,stroke:#333,stroke-width:2px
```

**Workflow:**

1.  **Document Upload:** The user uploads a document via the FastAPI endpoint.
2.  **Processing (`document_processor.py`):**
    - LangChain's `Document Loaders` read the content from PDF, DOCX, or TXT files.
    - `Text Splitters` divide the document into smaller, semantically meaningful chunks.
3.  **Embedding (`embedding.py`):**
    - A pre-trained Hugging Face model (`BM-K/KoSimCSE-Unsup-RoBERTa` by default) via `LangChain HuggingFaceEmbeddings` converts each text chunk into a numerical vector (embedding).
4.  **Indexing (`vector_store.py`):**
    - The generated embeddings and associated metadata (like source document ID, name, chunk index) are stored in the ChromaDB vector store.
5.  **Search Request:** The user sends a search query (keywords) via the API.
6.  **Query Embedding:** The search keywords are converted into an embedding using the same Hugging Face model.
7.  **Similarity Search (`vector_store.py`):**
    - ChromaDB performs a similarity search (e.g., cosine similarity) between the query embedding and the stored document chunk embeddings.
8.  **Results:** The system returns the document chunks most similar to the query, potentially filtered by a similarity score threshold.

**Key Components:**

- **FastAPI (`app.py`, `api_models.py`):** Provides the web API endpoints for document upload, search, and listing. Handles request/response validation using Pydantic models.
- **Search Engine (`search_engine.py`):** Orchestrates the core logic for document processing, indexing, and searching, coordinating between the document processor and vector store.
- **LangChain:** The core library facilitating the RAG pipeline, including:
  - **Document Loaders:** Interface for reading various file formats.
  - **Text Splitters:** Algorithms for chunking documents effectively.
  - **Embeddings Interface:** Wrapper around the Hugging Face embedding model.
  - **Vector Store Interface:** Wrapper for interacting with ChromaDB.
- **ChromaDB (`vector_store.py`):** The vector database used for efficient storage and retrieval of high-dimensional embeddings.
- **Hugging Face Model (`embedding.py`):** The transformer model used to generate text embeddings (specifically `BM-K/KoSimCSE-Unsup-RoBERTa` for Korean semantic understanding).
- **Configuration (`config.py`):** Manages settings like model names, database paths, chunking parameters, and search thresholds.

## Key Features

- **Document Upload:** Supports PDF, DOCX, and TXT file uploads with automatic processing and indexing.
- **Keyword Search:** Performs semantic search based on input keywords to find relevant document chunks.
- **Score-Based Filtering:** Calculates similarity scores and allows filtering results above a configurable threshold.
- **Document Listing:** Provides an endpoint to view the list of currently indexed documents.

## Installation and Setup

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd llm_search_db
    ```

2.  **Create and Activate a Virtual Environment:** (Recommended)

    ```bash
    python -m venv venv
    # Windows
    # venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    - **Note:** The `unstructured` library might require additional system dependencies (like `libmagic`, `poppler`) depending on the file types you intend to process. Refer to the [official unstructured installation guide](https://unstructured-io.github.io/unstructured/installation/full_installation.html) for details.

4.  **Review Configuration (`config.py`):** (Optional)

    - `EMBED_MODEL_NAME`: Hugging Face model for embeddings.
    - `VECTOR_DB_PATH`: Path to store ChromaDB data.
    - `LC_CHUNK_SIZE`, `LC_CHUNK_OVERLAP`: Parameters for text splitting.
    - `SIMILARITY_THRESHOLD`: Minimum similarity score for `/search_score` results.
    - `SEARCH_K`: Default number of results to return for searches.

5.  **Run the Application:**

    ```bash
    python app.py
    ```

    The server will start, typically accessible at `http://127.0.0.1:8000`.

6.  **Running with Docker:** (Optional)

    Alternatively, you can build and run the application using Docker:

    a. **Build the Docker Image:**
    `bash
    # Make sure Docker is running
    docker build -t llm_search_db . 
    # Replace 'llm_search_db' with your preferred image name
    `

    b. **Run the Docker Container:**
    ```bash # Basic run (data is ephemeral)
    docker run -p 8000:8000 llm_search_db

        # Run with persistent vector database storage
        # This mounts the local 'vector_db_chroma_lc' directory into the container
        # Ensure the local directory exists or adjust the path as needed
        # docker run -p 8000:8000 -v ./vector_db_chroma_lc:/app/vector_db_chroma_lc llm_search_db
        ```
        The application will be accessible at `http://localhost:8000`.

## API Endpoints

FastAPI provides interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

- **`POST /upload-document`**

  - **Description:** Uploads, processes, and indexes a document.
  - **Request Body:** `multipart/form-data` with a `file` field containing the document.
  - **Success Response (200):**
    ```json
    {
      "doc_id": "generated-uuid",
      "doc_name": "uploaded_file.pdf",
      "num_chunks_added": 15
    }
    ```

- **`POST /search`**

  - **Description:** Searches for documents based on keywords without score filtering. Supports optional metadata filtering.
  - **Request Body:**
    ```json
    {
      "keywords": ["search term1", "search term2"],
      "k": 5, // Optional: Number of results (defaults to config.SEARCH_K)
      "filter": null // Optional: ChromaDB metadata filter (e.g., {"doc_name": {"$eq": "specific.pdf"}})
    }
    ```
  - **Success Response (200):**
    ```json
    {
      "results": [
        {
          "page_content": "Content of the document chunk...",
          "metadata": {
            "source": "file.pdf",
            "doc_id": "...",
            "doc_name": "...",
            "chunk_index": 0
          },
          "score": null // Score is null for this endpoint
        }
        // ... more results
      ]
    }
    ```

- **`POST /search_score`**

  - **Description:** Searches documents by keywords and returns results exceeding the `SIMILARITY_THRESHOLD` along with their scores.
  - **Request Body:** Same as `/search`.
  - **Success Response (200):**
    ```json
    {
      "results": [
        {
          "page_content": "Content of the relevant document chunk...",
          "metadata": {"source": "file.pdf", ...},
          "score": 0.85 // Similarity score
        },
        // ... more results above the threshold
      ]
    }
    ```
    (Returns `{"results": []}` if no chunks meet the threshold).

- **`GET /documents`**
  - **Description:** Retrieves a list of all indexed documents (ID and name).
  - **Success Response (200):**
    ```json
    {
      "documents": [
        { "doc_id": "uuid-1", "doc_name": "file1.pdf" },
        { "doc_id": "uuid-2", "doc_name": "file2.docx" }
      ]
    }
    ```

## Technology Stack

- **Python:** 3.10+
- **FastAPI:** High-performance web framework for building APIs.
- **LangChain:** Framework for developing applications powered by language models, used here for the RAG pipeline.
- **ChromaDB:** Open-source vector database for embedding storage and search.
- **Hugging Face Transformers & Sentence Transformers:** Libraries for loading and using the embedding model.
- **Uvicorn:** ASGI server to run the FastAPI application.
- **Unstructured:** Library for parsing various document formats.
