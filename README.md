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

1.  **Document Upload:** The user uploads a document (PDF, DOCX, TXT, or a ZIP containing these) via the FastAPI endpoint.
    - **ZIP File Handling**: 
        - If a ZIP file is uploaded, it's extracted in a temporary directory.
        - Each supported file within the ZIP is processed individually.
        - Handles Unicode filenames (e.g., Korean) correctly, addressing potential encoding issues (UTF-8 flag, CP437 re-interpretation, NFC normalization).
        - The document name (`doc_name`) for files extracted from a ZIP will be their path *within* the ZIP file (e.g., `folder/report.pdf`), not prefixed by the ZIP filename.
2.  **Processing (`document_processor.py`):**
    - LangChain's `Document Loaders` read the content:
        - **PDFs**: Uses `PDFMinerLoader` (`langchain_community.document_loaders.PDFMinerLoader`).
        - **DOCX**: Uses `UnstructuredWordDocumentLoader`.
        - **TXT**: Uses `TextLoader`.
    - `KoreanSentenceSplitter` (custom class in `text_spliter.py` using `kss==4.5.4`) divides the document into smaller, semantically meaningful Korean sentences, which are then merged into chunks. This version of `kss` resolves previous `TypeError` issues.
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
- **Document Management UI (`documents_ui`):** A web-based interface for viewing uploaded documents, their processing status, and potentially managing them.
- **Document Processing (`document_processor.py`, `text_spliter.py`):** 
    - Manages loading of different file types (PDFs with `PDFMinerLoader`, DOCX, TXT).
    - Handles ZIP file extraction and individual file processing, including robust Unicode filename decoding.
    - Splits text into chunks using `KoreanSentenceSplitter` (based on `kss==4.5.4`) for Korean-language sentence segmentation before chunking.
- **LangChain:** The core library facilitating the RAG pipeline, including:
  - **Document Loaders:** Interface for reading various file formats (e.g., `PDFMinerLoader`, `UnstructuredWordDocumentLoader`, `TextLoader`).
  - **Text Splitters:** `KoreanSentenceSplitter` provides sentence-aware chunking for Korean.
  - **Embeddings Interface:** Wrapper around the Hugging Face embedding model.
  - **Vector Store Interface:** Wrapper for interacting with ChromaDB.
- **ChromaDB (`vector_store.py`):** The vector database used for efficient storage and retrieval of high-dimensional embeddings. 
    - The system uses a `master` collection (configurable via `MASTER_COLLECTION_NAME`) for general document chunks.
    - It is designed to additionally support up to 5 specialized collections for different document categories or purposes, allowing for more granular data organization and search.
- **Hugging Face Model (`embedding.py`):** The transformer model used to generate text embeddings (specifically `BM-K/KoSimCSE-Unsup-RoBERTa` for Korean semantic understanding).
- **Configuration (`config.py`):** Manages application settings such as model names, database paths, chunking parameters, and collection names. Key environment variables (or `.env` file settings) include:
    - `MASTER_COLLECTION_NAME`: Name for the primary ChromaDB collection (default: `master`). Other specific collection names can be managed as needed.
    - `EMBED_MODEL_NAME`: Hugging Face model for embeddings.
    - `VECTOR_DB_PATH`: Base path to store all ChromaDB collection data (e.g., `./vector_db_collections`). The `master` collection and any specialized collections will reside as sub-directories within this path (e.g., `./vector_db_collections/master`, `./vector_db_collections/special_collection_1`).
    - `LC_CHUNK_SIZE`, `LC_CHUNK_OVERLAP`: Parameters for text splitting.
    - `SIMILARITY_THRESHOLD`: Minimum similarity score for `/search_score` results.
    - `SEARCH_K`: Default number of results to return for searches.

## Key Features

- **Document Upload:** Supports PDF, DOCX, and TXT file uploads with automatic processing and indexing.
- **Keyword Search:** Performs semantic search based on input keywords to find relevant document chunks.
- **Score-Based Filtering:** Calculates similarity scores and allows filtering results above a configurable threshold.
- **Document Listing:** Provides an endpoint to view the list of currently indexed documents.
- **Document Management UI:** Offers a web interface for document management.

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

5.  **Run the Application:**

    ```bash
    python app.py
    ```

    The server will start, typically accessible at `http://127.0.0.1:8000`.

6.  **Running with Docker:**

    The application is configured to run with Docker using `docker-compose.yml` located *inside* the `VectorDB` directory.

    a. **Navigate to the `VectorDB` directory:**
    ```bash
    cd VectorDB 
    ```

    b. **Build the Docker Image:**
    ```bash
    # Make sure Docker is running
    docker-compose build --no-cache
    ```

    c. **Run the Docker Container:**
    ```bash
    docker-compose up -d --force-recreate
    ```
    The application will be accessible at `http://localhost:8000`.

    **Important Docker Notes:**
    - The `docker-compose.yml` is configured to mount the current `VectorDB` directory (`.` or `./`) as `/app` in the container, allowing for live code reloading if `uvicorn` is started with `--reload`.
    - It also mounts `../uploads` (i.e., the `uploads` directory at the project root, sibling to `VectorDB`) to `/uploads` in the container. Ensure this `uploads` directory exists at the project root: `/Users/yun-ungsang/Desktop/FISA-Final-Project/uploads`.
    - **Vector Database Persistence:** The `Dockerfile` currently *does not* copy a pre-existing `vector_db_chroma_lc` into the image. To persist your vector database across container restarts, the `docker-compose.yml` implicitly relies on the fact that the `VECTOR_DB_BASE_PATH` (e.g., `./vector_db_collections`) is within the `/app` mount. This means data will be saved to your local `VectorDB/vector_db_collections` directory.

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

- **`GET /documents_ui`**
  - **Description:** Provides a web-based user interface for viewing and managing uploaded documents.
  - **Response:** HTML content for the UI.

- **`GET /documents`**
  - **Description:** Retrieves a list of all indexed documents (ID and name) from the `master` collection (or a specified collection if API supports it).
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
- **KSS (`kss==4.5.4`):** Korean sentence splitter.
- **PDFMiner.six (`pdfminer.six`):** Used by `PDFMinerLoader` for PDF parsing.
