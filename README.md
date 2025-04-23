# LangChain 기반 키워드 문서 검색 시스템

## 개요

이 프로젝트는 FastAPI 웹 프레임워크와 LangChain 라이브러리를 사용하여 구축된 키워드 기반 문서 검색 시스템입니다. 사용자는 PDF, DOCX, TXT 등의 문서를 업로드할 수 있으며, 시스템은 이 문서들을 처리하여 벡터 데이터베이스(ChromaDB)에 저장합니다. 저장된 문서를 대상으로 키워드 기반의 의미론적 검색을 수행할 수 있으며, 유사도 점수를 기준으로 결과를 필터링하는 기능도 제공합니다.

## 아키텍처

```mermaid
graph TD
    subgraph "User Interaction"
        User(사용자) -- API Request --> FastAPI
    end

    subgraph "Application Layer (Python Modules)"
        FastAPI(app.py) -- Reads --> Config(config.py)
        FastAPI -- Uses --> APIModels(api_models.py)
        FastAPI -- Calls --> DocProcessor(document_processor.py)
        FastAPI -- Calls --> VectorStore(vector_store.py)

        DocProcessor -- Uses --> LangChainLoaders(LangChain Document Loaders)
        DocProcessor -- Uses --> LangChainSplitters(LangChain Text Splitters)

        VectorStore -- Uses --> EmbeddingModel(embedding.py)
        VectorStore -- Interacts with --> ChromaDB[(ChromaDB Vector Store)]

        EmbeddingModel -- Uses --> HFEmbeddings(LangChain HuggingFaceEmbeddings)
        HFEmbeddings -- Loads --> HFModel[Hugging Face Model\n(BM-K/KoSimCSE-Unsup-RoBERTa)]
    end

    subgraph "External Services / Data"
        ChromaDB -- Stores/Retrieves --> Embeddings & Metadata
        User -- Uploads --> UploadDir[/uploads/]
        ChromaDB -- Persists data --> VectorDBDir[/vector_db_chroma_lc/]
    end

    style User fill:#f9f,stroke:#333,stroke-width:2px
    style FastAPI fill:#ccf,stroke:#333,stroke-width:2px
    style ChromaDB fill:#cfc,stroke:#333,stroke-width:2px
    style HFModel fill:#ffc,stroke:#333,stroke-width:2px
```

- **FastAPI (`app.py`):** API 엔드포인트 제공 및 전체 워크플로우 조율.
- **LangChain:** 문서 로딩, 분할, 임베딩, 벡터 저장소 연동 등 핵심 RAG 파이프라인 구성.
  - **Document Loaders (`document_processor.py`):** PDF, DOCX, TXT 파일 로드.
  - **Text Splitters (`document_processor.py`):** 문서를 의미 있는 청크로 분할.
  - **Embeddings (`embedding.py`):** Hugging Face 모델(`BM-K/KoSimCSE-Unsup-RoBERTa`)을 사용하여 텍스트를 벡터로 변환.
  - **Vector Store (`vector_store.py`):** ChromaDB를 사용하여 벡터 및 메타데이터 저장/검색.
- **ChromaDB:** 벡터 데이터베이스.
- **Config (`config.py`):** 주요 설정값 관리.

## 주요 기능

- **문서 업로드:** PDF, DOCX, TXT 파일 업로드 및 자동 처리/인덱싱.
- **키워드 검색:** 입력된 키워드를 기반으로 의미적으로 유사한 문서 청크 검색.
- **점수 기반 검색:** 유사도 점수를 계산하고, 설정된 임계값 이상의 결과만 필터링하여 반환.
- **문서 목록 조회:** 현재 인덱싱된 문서 목록 확인.

## 설치 및 실행

1.  **저장소 복제:**

    ```bash
    git clone <repository_url>
    cd llm_search_db
    ```

2.  **가상 환경 생성 및 활성화:** (권장)

    ```bash
    python -m venv venv
    # Windows
    # venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **필요 라이브러리 설치:**

    ```bash
    pip install -r requirements.txt
    ```

    - **참고:** `unstructured` 라이브러리는 파일 유형에 따라 추가적인 시스템 라이브러리(예: `libmagic`, `poppler`) 설치가 필요할 수 있습니다. [unstructured 설치 가이드](https://unstructured-io.github.io/unstructured/installation/full_installation.html)를 참조하세요.

4.  **설정 확인 (`config.py`):** (선택 사항)

    - `EMBED_MODEL_NAME`: 사용할 Hugging Face 임베딩 모델 이름.
    - `VECTOR_DB_PATH`: ChromaDB 데이터 저장 경로.
    - `LC_CHUNK_SIZE`, `LC_CHUNK_OVERLAP`: 텍스트 분할 시 청크 크기 및 오버랩.
    - `SIMILARITY_THRESHOLD`: `/search_score` 엔드포인트에서 사용할 유사도 임계값.
    - `SEARCH_K`: 검색 시 반환할 기본 결과 수.

5.  **애플리케이션 실행:**
    ```bash
    python app.py
    ```
    서버가 `http://0.0.0.0:8000` (또는 `http://127.0.0.1:8000`)에서 실행됩니다.

## API 엔드포인트

FastAPI는 자동 API 문서(`http://127.0.0.1:8000/docs`)를 제공합니다.

- **`POST /upload-document`**

  - 설명: 문서를 업로드하고 처리하여 벡터 저장소에 추가합니다.
  - 요청 본문: `multipart/form-data` 형식의 파일 (`file` 필드).
  - 성공 응답 (200):
    ```json
    {
      "doc_id": "generated-uuid",
      "doc_name": "uploaded_file.pdf",
      "num_chunks_added": 15
    }
    ```

- **`POST /search`**

  - 설명: 키워드로 문서를 검색합니다 (점수 필터링 없음).
  - 요청 본문:
    ```json
    {
      "keywords": ["검색어1", "검색어2"],
      "k": 5,
      "filter": null
    }
    ```
    또는 (필터링 시)
    ```json
    {
      "keywords": ["검색어"],
      "k": 3,
      "filter": { "doc_name": { "$eq": "specific_doc.pdf" } }
    }
    ```
  - 성공 응답 (200):
    ```json
    {
      "results": [
        {
          "page_content": "문서 청크 내용...",
          "metadata": {"source": "file.pdf", "doc_id": "...", "doc_name": "...", "chunk_index": 0},
          "score": null
        },
        ...
      ]
    }
    ```

- **`POST /search_score`**

  - 설명: 키워드로 문서를 검색하고, 설정된 유사도 임계값(`SIMILARITY_THRESHOLD`) 이상인 결과만 점수와 함께 반환합니다.
  - 요청 본문: `/search`와 동일.
  - 성공 응답 (200):
    ```json
    {
      "results": [
        {
          "page_content": "문서 청크 내용...",
          "metadata": {"source": "file.pdf", ...},
          "score": 0.85
        },
        ...
      ]
    }
    ```
    (임계값 미달 시 `results`는 빈 리스트 `[]`가 됩니다.)

- **`GET /documents`**
  - 설명: 현재 인덱싱된 문서 목록(ID 및 이름)을 반환합니다.
  - 성공 응답 (200):
    ```json
    {
      "documents": [
        { "doc_id": "uuid-1", "doc_name": "file1.pdf" },
        { "doc_id": "uuid-2", "doc_name": "file2.docx" }
      ]
    }
    ```

## 기술 스택

- Python 3.10+
- FastAPI: 웹 프레임워크
- LangChain: RAG 파이프라인 구축 (문서 로딩, 분할, 임베딩, 벡터 저장소 연동)
- ChromaDB: 벡터 데이터베이스
- Hugging Face Transformers & Sentence Transformers: 임베딩 모델 로딩 및 사용
- Uvicorn: ASGI 서버
