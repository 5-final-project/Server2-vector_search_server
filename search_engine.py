import logging
from typing import List, Dict, Any, Union

from config import config # Import config instance
# Embedder is used via vector_store now
from vector_store import vector_db # Import vector_db instance

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, vector_db_instance):
        self.vector_db = vector_db_instance
        # Embedder is accessed via vector_db if needed, or directly if required elsewhere

    def search_by_keywords(self, keywords: Union[List[str], str], k: int = 5):
        """Perform keyword-based search using the vector store."""
        query = " ".join(keywords) if isinstance(keywords, list) else keywords
        logger.info(f"Searching for keywords: '{query}' with k={k}")

        try:
            # Perform initial vector search
            # The vector_db.search method already handles embedding the query
            candidate_results = self.vector_db.search(query_text=query, k=k * 2) # Fetch more candidates for reranking

            if not candidate_results:
                logger.info("No initial results found from vector search.")
                return []

            # Group results by document ID for potential document-level relevance
            doc_results = {}
            for result in candidate_results:
                doc_id = result['metadata'].get('doc_id')
                if doc_id:
                    if doc_id not in doc_results:
                        doc_results[doc_id] = []
                    doc_results[doc_id].append(result)

            # Rerank results (optional, can be refined)
            # The current reranking logic relies heavily on metadata that might need adjustment
            # For simplicity, let's start with the direct results from ChromaDB search,
            # as ChromaDB's search itself is quite effective.
            # We can re-introduce reranking later if needed.
            # reranked_results = self.rerank_results(query, keywords, doc_results)

            # For now, return the top k results directly from the search
            # Sort by score just in case ChromaDB doesn't guarantee order perfectly
            # (though it usually does)
            sorted_results = sorted(candidate_results, key=lambda x: x['score'], reverse=True)

            logger.info(f"Found {len(sorted_results)} potential results, returning top {k}.")
            return sorted_results[:k]

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    # Reranking logic can be added back here if necessary.
    # It would need access to chunk text/metadata, potentially requiring
    # fetching more data using vector_db.get_metadata(chunk_id).
    # def rerank_results(self, query, keywords, doc_results): ...

# Instantiate the search engine
search_engine = SearchEngine(vector_db)
