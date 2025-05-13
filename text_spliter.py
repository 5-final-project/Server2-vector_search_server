from typing import List, Optional, Any, Sequence
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document
import kss
import logging

logger = logging.getLogger(__name__)

class KoreanSentenceSplitter(TextSplitter):
    """
    Splits text into sentences using KSS and then groups them into chunks
    respecting chunk size and overlap, inheriting from TextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        backend: str = 'auto', # KSS backend option
        length_function = len,
        keep_separator: bool = False, # Keep for TextSplitter compatibility
        add_start_index: bool = False, # Keep for TextSplitter compatibility
        strip_whitespace: bool = True, # Keep for TextSplitter compatibility
        **kwargs: Any, # Catch any other TextSplitter args
    ):
        """Initializes the KoreanSentenceSplitter.

        Args:
            chunk_size: Max size of chunks (in characters).
            chunk_overlap: Max overlap between chunks (in characters).
            backend: KSS backend ('mecab', 'pecab', 'fast', 'auto').
            length_function: Function to measure text length.
            keep_separator: Passed to TextSplitter (usually False for sentence splitting).
            add_start_index: Passed to TextSplitter.
            strip_whitespace: Passed to TextSplitter.
        """
        # Pass only valid TextSplitter arguments to super().__init__
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator, # KSS handles sentences, so separators aren't the primary mechanism
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
            **kwargs # Pass any other valid TextSplitter kwargs
        )
        # Store custom arguments
        self.backend = backend
        # Store chunk_size and chunk_overlap locally as superclass doesn't expose them easily
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def _split_text_with_kss(self, text: str) -> List[str]:
        """Splits text into sentences using KSS."""
        try:
            sentences = kss.split_sentences(text, backend=self.backend)
            # Remove empty strings that might result from splitting
            sentences = [s.strip() for s in sentences if s.strip()]
            logger.debug(f"KSS split into {len(sentences)} sentences.")
            return sentences
        except Exception as e:
            logger.error(f"Error splitting text with KSS (backend: {self.backend}): {e}", exc_info=True)
            # Fallback: split by newline if KSS fails
            logger.warning("Falling back to newline splitting due to KSS error.")
            sentences = text.split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences

    def split_text(self, text: str) -> List[str]:
        """Splits text by sentences and merges them into chunks."""
        sentences = self._split_text_with_kss(text)
        if not sentences:
            return []

        # Use the merge_splits logic adapted for sentences
        separator = " " # How to join sentences
        return self._merge_sentences(sentences, separator)

    def _merge_sentences(self, sentences: List[str], separator: str) -> List[str]:
        """Merges sentences into chunks respecting chunk_size and chunk_overlap."""
        chunks = []
        current_chunk_sentences: List[str] = []
        current_length = 0
        separator_len = self._length_function(separator)

        for i, sentence in enumerate(sentences):
            sentence_len = self._length_function(sentence)

            # Handle sentences longer than chunk_size
            if sentence_len > self._chunk_size:
                logger.warning(
                    f"Sentence starting with '{sentence[:80]}...' is longer "
                    f"({sentence_len} chars) than chunk size ({self._chunk_size}). "
                    "Splitting it naively."
                )
                # If a single sentence is too long, we might need to split it further
                # For now, we add it as its own chunk, potentially exceeding the limit.
                # A more robust approach would use a fallback splitter here.
                if current_chunk_sentences: # Add previous chunk first
                    chunks.append(separator.join(current_chunk_sentences))
                chunks.append(sentence)
                current_chunk_sentences = []
                current_length = 0
                continue

            # Check if adding the next sentence exceeds chunk_size
            potential_length = current_length + sentence_len + (separator_len if current_chunk_sentences else 0)

            if potential_length <= self._chunk_size:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_length = potential_length
            else:
                # Current chunk is full, finalize it
                if current_chunk_sentences:
                    chunks.append(separator.join(current_chunk_sentences))

                # Start new chunk, considering overlap
                # Find sentences from the end of the previous chunk to use as overlap
                overlap_sentences: List[str] = []
                overlap_length = 0
                # Iterate backwards from the current sentence's *predecessor*
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    prev_sentence = current_chunk_sentences[j]
                    prev_sentence_len = self._length_function(prev_sentence)
                    potential_overlap_len = overlap_length + prev_sentence_len + (separator_len if overlap_sentences else 0)

                    if potential_overlap_len <= self._chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence) # Add to beginning
                        overlap_length = potential_overlap_len
                    else:
                        # Stop if adding the next sentence exceeds overlap limit
                        break

                # Start the new chunk with the overlap sentences
                current_chunk_sentences = overlap_sentences
                current_length = overlap_length

                # Add the current sentence to the new chunk (if it fits with overlap)
                potential_length_with_overlap = current_length + sentence_len + (separator_len if current_chunk_sentences else 0)
                if potential_length_with_overlap <= self._chunk_size:
                     current_chunk_sentences.append(sentence)
                     current_length = potential_length_with_overlap
                else:
                     # Edge case: Even with overlap, the current sentence doesn't fit.
                     # Start a new chunk *only* with the current sentence.
                     # This might happen if overlap is large and the current sentence is also large.
                     # Finalize the overlap chunk first (if it exists)
                     if current_chunk_sentences:
                         chunks.append(separator.join(current_chunk_sentences))
                     current_chunk_sentences = [sentence]
                     current_length = sentence_len


        # Add the last remaining chunk
        if current_chunk_sentences:
            chunks.append(separator.join(current_chunk_sentences))

        # Optional: Post-process chunks (e.g., strip whitespace) if needed,
        # though TextSplitter's strip_whitespace might handle some cases.
        if self._strip_whitespace:
             chunks = [chunk.strip() for chunk in chunks]

        logger.info(f"Split text into {len(chunks)} chunks.")
        return chunks

    # Override split_documents to use our custom split_text
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits Documents by calling split_text."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        # Use create_documents from the parent class, which handles metadata correctly
        return self.create_documents(texts, metadatas=metadatas)
