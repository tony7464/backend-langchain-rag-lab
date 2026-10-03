"""Vector store and retrieval helpers."""

from lib.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
)


def build_embeddings():
    """Build the local Ollama embeddings object used by Chroma."""

    from langchain_ollama import OllamaEmbeddings

    # The same embedding model must be used for seeding and for querying,
    # otherwise the question vectors won't be comparable to the stored chunks.
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store():
    """Build the Chroma vector store used by the RAG pipeline."""

    from langchain_chroma import Chroma

    # Points at the persisted collection created by seed_chroma.py.
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=build_embeddings(),
    )


def retrieve_context(question, *, vector_store=None, top_k=DEFAULT_TOP_K):
    """Retrieve scored documents for a question.

    Returns a list of (document, distance) tuples.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("A non-blank question is required for retrieval.")

    # Tests pass in a fake store; real requests build Chroma on demand. This
    # "dependency injection" keeps the tests fast and free of Ollama/Chroma.
    if vector_store is None:
        vector_store = build_vector_store()

    # Scores come back alongside documents so we can report distance in sources.
    return vector_store.similarity_search_with_score(question.strip(), k=top_k)
