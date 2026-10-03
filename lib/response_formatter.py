"""Response formatting helpers for the LangChain RAG API."""

CHAIN_EXPRESSION = "ChatPromptTemplate | ChatOllama | StrOutputParser"

COMPONENTS = {
    "vector_store": "Chroma",
    "retrieval_method": "similarity_search_with_score",
    "prompt_template": "ChatPromptTemplate",
    "chat_model": "ChatOllama",
    "output_parser": "StrOutputParser",
}

SCORE_TYPE = "Chroma distance; lower usually means closer in this lesson setup"

FALLBACK_ANSWER = (
    "I do not have enough approved runbook context to answer that reliably."
)

# Defaults used when a retrieved chunk is missing a metadata field, so the API
# always returns the same keys even if the data is incomplete.
SOURCE_DEFAULTS = {
    "source_id": "unknown",
    "title": "Untitled",
    "category": "Uncategorized",
    "section": "Unspecified",
    "chunk_id": "unknown",
}


def format_sources(scored_documents):
    """Format retrieved documents as source metadata for the API response."""

    sources = []
    for document, distance in scored_documents:
        metadata = document.metadata or {}

        # Copy only the citation fields. The chunk text (page_content) is left
        # out on purpose: sources show *where* the answer came from, not the
        # full runbook text.
        source = {
            key: metadata.get(key) or default
            for key, default in SOURCE_DEFAULTS.items()
        }
        source["distance"] = round(float(distance), 4)
        sources.append(source)

    return sources


def format_langchain_debug(scored_documents, context, top_k, fallback):
    """Return LangChain debug metadata for inspectability."""

    return {
        "chain_expression": CHAIN_EXPRESSION,
        # Copy so callers can't accidentally mutate the shared constant.
        "components": dict(COMPONENTS),
        "retrieved_count": len(scored_documents),
        "retrieved_chunk_ids": [
            (document.metadata or {}).get("chunk_id", SOURCE_DEFAULTS["chunk_id"])
            for document, _distance in scored_documents
        ],
        "top_k": top_k,
        "context_characters": len(context),
        "fallback": fallback,
        "score_type": SCORE_TYPE,
    }


def format_success_response(answer, sources, debug):
    """Format a successful RAG response."""

    return {
        "answer": answer.strip(),
        "sources": sources,
        "langchain": debug,
    }


def format_fallback_response(debug):
    """Format a safe response when no usable context is available."""

    # Same shape as a success response, so clients don't need special parsing.
    # Empty sources make it clear nothing approved backed this answer.
    return format_success_response(FALLBACK_ANSWER, [], debug)


def format_error_response(error, message):
    """Format an API error response."""

    return {"error": error, "message": message}
