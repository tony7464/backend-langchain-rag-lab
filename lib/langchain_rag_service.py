"""LangChain-supported RAG workflow service."""

from lib.config import CHAT_MODEL, DEFAULT_TOP_K
from lib.response_formatter import (
    format_fallback_response,
    format_langchain_debug,
    format_sources,
    format_success_response,
)
from lib.vector_store import retrieve_context


class LangChainServiceError(Exception):
    """Raised when the LangChain RAG service cannot complete a request."""


def build_chat_model():
    """Build the local chat model wrapper."""

    from langchain_ollama import ChatOllama

    # temperature=0 keeps answers as deterministic as possible, which suits
    # runbook guidance better than creative variation.
    return ChatOllama(model=CHAT_MODEL, temperature=0)


def build_chain():
    """Build the LangChain prompt to model to parser sequence."""

    # Imported inside the function so each call picks up the current objects
    # (this is also what lets the tests swap in fakes).
    from langchain_core.output_parsers import StrOutputParser

    from lib.prompt_templates import build_rag_prompt

    prompt_template = build_rag_prompt()
    llm = build_chat_model()

    # The | operator chains runnables: prompt fills variables -> model generates
    # a chat message -> parser turns that message into a plain string.
    return prompt_template | llm | StrOutputParser()


def has_usable_context(scored_documents):
    """Return True when retrieval produced at least one document with text."""

    for document, _distance in scored_documents:
        text = getattr(document, "page_content", "")
        if isinstance(text, str) and text.strip():
            return True
    return False


def format_context(scored_documents):
    """Format retrieved LangChain documents into prompt-ready context text."""

    blocks = []
    for index, (document, distance) in enumerate(scored_documents, start=1):
        metadata = (document.metadata or {})

        # Labeling each chunk with its metadata lets the model (and a human
        # debugging the prompt) see exactly which runbook each fact came from.
        blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"Source ID: {metadata.get('source_id', 'unknown')}",
                    f"Title: {metadata.get('title', 'Untitled')}",
                    f"Category: {metadata.get('category', 'Uncategorized')}",
                    f"Section: {metadata.get('section', 'Unspecified')}",
                    f"Chunk ID: {metadata.get('chunk_id', 'unknown')}",
                    f"Distance: {float(distance):.4f}",
                    f"Text: {(document.page_content or '').strip()}",
                ]
            )
        )

    return "\n\n".join(blocks)


def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the LangChain-supported RAG workflow for one validated question."""

    try:
        cleaned_question = question.strip()

        # 1. Retrieve: find the closest approved runbook chunks.
        scored_documents = retrieve_context(
            cleaned_question, vector_store=vector_store, top_k=top_k
        )

        # Blank chunks must never be cited or sent to the model.
        usable_documents = [
            (doc, distance)
            for doc, distance in scored_documents
            if isinstance(getattr(doc, "page_content", None), str)
            and doc.page_content.strip()
        ]

        # 2. Fallback: with no usable context, never call the model. Returning a
        #    fixed safe answer is how we avoid hallucinated incident steps.
        if not has_usable_context(scored_documents):
            debug = format_langchain_debug(
                scored_documents, context="", top_k=top_k, fallback=True
            )
            return format_fallback_response(debug)

        # 3. Build the context string that fills {context} in the prompt.
        context = format_context(usable_documents)

        # 4. Generate: use the injected chain (tests) or build the real one.
        if chain is None:
            chain = build_chain()
        answer = chain.invoke({"context": context, "question": cleaned_question})

        # 5. Verify: a blank answer is a failure, not a valid response.
        if not isinstance(answer, str) or not answer.strip():
            raise LangChainServiceError("The model returned an empty answer.")

        # 6. Attach sources and debug metadata so the answer is inspectable.
        sources = format_sources(usable_documents)
        debug = format_langchain_debug(
            usable_documents, context=context, top_k=top_k, fallback=False
        )
        return format_success_response(answer, sources, debug)

    except LangChainServiceError:
        # Already a service error (e.g. empty answer); don't wrap it twice.
        raise
    except Exception as exc:
        # Turn any lower-level failure (Chroma, Ollama, etc.) into one error type
        # the Flask route knows how to handle. "from exc" keeps the traceback.
        raise LangChainServiceError(f"LangChain RAG service failed: {exc}") from exc
