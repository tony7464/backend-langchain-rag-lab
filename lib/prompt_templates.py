"""Prompt template helpers for the LangChain RAG workflow."""

SYSTEM_PROMPT = """
You are an internal reliability assistant.

Answer using only the approved retrieved context provided by the backend.
If the context does not contain enough information to answer reliably, say that
you do not have enough approved context to answer. Do not invent policies,
procedures, metrics, or incident steps.

Keep the response concise, specific, and useful to an engineer during an incident.
"""

# The human message carries the per-request data. {context} and {question} are
# template variables that LangChain fills in when the chain is invoked.
HUMAN_PROMPT = """
Approved runbook context:
{context}

Engineer question:
{question}

Answer only from the approved context above. If the context does not support
an answer, say so instead of guessing. Do not add unsupported claims.
"""


def build_rag_prompt():
    """Build the reusable LangChain prompt template for RAG answers."""

    # Imported here so the module loads even before dependencies are installed.
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
