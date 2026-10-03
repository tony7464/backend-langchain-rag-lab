"""Flask API for the LangChain RAG lab."""

from flask import Flask, jsonify, request

from lib.langchain_rag_service import LangChainServiceError, answer_question
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed LangChain RAG response."""

        # silent=True returns None instead of raising on missing/invalid JSON,
        # so the validator can answer with our own structured 400 error.
        payload = request.get_json(silent=True)

        question, error = validate_question_payload(payload)
        if error:
            return jsonify(error), 400

        # The route only handles HTTP concerns. Retrieval, prompting, and the
        # model call all live in the service layer.
        try:
            response = answer_question(question)
        except LangChainServiceError as exc:
            # 502 Bad Gateway: our API is fine, but an upstream dependency
            # (vector store or model) failed.
            error = format_error_response("langchain_service_error", str(exc))
            return jsonify(error), 502

        return jsonify(response), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
