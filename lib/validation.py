"""Request validation helpers for POST /api/ask."""

from lib.config import MAX_QUESTION_LENGTH, MIN_QUESTION_LENGTH


def _error(error, message):
    """Build the (None, error_dict) tuple returned for every invalid payload."""

    return None, {"error": error, "message": message}


def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """

    # The body must be a JSON object like {"question": "..."}. This also catches
    # None, which is what Flask's get_json(silent=True) returns for bad JSON.
    if not isinstance(payload, dict):
        return _error("invalid_request", "Request body must be a JSON object.")

    if "question" not in payload:
        return _error("missing_question", "The 'question' field is required.")

    question = payload["question"]
    if not isinstance(question, str):
        return _error("invalid_question", "The 'question' field must be a string.")

    # Strip first so whitespace-only input counts as blank and length checks
    # measure the real question, not the padding around it.
    question = question.strip()

    if not question:
        return _error("empty_question", "The 'question' field cannot be blank.")

    if len(question) < MIN_QUESTION_LENGTH:
        return _error(
            "short_question",
            f"The question must be at least {MIN_QUESTION_LENGTH} characters long.",
        )

    if len(question) > MAX_QUESTION_LENGTH:
        return _error(
            "long_question",
            f"The question must be {MAX_QUESTION_LENGTH} characters or fewer.",
        )

    return question, None
