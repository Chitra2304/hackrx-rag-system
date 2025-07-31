import logging

logger = logging.getLogger(__name__)

def generate_response(decision):
    """Format decision as JSON."""
    try:
        response = {
            "decision": decision["decision"],
            "amount": decision["amount"],
            "justification": decision["justification"],
            "clauses": decision["clauses"]
        }
        logger.info(f"Generated response: {response['decision']}")
        return response
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        raise