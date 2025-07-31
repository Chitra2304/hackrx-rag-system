import spacy
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_lg")

def anonymize_text(text):
    """Anonymize sensitive information in text while preserving key insurance terms."""
    try:
        if not text or not text.strip():
            logger.error("Empty text provided for anonymization")
            return text
        
        # Protect insurance-specific terms
        protected_terms = [
            "hospitalization", "treatment", "surgery", "insured", "sum insured",
            "waiting period", "policy", "claim", "pre-approval", "procedure"
        ]
        placeholders = {}
        for i, term in enumerate(protected_terms):
            placeholder = f"__PROTECTED_{i}__"
            placeholders[placeholder] = term
            text = text.replace(term, placeholder, -1)
        
        doc = nlp(text)
        anonymized_text = text
        
        # Replace sensitive entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "NORP"]:
                anonymized_text = anonymized_text.replace(ent.text, "[REDACTED]")
        
        # Replace email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        anonymized_text = re.sub(email_pattern, "[REDACTED_EMAIL]", anonymized_text)
        
        # Replace phone numbers
        phone_pattern = r'\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        anonymized_text = re.sub(phone_pattern, "[REDACTED_PHONE]", anonymized_text)
        
        # Restore protected terms
        for placeholder, term in placeholders.items():
            anonymized_text = anonymized_text.replace(placeholder, term)
        
        logger.info("Text anonymized successfully")
        return anonymized_text
    except Exception as e:
        logger.error(f"Anonymization error: {str(e)}", exc_info=True)
        return text