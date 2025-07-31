import spacy
import re
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_lg")

# Lazy initialization of SentenceTransformer
_model = None

def get_model():
    """Load SentenceTransformer lazily."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model...")
        _model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return _model

def parse_query(query):
    """Parse query to extract dynamic entities and generate embedding."""
    try:
        if not query.strip():
            logger.error("Empty query provided")
            raise ValueError("Query cannot be empty")
        doc = nlp(query)
        entities = {}
        
        # Extract SpaCy entities
        for ent in doc.ents:
            entities[ent.label_.lower()] = ent.text
        
        # Extract insurance-specific terms
        age_match = re.search(r"(\d+)(M|F)", query, re.IGNORECASE)
        if age_match:
            entities["age"] = f"{age_match.group(1)}{age_match.group(2).upper()}"
        
        procedure_match = re.search(r"\b(\w+ectomy|surgery|operation|[\w\s]+?\s*(surgery|operation))\b", query, re.IGNORECASE)
        if procedure_match:
            entities["procedure"] = procedure_match.group(0).lower()
        
        duration_match = re.search(r"(\d+\s*(month|year)\s*(policy)?)", query, re.IGNORECASE)
        if duration_match:
            entities["policy_duration"] = duration_match.group(0).lower().replace(" policy", "")
        
        if "pre-approval" in query.lower() or "pre-approved" in query.lower():
            entities["pre_approval"] = True
        
        # Extract key-value pairs
        kv_match = re.findall(r"(\w+):\s*([^,]+?)(?:,|$)", query, re.IGNORECASE)
        for key, value in kv_match:
            entities[key.lower()] = value.strip()
        
        # Extract other relevant terms
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"] and token.text.lower() not in entities:
                entities[token.text.lower()] = token.text
        
        model = get_model()
        embedding = model.encode(query, convert_to_numpy=True)
        logger.info(f"Parsed query: {query}, Entities: {entities}")
        return entities, embedding
    except Exception as e:
        logger.error(f"Query parsing error: {str(e)}", exc_info=True)
        model = get_model()
        return {}, model.encode(query, convert_to_numpy=True)