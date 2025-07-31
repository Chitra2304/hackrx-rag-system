import faiss
import redis
import numpy as np
import os
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Redis
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}", exc_info=True)
    raise

# Initialize SentenceTransformer lazily
_model = None
def get_model():
    """Load SentenceTransformer lazily."""
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model...")
        _model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return _model

def search_clauses(query_embedding, top_k):
    """Search for relevant clauses using FAISS with similarity threshold."""
    try:
        logger.info(f"Searching for top {top_k} clauses")
        clauses = []
        faiss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index"))
        faiss_path = os.path.join(faiss_dir, "index.faiss")
        
        if not os.path.exists(faiss_path):
            logger.error(f"FAISS index not found at {faiss_path}")
            return clauses
        
        # Load FAISS index
        index = faiss.read_index(faiss_path)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = index.search(query_embedding, top_k)
        
        # Initialize model for clause embeddings
        model = get_model()
        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))
        
        # Retrieve and filter clauses
        for idx, distance in zip(indices[0], distances[0]):
            for key in redis_client.keys("doc_*:*"):
                if f":{idx}" in key:
                    clause = redis_client.get(key)
                    if clause:
                        # Compute cosine similarity
                        clause_embedding = model.encode(clause, convert_to_numpy=True)
                        cosine_sim = np.dot(query_embedding, clause_embedding.T) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(clause_embedding)
                        )
                        if cosine_sim >= similarity_threshold and any(
                            term in clause.lower() for term in ["hospitalization", "treatment", "surgery", "insured", "waiting period"]
                        ):
                            clauses.append(clause)
        
        logger.info(f"Retrieved {len(clauses)} relevant clauses")
        return clauses[:top_k]
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return []