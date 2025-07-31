import pdfplumber
import faiss
import numpy as np
import redis
import os
import logging
from dotenv import load_dotenv
from backend.utils.chunker import chunk_text
from backend.utils.anonymizer import anonymize_text
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

def process_document(file_path, doc_id):
    """Process a document, extract text, chunk, anonymize, and store embeddings."""
    try:
        logger.info(f"Starting document processing for {file_path} with doc_id {doc_id}")
        
        # Verify file exists
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist")
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        # Extract text from PDF
        logger.info(f"Extracting text from {file_path}")
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"PDF text extraction failed for {file_path}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to extract text from {file_path}: {str(e)}")
        
        if not text.strip():
            logger.error(f"No text extracted from {file_path}")
            raise ValueError(f"No text extracted from {file_path}. Ensure the PDF contains readable text.")
        
        # Chunk text
        logger.info(f"Chunking text for {doc_id}")
        chunks = chunk_text(text)
        if not chunks:
            logger.error(f"No chunks created for {file_path}")
            raise ValueError(f"No chunks created for {file_path}")
        
        # Anonymize and create embeddings
        logger.info(f"Anonymizing and embedding {len(chunks)} chunks for {doc_id}")
        model = get_model()
        embeddings = []
        redis_keys = []
        for i, chunk in enumerate(chunks):
            try:
                anon_chunk = anonymize_text(chunk)
                if not anon_chunk.strip():
                    logger.warning(f"Chunk {i} for {doc_id} is empty after anonymization")
                    continue
                embedding = model.encode(anon_chunk, convert_to_numpy=True)
                embeddings.append(embedding)
                redis_key = f"doc_{doc_id}:{i}"
                redis_client.set(redis_key, anon_chunk)
                redis_keys.append(redis_key)
                logger.debug(f"Stored chunk {i} in Redis with key {redis_key}")
            except Exception as e:
                logger.error(f"Failed to process chunk {i} for {doc_id}: {str(e)}", exc_info=True)
                raise
        
        if not embeddings:
            logger.error(f"No valid embeddings created for {file_path}")
            raise ValueError(f"No valid embeddings created for {file_path}")
        
        # Store embeddings in FAISS
        logger.info(f"Storing {len(embeddings)} embeddings in FAISS for {doc_id}")
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        
        faiss_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index"))
        os.makedirs(faiss_dir, exist_ok=True)
        faiss_path = os.path.join(faiss_dir, f"{doc_id}.faiss")
        
        # Verify FAISS path is writable
        if not os.access(faiss_dir, os.W_OK):
            logger.error(f"No write permission for FAISS directory {faiss_dir}")
            raise PermissionError(f"No write permission for FAISS directory {faiss_dir}")
        
        faiss.write_index(index, faiss_path)
        logger.info(f"Saved FAISS index to {faiss_path}")
        
        # Store FAISS index mapping in Redis
        redis_client.set(f"faiss_index:{doc_id}", faiss_path)
        logger.info(f"Processed {len(chunks)} chunks and stored {len(embeddings)} embeddings for {doc_id}")
        logger.debug(f"Redis keys created: {redis_keys}")
    
    except Exception as e:
        logger.error(f"Document processing failed for {file_path}: {str(e)}", exc_info=True)
        raise