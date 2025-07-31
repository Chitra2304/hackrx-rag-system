import faiss
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

faiss_index_path = "data/faiss_index/index.faiss"

def check_faiss_index():
    if not os.path.exists(faiss_index_path):
        logger.error(f"FAISS index not found at {faiss_index_path}")
        return
    index = faiss.read_index(faiss_index_path)
    logger.info(f"FAISS index loaded. Total vectors: {index.ntotal}")
    if index.ntotal == 0:
        logger.warning("FAISS index is empty!")
    else:
        logger.info("FAISS index contains vectors, likely from processed documents.")

if __name__ == "__main__":
    check_faiss_index()
    