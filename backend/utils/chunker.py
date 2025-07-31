import spacy
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_lg")

def chunk_text(text, chunk_size=None):
    """Chunk text into sentences or fixed-size chunks."""
    try:
        if not text or not text.strip():
            logger.error("Empty text provided for chunking")
            return []
        doc = nlp(text)
        chunks = []
        current_chunk = ""
        chunk_size = int(os.getenv("CHUNK_SIZE", 100)) if chunk_size is None else chunk_size
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            if len(sent_text) > chunk_size:
                # Split long sentences
                words = sent_text.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            elif len(current_chunk) + len(sent_text) + 1 <= chunk_size:
                current_chunk += sent_text + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent_text + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Chunking error: {str(e)}", exc_info=True)
        return []