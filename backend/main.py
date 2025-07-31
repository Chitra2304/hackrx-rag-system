from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.modules.doc_processor import process_document
from backend.modules.query_parser import parse_query
from backend.modules.semantic_search import search_clauses
from backend.modules.decision_engine import evaluate_clauses
from backend.modules.response_generator import generate_response
import os
import logging
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        logger.info(f"Received upload request for {file.filename}")
        if not file.filename.endswith(('.pdf', '.docx', '.eml')):
            logger.error(f"Unsupported file format: {file.filename}")
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF, DOCX, and EML are allowed.")
        
        upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "uploads"))
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        # Verify directory is writable
        if not os.access(upload_dir, os.W_OK):
            logger.error(f"No write permission for upload directory {upload_dir}")
            raise HTTPException(status_code=500, detail=f"No write permission for upload directory {upload_dir}")
        
        # Save the uploaded file
        logger.info(f"Saving file to {file_path}")
        content = await file.read()
        if not content:
            logger.error(f"Uploaded file {file.filename} is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Verify file was saved
        if not os.path.exists(file_path):
            logger.error(f"Failed to save file {file_path}")
            raise HTTPException(status_code=500, detail=f"Failed to save file {file_path}")
        
        # Process the document
        doc_id = file.filename.replace(".", "_")
        logger.info(f"Processing document {doc_id}")
        process_document(file_path, doc_id)
        logger.info(f"Successfully processed {file.filename}")
        return {"status": "success", "filename": file.filename}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Upload error for {file.filename if file else 'unknown file'}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process_query")
async def process_query(request: QueryRequest):
    """Process a user query and return a decision."""
    try:
        logger.info(f"Processing query: {request.query}")
        entities, embedding = parse_query(request.query)
        clauses = search_clauses(embedding, int(os.getenv("TOP_K_RESULTS", 10)))
        decision = evaluate_clauses(entities, clauses)
        response = generate_response(decision)
        logger.info(f"Processed query: {request.query}, Decision: {decision['decision']}")
        return response
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)