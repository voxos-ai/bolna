from fastapi import FastAPI, UploadFile
from bolna.ingestion_server.datachecks import RAGConfig
from bolna.ingestion_server import RAGFactory
import uvicorn
import time
from typing import Dict
import dotenv

dotenv.load_dotenv()
DB: Dict[str, RAGConfig] = {}

rag_factory = RAGFactory()
app = FastAPI()


@app.get("/")
def heartbeat() -> float:
    """Health check endpoint that returns the current server time."""
    return time.time()


@app.post("/create-rag")
def create_rag(request: RAGConfig) -> Dict[str, str]:
    """Create a RAG configuration and return its ID.

    Args:
        request (RAGConfig): The RAG configuration to create.

    Returns:
        Dict[str, str]: A dictionary containing the created RAG ID.
    """
    print(request)
    rag_id = rag_factory.make_rag(request)
    return {"rag_id": rag_id}


@app.post("/rag-upload-file/{rag_id}")
async def rag_upload_file(file: UploadFile, rag_id: str) -> Dict[str, str]:
    """Upload a file for a specific RAG ID.

    Args:
        file (UploadFile): The file to upload.
        rag_id (str): The ID of the RAG to associate with the file.

    Returns:
        Dict[str, str]: A dictionary containing the upload status and index.
    """
    try:
        task = await rag_factory.file_ingest(rag_name=rag_id, file=file)
        return {"index": task._index, "status": task._status, "message": "DONE"}
    except Exception as e:
        return {"index": None, "status": "ERROR", "message": f"{e}"}
    

@app.get("/rag-retrive/{rag_id}/{index}")
async def rag_retrive(query: str, rag_id: str, index: str) -> list:
    """Retrieve documents based on a query for a specific RAG ID and index.

    Args:
        query (str): The query string to search for.
        rag_id (str): The ID of the RAG to search in.
        index (str): The index to search in.

    Returns:
        list: A list of documents matching the query.
    """
    docs = await rag_factory.retrieve_query(rag_name=rag_id, index=index, query=query)
    send_filter = [{"text": node.text, "score": node.score} for node in docs]
    return send_filter