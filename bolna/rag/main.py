import os
import logging
import uuid
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser

# Import your base classes and MongoDB implementation
from base import DatabaseConnector, VectorSearchEngine
from mongodb_rag import MongoDBConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variables
db_connector = None
search_engine = None

class EmbeddingConfig(BaseModel):
    model: str

class DatabaseConfig(BaseModel):
    connection_string: str
    db_name: str
    collection_name: str
    index_name: str
    llm_model: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-3-small"
    embedding_dimensions: Optional[int] = 256

class ProviderConfig(BaseModel):
    provider: str
    provider_config: Dict[str, DatabaseConfig]  # DatabaseConfig nested here

class RAGConfig(BaseModel):
    provider: ProviderConfig
    openai_api_key: str

class QueryRequest(BaseModel):
    query: str
    similarity_top_k: int = 3

@app.post("/setup")
async def setup_rag(config: RAGConfig):
    global db_connector, search_engine

    logger.info("Setting up RAG system...")
    os.environ["OPENAI_API_KEY"] = config.openai_api_key

    provider = config.provider
    if provider.provider.lower() == "mongodb":
        db_config = provider.provider_config.get("mongodb")
        if db_config:
            db_connector = MongoDBConnector(db_config)  # Make sure you're passing the right config
        else:
            raise HTTPException(status_code=400, detail="MongoDB configuration is required.")

    db_connector.connect()
    doc_count = db_connector.collection.count_documents({})
    logger.info(f"Documents in collection: {doc_count}")

    if doc_count == 0:
        logger.info("No documents found in the collection, ingestion is required.")
        return {"message": "No documents found in the collection, ingestion is required."}

    search_engine = VectorSearchEngine(db_connector)
    search_engine.setup_llama_index()
    search_engine.create_index()
    
    logger.info("RAG system setup complete")
    return {
        "message": "RAG system setup complete",
        "db_name": db_connector.db_name,
        "collection_name": db_connector.collection_name
    }

@app.post("/query")
async def query(request: QueryRequest):
    if not search_engine:
        raise HTTPException(status_code=400, detail="RAG system not set up. Call /setup first.")

    logger.info(f"Received query: {request.query}")
    response = search_engine.query(request.query, similarity_top_k=request.similarity_top_k)

    formatted_response = {
        "query": request.query,
        "response": str(response),
        "source_nodes": [
            {
                "score": node.score,
                "content": node.node.get_content()[:20],
                "metadata": node.node.metadata
            } for node in response.source_nodes
        ]
    }

    logger.info("Query response generated")
    return formatted_response

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    if not db_connector or not search_engine:
        raise HTTPException(status_code=400, detail="RAG system not set up. Call /setup first.")

    logger.info(f"Ingesting {len(files)} files")

    try:
        ingestion_id = str(uuid.uuid4())
        temp_dir = f"temp_upload_{ingestion_id}"
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded files temporarily
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

        # Use SimpleDirectoryReader to load documents
        documents = SimpleDirectoryReader(temp_dir).load_data()

        # Add ingestion_id to document metadata
        for doc in documents:
            doc.metadata["ingestion_id"] = ingestion_id

        # Create nodes
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)

        # Generate embeddings and add to vector store
        vector_store = db_connector.setup_vector_store()
        vector_store.add(nodes)

        # Clean up temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        logger.info(f"Ingested {len(nodes)} nodes with ingestion ID: {ingestion_id}")
        return {"message": f"Ingested {len(nodes)} nodes", "ingestion_id": ingestion_id}
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/status")
async def status():
    return {
        "rag_system_ready": db_connector is not None and search_engine is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
