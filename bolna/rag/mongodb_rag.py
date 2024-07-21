import os
import logging
from typing import List, Optional
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from bolna.rag.base import DatabaseConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBConfig(BaseModel):
    connection_string: str
    db_name: str
    collection_name: str
    index_name: str
    llm_model: Optional[str] = "gpt-3.5-turbo"  # Default LLM model
    embedding_model: Optional[str] = "text-embedding-3-small"  # Default embedding model
    embedding_dimensions: Optional[int] = 256  # Default dimensions

class MongoDBConnector(DatabaseConnector):
    def __init__(self, config: MongoDBConfig):
        super().__init__(config.db_name)
        self.config = config
        self.client = None
        self.db = None
        self.collection = None
        self.collection_name = config.collection_name  # Add this line

    def connect(self):
        try:
            self.client = MongoClient(self.config.connection_string)
            self.db = self.client[self.config.db_name]
            self.collection = self.db[self.config.collection_name]
            logger.info("Connection to MongoDB successful")
        except ConnectionFailure as e:
            logger.error(f"Connection failed: {e}")
            raise

    def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    def verify_data(self):
        doc_count = self.collection.count_documents({})
        logger.info(f"Total documents in collection: {doc_count}")
        if doc_count > 0:
            logger.info(f"Documents found in the collection.")
        else:
            logger.warning("No documents found in the collection.")

    def setup_vector_store(self):
        return MongoDBAtlasVectorSearch(
            self.client,
            db_name=self.config.db_name,
            collection_name=self.config.collection_name,
            index_name=self.config.index_name
        )

class RAGEngine:
    def __init__(self, db_connector: MongoDBConnector):
        self.db_connector = db_connector
        self.index = None

    def setup(self):
        embed_model = OpenAIEmbedding(model=self.db_connector.config.embedding_model, dimensions=self.db_connector.config.embedding_dimensions)
        llm = OpenAI(model=self.db_connector.config.llm_model)
        Settings.llm = llm
        Settings.embed_model = embed_model

        vector_store = self.db_connector.setup_vector_store()
        self.index = VectorStoreIndex.from_vector_store(vector_store)

    def query(self, query_text: str, similarity_top_k: int = 5):
        if not self.index:
            raise ValueError("Index not created. Call setup() first.")
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(query_text)
        
        logger.info(f"Query: {query_text}")
        logger.info(f"Response: {response.response}")
        for node in response.source_nodes:
            logger.info(f"Source - Score: {node.score}, Content: {node.node.get_content()[:10]}")
            #logger.info(f"Source - Score: {node.score}, Content: {node.node.get_content()[:10]}..., Metadata: {node.node.metadata}")

        return response

def main():
    OPENAI_API_KEY = "***"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    config = MongoDBConfig(
        connection_string="mongodb+srv://vipul:qqgr4bwAYl5pZSU9@testing-rag.nqaknom.mongodb.net/",
        db_name="movies",
        collection_name="movies_records",
        index_name="vector_index",
    )

    try:
        db_connector = MongoDBConnector(config)
        db_connector.connect()
        db_connector.verify_data()

        rag_engine = RAGEngine(db_connector)
        rag_engine.setup()

        query = "Any romantic movie for me? you can give anything you want?"
        rag_engine.query(query)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'db_connector' in locals():
            db_connector.disconnect()

if __name__ == "__main__":
    main()
