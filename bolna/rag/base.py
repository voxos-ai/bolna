from abc import ABC, abstractmethod
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

class DatabaseConnector(ABC):
    def __init__(self, db_name: str):
        self.db_name = db_name

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def verify_data(self):
        pass

    @abstractmethod
    def setup_vector_store(self):
        pass

    
class VectorSearchEngine:
    def __init__(self, db_connector):
        self.db_connector = db_connector
        self.index = None

    def setup_llama_index(self):
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=256)
        llm = OpenAI()
        Settings.llm = llm
        Settings.embed_model = embed_model

    def create_index(self):
        vector_store = self.db_connector.setup_vector_store()
        self.index = VectorStoreIndex.from_vector_store(vector_store)

    def query(self, query_text, similarity_top_k=5):
        if not self.index:
            raise ValueError("Index not created. Call create_index() first.")
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        return query_engine.query(query_text)