from typing import Any, Coroutine
from pymongo import MongoClient
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from bolna.ingestion_server.embeddings.base import BaseEmbed
from bolna.ingestion_server.rags.base import BaseRAG
from bolna.ingestion_server.datachecks import ProviderConfig, MongoDBConfig

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever

class MongoDB(BaseRAG):
    """
    MongoDB class for managing vector storage and retrieval using MongoDB.

    Attributes:
        similarity_top_k (int): Number of top similar items to retrieve.
        config (MongoDBConfig): Configuration for MongoDB.
        client (MongoClient): MongoDB client instance.
    """

    def __init__(self, embedding_name: BaseEmbed, config: ProviderConfig) -> None:
        """
        Initializes the MongoDB instance.

        Args:
            embedding_name (BaseEmbed): The embedding model used.
            config (ProviderConfig): Configuration for the provider.
        """
        super().__init__("MongoDB", embedding_name, config.chunk_size, config.overlapping)
        self.similarity_top_k = config.similarity_top_k
        self.config: MongoDBConfig = config.rag
        self.client = MongoClient(self.config.uri)

    async def append_index(self, nodes) -> Coroutine[Any, Any, str]:
        """Appends nodes to the existing index.

        Args:
            nodes: The nodes to append.

        Returns:
            Coroutine: A coroutine that calls the base class method.
        """
        return await super().append_index(nodes)

    async def get_docs_index(self, query: str, index: str):
        """Retrieves documents from the index based on a query.

        Args:
            query (str): The query to search for.
            index (str): The index to search in.

        Returns:
            Retrieved documents based on the query.
        """
        vector_store = MongoDBAtlasVectorSearch(
            self.client,
            db_name=self.config.db,
            collection_name=self.config.collection_name,
            vector_index_name=index
        )
        vector_store_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_store_index = VectorStoreIndex(nodes=[], storage_context=vector_store_context)
        vector_store_retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=self.similarity_top_k)
        return vector_store_retriever.retrieve(query)

    async def delete_index(self, index: str) -> Coroutine[Any, Any, bool]:
        """Deletes an index.

        Args:
            index (str): The index to delete.

        Returns:
            Coroutine: A coroutine that calls the base class method.
        """
        return await super().delete_index(index)

    async def add_index(self, nodes) -> str:
        """Adds nodes to the index and creates a new index.

        Args:
            nodes: The nodes to add.

        Returns:
            str: The name of the created index.
        """
        index = self.generate_index_name()
        vector_store = MongoDBAtlasVectorSearch(
            self.client,
            db_name=self.config.db,
            collection_name=self.config.collection_name,
            index_name=index
        )
        vector_store_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_store_index = VectorStoreIndex(nodes=nodes, storage_context=vector_store_context, embed_model=self.embeding_model)
        return index