from typing import Any, Coroutine
from bolna.ingestion_server.embeddings.base import BaseEmbed
from bolna.ingestion_server.rags.base import BaseRAG
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from bolna.ingestion_server.datachecks import ProviderConfig, LanceDBConfig
from llama_index.core.retrievers import VectorIndexRetriever

class LanceDB(BaseRAG):
    """
    LanceDB class for managing vector storage and retrieval using LanceDB.

    Attributes:
        similarity_top_k (int): Number of top similar items to retrieve.
        config (LanceDBConfig): Configuration for LanceDB.
        loc (str): Location for the vector database.
        path (str): Path to the vector database data.
    """

    def __init__(self, embedding_name: BaseEmbed, config: ProviderConfig) -> None:
        """
        Initializes the LanceDB instance.

        Args:
            embedding_name (BaseEmbed): The embedding model used.
            config (ProviderConfig): Configuration for the provider.
        """
        super().__init__("LanceDB", embedding_name, config.chunk_size, config.overlapping)
        self.similarity_top_k = config.similarity_top_k
        self.config: LanceDBConfig = config.rag
        self.loc = self.config.loc
        self.path = f"vectordb_data/{self.loc}"

    async def append_index(self, nodes) -> Coroutine[Any, Any, str]:
        """Appends nodes to the existing index.

        Args:
            nodes: The nodes to append.

        Returns:
            Coroutine: A coroutine that returns None.
        """
        return None
    
    async def add_index(self, nodes) -> str:
        """Adds nodes to the index and creates a new table.

        Args:
            nodes: The nodes to add.

        Returns:
            str: The name of the created table.
        """
        table_name = self.generate_index_name()
        # TODO: add reranking in the DB
        vector_store = LanceDBVectorStore(self.path, table_name=table_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=self.embeding_model)
        return table_name
    
    async def delete_index(self, index: str) -> bool:
        """Deletes an index.

        Args:
            index (str): The index to delete.

        Returns:
            bool: Result of the deletion operation.
        """
        return await super().delete_index(index)
    
    async def get_docs_index(self, query: str, index: str):
        """Retrieves documents from the index based on a query.

        Args:
            query (str): The query to search for.
            index (str): The index to search in.

        Returns:
            Retrieved documents based on the query.
        """
        vector_store = LanceDBVectorStore(uri=self.path, table_name=index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes=[], storage_context=storage_context)
        query_engine = VectorIndexRetriever(vector_index, similarity_top_k=self.similarity_top_k)
        return query_engine.retrieve(query)

        # query_engine = vector_index.as_query_engine(llm=self.llm)


