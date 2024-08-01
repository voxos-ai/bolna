import os
import dotenv
from uuid import uuid4

from llama_parse import LlamaParse
from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    SentenceSplitter,
    TextSplitter
)
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from bolna.ingestion_server.embeddings import BaseEmbed
from bolna.ingestion_server.utils import configure_logger


dotenv.load_dotenv()

logger = configure_logger(__name__)

class BaseRAG:
    """
    Base class for Retrieval-Augmented Generation (RAG) systems.
    
    Attributes:
        provider (str): The provider for the RAG system.
        base_embed (BaseEmbed): The embedding model used.
        embeding_model: The actual embedding model instance.
        chunk_size (int): Size of the chunks for splitting documents.
        overlapping (int): Overlap size for chunking.
        LLAMA_CLOUD (str): API key for Llama Cloud.
        parse (LlamaParse): Instance of LlamaParse for data parsing.
        OPENAI_API_KEY (str): API key for OpenAI.
        llm (OpenAI): Instance of OpenAI model.
    """

    def __init__(self, provider: str, embedding_name: BaseEmbed, chunk_size: int, overlapping: int) -> None:
        """
        Initializes the BaseRAG instance with the specified parameters.

        Args:
            provider (str): The provider for the RAG system.
            embedding_name (BaseEmbed): The embedding model used.
            chunk_size (int): Size of the chunks for splitting documents.
            overlapping (int): Overlap size for chunking.
        
        Raises:
            ValueError: If required environment variables are not set.
        """
        self.provider = provider
        self.base_embed: BaseEmbed = embedding_name
        self.embeding_model = self.base_embed.get_embedding()
        self.chunk_size = chunk_size
        self.overlapping = overlapping
        self.LLAMA_CLOUD = os.getenv("LLAMA_CLOUD_API_KEY")
        
        if self.LLAMA_CLOUD is None:
            raise ValueError("LLAMA_CLOUD_API_KEY is not set in .env")
        
        self.parse = LlamaParse(api_key=self.LLAMA_CLOUD, result_type="markdown")
        
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if self.OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        
        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=self.OPENAI_API_KEY)

        Settings.embed_model = self.embeding_model
        Settings.llm = self.llm

    def generate_index_name(self) -> str:
        """Generates a unique index name using UUID.

        Returns:
            str: A unique index name.
        """
        return str(uuid4())
    
    async def generate_nodes_sentence_splitter(self, file_loc: str):
        """Generates nodes using a sentence splitter.

        Args:
            file_loc (str): The file location to load data from.

        Returns:
            nodes: The generated nodes after processing.
        """
        docs = await self.parse.aload_data(file_path=file_loc)
        node_parser = MarkdownElementNodeParser(num_workers=8, llm=self.llm)
        nodes = await node_parser.aget_nodes_from_documents(docs)
        nodes, _ = node_parser.get_nodes_and_objects(nodes)
        nodes = await SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlapping).aget_nodes_from_documents(nodes)
        return nodes

    async def generate_nodes_text_splitter(self, file_loc: str):
        """Generates nodes using a text splitter.

        Args:
            file_loc (str): The file location to load data from.

        Returns:
            nodes: The generated nodes after processing.
        """
        docs = await self.parse.aload_data(file_path=file_loc)
        node_parser = MarkdownElementNodeParser(num_workers=8, llm=self.llm)
        nodes = await node_parser.aget_nodes_from_documents(docs)
        nodes, _ = node_parser.get_nodes_and_objects(nodes)
        nodes = await TextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlapping).aget_nodes_from_documents(nodes)
        return nodes
    
    async def append_index(self, nodes) -> str:
        """Appends nodes to the existing index.

        Args:
            nodes: The nodes to append.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    async def add_index(self, nodes) -> str:
        """Adds nodes to the index.

        Args:
            nodes: The nodes to add.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    async def delete_index(self, index: str) -> bool:
        """Deletes an index.

        Args:
            index (str): The index to delete.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    async def get_docs_index(self, query: str, index: str):
        """Retrieves documents from the index based on a query.

        Args:
            query (str): The query to search for.
            index (str): The index to search in.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError