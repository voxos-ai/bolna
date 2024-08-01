from bolna.ingestion_server.embeddings import EmbedProviders
from bolna.ingestion_server.rags import RAGProviders, BaseRAG
from bolna.ingestion_server.datachecks import RAGConfig, RAGTask, RAGTaskStatus
import asyncio
from threading import Thread
from bolna.ingestion_server.utils import configure_logger
from typing import Dict
from uuid import uuid4
import tempfile
import os

logger = configure_logger(__name__)

class RAG:
    """
    Retrieval-Augmented Generation (RAG) implementation.
    
    This class handles the ingestion and storage of documents for RAG.
    """

    def __init__(self, VectorDB: BaseRAG, Workers: int = 2) -> None:
        """
        Initialize the RAG instance.

        Args:
            VectorDB (BaseRAG): The vector database to use.
            Workers (int, optional): Number of worker threads. Defaults to 2.
        """
        self.file_process_task_queue: asyncio.Queue = asyncio.Queue()
        self.file_store_task_queue: asyncio.Queue = asyncio.Queue()
        
        self.VectorDB: BaseRAG = VectorDB
        self.Workers: int = Workers

        self.RAG_THREAD = Thread(target=self.start)
        self.shutdown = False

    async def __shutdown_loop(self):
        """Monitor the shutdown flag."""
        while not self.shutdown:
            await asyncio.sleep(0.5)

    async def __ingestion_task(self):
        """Process ingestion tasks from the queue."""
        while not self.shutdown:
            task: RAGTask = await self.file_process_task_queue.get()
            task._status = RAGTaskStatus.PROCESSING
            try:
                nodes = await self.VectorDB.generate_nodes_sentence_splitter(task.file_loc)
            except Exception as e:
                logger.error(f"ERROR in {e}")
                task._status = RAGTaskStatus.ERROR
                continue
            task._nodes = nodes
            await self.file_store_task_queue.put(task)

    async def __nodes_storage(self):
        """Store processed nodes in the vector database."""
        while not self.shutdown:
            task: RAGTask = await self.file_store_task_queue.get()
            try:
                index = await self.VectorDB.add_index(task._nodes)
            except Exception as e:
                logger.error(f"ERROR in {e}")
                task._status = RAGTaskStatus.ERROR
                continue
            task._index = index
            task._status = RAGTaskStatus.SUCESSFUL

    def start(self):
        """Start the RAG processing loop."""
        loop = asyncio.new_event_loop()
        ingestion_task_pool = [loop.create_task(self.__ingestion_task()) for _ in range(self.Workers)]
        file_storage = loop.create_task(self.__nodes_storage())
        loop.run_until_complete(self.__shutdown_loop())
        file_storage.cancel()
        for t in ingestion_task_pool:
            t.cancel()
        loop.close()


class RAGFactory:
    """
    Factory class for creating and managing RAG instances.
    """

    def __init__(self) -> None:
        """Initialize the RAGFactory."""
        self.RAGS: Dict[str, RAG] = dict()

    def make_rag(self, config: RAGConfig):
        """
        Create a new RAG instance.

        Args:
            config (RAGConfig): Configuration for the RAG instance.

        Returns:
            str: Unique identifier for the created RAG instance.
        """
        rag_name = f"RAG-{uuid4()}"
        embedding_name = EmbedProviders[config.provider_config.embedding_name.provider](config.provider_config.embedding_name.embedding_model_name)
        vector_db = RAGProviders[config.provider](embedding_name, config.provider_config)
        rag = RAG(vector_db, config.provider_config.worker)
        rag.RAG_THREAD.start()
        self.RAGS[rag_name] = rag
        return rag_name
    
    def stop_all(self):
        """Stop all RAG instances."""
        for rag in self.RAGS.values():
            rag.shutdown = True

    def stop(self, rag_name: str):
        """
        Stop a specific RAG instance.

        Args:
            rag_name (str): Identifier of the RAG instance to stop.

        Raises:
            ValueError: If the specified RAG instance doesn't exist.
        """
        if rag_name in self.RAGS.keys():
            self.RAGS[rag_name].shutdown = True
            self.RAGS.pop(rag_name)
        else:
            raise ValueError("No RAG with that ID exists")
    
    async def file_ingest(self, rag_name, file) -> RAGTask:
        """
        Ingest a file into a RAG instance.

        Args:
            rag_name (str): Identifier of the RAG instance.
            file: File object to ingest.

        Returns:
            RAGTask: Task object representing the ingestion process.

        Raises:
            ValueError: If the specified RAG instance doesn't exist or if the file type is unsupported.
        """
        if rag_name not in self.RAGS.keys():
            raise ValueError(f"RAG: {rag_name} does not exist")
        if file.content_type not in ["application/pdf", "application/x-pdf"]:
            raise ValueError("Only PDF files are supported for now")
        
        task_id = str(uuid4())
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(await file.read())
        prev = temp_file.name
        file_name = f"/tmp/{task_id}.pdf"
        os.rename(prev, file_name)
        task = RAGTask(file_loc=file_name)
        await self.RAGS[rag_name].file_process_task_queue.put(task)

        while task._status in [RAGTaskStatus.WAIT, RAGTaskStatus.PROCESSING]:
            await asyncio.sleep(0.4)
        
        os.rename(file_name, prev)
        return task

    async def retrieve_query(self, rag_name: str, index: str, query: str):
        """
        Retrieve documents based on a query.

        Args:
            rag_name (str): Identifier of the RAG instance.
            index (str): Index to search in.
            query (str): Query string.

        Returns:
            List of relevant documents.
        """
        rag = self.RAGS[rag_name]
        return await rag.VectorDB.get_docs_index(query=query, index=index)