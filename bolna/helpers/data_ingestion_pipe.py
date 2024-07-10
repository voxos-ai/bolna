from uuid import uuid4
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_parse import LlamaParse
from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    SentenceSplitter
)
from typing import Dict, Tuple
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
import dotenv
import asyncio
from .logger_config import configure_logger
import tempfile
import threading

dotenv.load_dotenv()

lance_db = "/tmp/RAG"
llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")
chatgpt = os.getenv("OPENAI_API_KEY")
logger = configure_logger(__name__)
ingestion_tasks = []

logger.info(f"{lance_db}")
async def ingestion_task(temp_file_name:str,table_name:str,chunk_size:int = 512,overlaping:int = 200):
    parser = LlamaParse(
        api_key=llama_cloud_key,
        result_type="markdown",
    )
    embed_model = OpenAIEmbedding(model="text-embedding-3-small",api_key=chatgpt)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2,api_key=chatgpt)
    logger.info(f"Emdeding model, LLM model and Llama Parser were loaded")
    if LanceDBVectorStore(lance_db)._table_exists(table_name):
        vector_store = LanceDBVectorStore(lance_db,table_name,mode="append")
        logger.info(f"vector store is loaded")
        docs = await parser.aload_data(temp_file_name)
        node_parser = MarkdownElementNodeParser(num_workers=8,llm=llm)
        nodes = await node_parser.aget_nodes_from_documents(docs)
        nodes, objects = node_parser.get_nodes_and_objects(nodes)
        nodes = await SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlaping).aget_nodes_from_documents(
            nodes
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes=nodes,storage_context=storage_context,embed_model=embed_model)
    else:
        vector_store = LanceDBVectorStore(lance_db,table_name)
        logger.info(f"vector store is loaded")
        docs = await parser.aload_data(temp_file_name)
        node_parser = MarkdownElementNodeParser(num_workers=8,llm=llm)
        nodes = await node_parser.aget_nodes_from_documents(docs)
        nodes, objects = node_parser.get_nodes_and_objects(nodes)
        nodes = await SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlaping).aget_nodes_from_documents(
            nodes
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes=nodes,storage_context=storage_context,embed_model=embed_model)

def create_table(table_name,temp_file_name:str):
    loop = asyncio.new_event_loop()
    # table_name = str(uuid4())
    loop.run_until_complete(ingestion_task(temp_file_name=temp_file_name,table_name=table_name))

class TaskStatus:
    SUCCESS = "SUCCESS"
    WAIT = "WAIT"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"
class IngestionTask:
    def __init__(self,file_name:str, chunk_size:int = 512, overlaping:int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlaping = overlaping
        self.file_name = file_name
        
        self._status:int = TaskStatus.WAIT
        self._table_id:str = None
        self._message:str = None
class IngestionPipeline:
    def __init__(self,nuof_process:int=2) -> None:
        self.task_queue:asyncio.queues.Queue = asyncio.queues.Queue()
        # self.TaskIngestionThread = threading.Thread(target=self.__start)
        self.task_keeping_status_dict:Dict[str:IngestionTask] = dict()
        # self.TaskIngestionThread.start()
        # ingestion_task = asyncio.get_event_loop().create_task(self.start())
        ingestion_tasks = []
        for _ in range(nuof_process):
            ingestion_tasks.append(asyncio.create_task(self.start()))
        # asyncio.gather(ingestion_tasks)

    async def add_task(self,task_id:str, task:IngestionTask):
        self.task_keeping_status_dict[task_id] = task
        await self.task_queue.put(task)

    async def check_task_status(self,task_id)->Tuple[int,str,str]:
        task:IngestionTask = self.task_keeping_status_dict.get(task_id)
        return task._status, task._table_id, task._message
    
    async def start(self):
        logger.info("INGESTION PROCESS STARTED")
        while True:
            
            task:IngestionTask = await self.task_queue.get()
            
            logger.info(f"got packet for processing")
            task._status = TaskStatus.PROCESSING
            table_id = str(uuid4())
            try:
                await ingestion_task(task.file_name,table_id,task.chunk_size,task.overlaping)
                task._table_id = table_id
                task._message = "every thing run succesfully"
                task._status = TaskStatus.SUCCESS
            except Exception as e:
                logger.info(f"ERROR: {e}")
                task._message = "there is an error"
                task._status = TaskStatus.ERROR
    def _start(self):
        # loop = asyncio.new_event_loop()
        # try: 
        #     loop.run_in_executor(self.start())
        # except Exception as e:
        #     logger.info(f"error: {e}")
        asyncio.run(self.start())
    