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
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
import dotenv
import asyncio
from .logger_config import configure_logger
dotenv.load_dotenv()

lance_db = os.getenv("LANCEDB_DIR","RAG")
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
    if LanceDBVectorStore(f"./{lance_db}")._table_exists(table_name):
        vector_store = LanceDBVectorStore(f"./{lance_db}",table_name,mode="append")
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
        vector_store = LanceDBVectorStore(f"./{lance_db}",table_name)
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