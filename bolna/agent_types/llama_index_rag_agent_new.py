import os
import time
import asyncio
from typing import List, Tuple
import dotenv
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.selectors import LLMSingleSelector
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.data_ingestion_pipe import lance_db
from .base_agent import BaseAgent

dotenv.load_dotenv()
logger = configure_logger(__name__)

class LlamaIndexRagNew(BaseAgent):
    def __init__(self, vector_id, temperature, model):
        super().__init__()
        self.vector_id = vector_id
        self.temperature = temperature
        self.model = model
        self.OPENAI_KEY = os.getenv('OPENAI_API_KEY')

        # Settings
        Settings.llm = OpenAI(model=self.model, temperature=self.temperature, api_key=self.OPENAI_KEY)

        # Initialize components
        # asyncio.run(self.initialize_components())
        self.initialize_components()

    def initialize_components(self):
        self.vector_store = LanceDBVectorStore(lance_db, table_name=self.vector_id)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # loop = asyncio.get_running_loop()
        # with ThreadPoolExecutor() as pool:
        #     self.vector_index = await loop.run_in_executor(pool, VectorStoreIndex.from_vector_store, storage_context.vector_store)
        self.vector_index =  VectorStoreIndex.from_vector_store(storage_context.vector_store)
        self.query_engine = self.vector_index.as_query_engine(similarity_top_k=8, streaming=True)
        
        self.tools = [
            QueryEngineTool(
                self.query_engine,
                metadata=ToolMetadata(
                    name="search",
                    description="Search the document, pass the entire user message in the query"
                )
            )
        ]

        # Routing Query Engine
        # self.r_agent = RouterQueryEngine(
        #     query_engine_tools = self.tools,
        #     selector = LLMSingleSelector.from_defaults(),
        #     verbose = True
        # )

        # LLM Agent
        self.agent = OpenAIAgent.from_tools(tools=self.tools, verbose=True)

    def conversion(self, history: List[dict]) -> Tuple[ChatMessage, List[ChatMessage]]:
        ret = []
        message = ChatMessage(role=history[-1]['role'], content=history[-1]['content'])
        history = history[:-1]
        for mess in history:
            ret.append(ChatMessage(role=mess['role'], content=mess['content']))
        return message, ret

    async def generate(self, message, **kwargs):
        logger.info(f"LLAMA INDEX: {message}")
        message, history = self.conversion(message)

        async def process_query():
            start_time = time.time()

            async with asyncio.TaskGroup() as tg:
                agent_task = tg.create_task(asyncio.to_thread(self.agent.stream_chat, message.content, chat_history=history))
                vector_task = tg.create_task(asyncio.to_thread(self.vector_index.as_query_engine(), message.content))

            logger.info("Agent Streaming Result:")
            for token in agent_task.result().response_gen:
                yield token, False, 0.99

            print(f"Vector Index result : {vector_task.result()}")
            logger.info(f"Total time taken: {time.time() - start_time}")

        async for token, is_end, confidence in process_query():
            yield token, is_end, confidence