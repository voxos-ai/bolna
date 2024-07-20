import os
import time
import asyncio
from typing import List, Tuple, Generator, AsyncGenerator
import dotenv

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.agent.openai import OpenAIAgent

from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.data_ingestion_pipe import lance_db
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)
from llama_index.core.tools import FunctionTool
dotenv.load_dotenv()
logger = configure_logger(__name__)

def multiply(x, y):
  """Multiply two numbers."""
  return x * y
class LlamaIndexRag(BaseAgent):
    """
    A class that implements a RAG (Retrieval-Augmented Generation) system using LlamaIndex.

    This class sets up and manages an OpenAI-based language model, a vector store, and an agent
    for performing document search and question answering tasks.

    Attributes:
        vector_id (str): Identifier for the vector store.
        temperature (float): Temperature setting for the language model.
        model (str): The name of the OpenAI model to use.
        buffer (int): Size of the token buffer for streaming responses.
        max_tokens (int): Maximum number of tokens for the language model output.
        OPENAI_KEY (str): OpenAI API key retrieved from environment variables.
        llm (OpenAI): Instance of the OpenAI language model.
        vector_store (LanceDBVectorStore): Vector store for document storage and retrieval.
        vector_index (VectorStoreIndex): Index built on top of the vector store.
        query_engine: Query engine for searching the vector index.
        agent (OpenAIAgent): Agent that uses the query engine to answer questions.
    """

    def __init__(self, vector_id: str, temperature: float, model: str, buffer: int = 40, max_tokens: int = 100):
        """
        Initialize the LlamaIndexRag instance.

        Args:
            vector_id (str): Identifier for the vector store.
            temperature (float): Temperature setting for the language model.
            model (str): The name of the OpenAI model to use.
            buffer (int, optional): Size of the token buffer for streaming responses. Defaults to 40.
            max_tokens (int, optional): Maximum number of tokens for the language model output. Defaults to 100.
        """
        super().__init__()
        self.vector_id = vector_id
        self.temperature = temperature
        self.model = model
        self.buffer = buffer
        self.max_tokens = max_tokens
        self.OPENAI_KEY = os.getenv('OPENAI_API_KEY')
        self.LANCE_DB = os.getenv("LANCEDB_DIR")

        self.event_queue:asyncio.Queue = asyncio.Queue()
        self.count = -1
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        self.callback_manager = CallbackManager([self.llama_debug])
        self.bool = False

        self._setup_llm()
        self._setup_vector_store()
        self._setup_agent()

    

    async def __event_monitor(self):
        self.count = -1
        while self.count != -999:
            ls = self.llama_debug.get_events()
            # logger.info(f"EVENT MONITOR: {ls}")
            for i in range(self.count+1,len(ls)):
                if ls[i].event_type.value == "agent_step" and self.count != -1:
                    await self.event_queue.put("END")
                await self.event_queue.put(ls[i])
                self.count += 1
                logger.info(f"COUNTER INCREMNETED: {self.count}")
            logger.info(f"event:{[l.event_type.value for l in ls]}")
            logger.info(f"COUNT : {self.count}")
            await asyncio.sleep(0.3)

    def _setup_llm(self):
        """Set up the OpenAI language model."""
        self.llm = OpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.OPENAI_KEY,
            max_tokens=self.max_tokens
        )

    def _setup_vector_store(self):
        """Set up the vector store and index."""
        logger.info(f"LLAMA INDEX VALUES: {(lance_db, self.vector_id)}")
        self.vector_store = LanceDBVectorStore(uri=self.LANCE_DB , table_name=self.vector_id)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_index = VectorStoreIndex([], storage_context=storage_context, callback_manager=self.callback_manager)
        self.query_engine = self.vector_index.as_query_engine()

    def _setup_agent(self):
        """Set up the OpenAI agent with the query engine tool."""
        tools = [
            QueryEngineTool(
                self.query_engine,
                metadata=ToolMetadata(
                    name="search",
                    description="Search the document, pass the entire user message in the query",
                ),
            ),
            FunctionTool.from_defaults(multiply)
        ]
        self.agent = OpenAIAgent.from_tools(tools=tools, verbose=True, callback_manager=self.callback_manager)
        logger.info("LLAMA INDEX AGENT IS CREATED")

    @staticmethod
    def convert_history(history: List[dict]) -> Tuple[ChatMessage, List[ChatMessage]]:
        """
        Convert a list of message dictionaries to ChatMessage objects.

        Args:
            history (List[dict]): A list of dictionaries containing message data.

        Returns:
            Tuple[ChatMessage, List[ChatMessage]]: A tuple containing the latest message and the chat history.
        """
        message = ChatMessage(role=history[-1]['role'], content=history[-1]['content'])
        chat_history = [ChatMessage(role=msg['role'], content=msg['content']) for msg in history[:-1]]
        return message, chat_history

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, float, bool], None]:
        """
        Generate a response based on the input message and chat history.

        This method streams the generated response, yielding chunks of text along with metadata.

        Args:
            message (List[dict]): A list of dictionaries containing the message data and chat history.
            **kwargs: Additional keyword arguments (unused in this implementation).

        Yields:
            Tuple[str, bool, float, bool]: A tuple containing:
                - The generated text chunk.
                - A boolean indicating if this is the final chunk.
                - The latency of the first token.
                - A boolean indicating if the response was truncated (always False in this implementation).
        """
        logger.info(f"Generate function Input: {message}")
        monitor = asyncio.create_task(self.__event_monitor())
        self.llama_debug.flush_event_logs()
        self.count = -1
        message, history = await asyncio.to_thread(self.convert_history, message)
        
        
        latency = -1
        start_time = time.time()
        # token_generator = await asyncio.to_thread(self.agent.stream_chat, message.content, history)
        # logger.info(f"LATENCY:  {time.time() - start_time}")

        token_generator = await asyncio.create_task(self.agent.astream_chat(message.content, history))
        logger.info(f"LATENCY:  {time.time() - start_time}")
        wait_flag = True
        while True:
            data = await self.event_queue.get()
            logger.info("GOT DATA")
            if type(data) == str:
                end_point = await self.event_queue.get()
                logger.info(f"EVENT QUEUE END DATA: {data}")
                logger.info(end_point)
                if end_point.payload.get("response", False):
                    logger.info(f"TYPE: {type(end_point.payload['response'])}")
                    buffer = ""
                    # buffer and stuff
                    async for token in end_point.payload["response"].async_response_gen():
                        if latency < 0:
                            latency = time.time() - start_time
                        buffer += " " + token
                        if len(buffer.split()) >= self.buffer:
                            yield buffer.strip(), False, latency, False
                            logger.info(f"LLM BUFFER FULL BUFFER OUTPUT: {buffer}, {latency}")
                            buffer = ""
                    if len(buffer) > 1:
                        logger.info(f"LLM BUFFER FLUSH BUFFER OUTPUT: {buffer}, {latency}")
                        yield buffer.strip(), True, latency, False
                    # self.bool = True
                    # self.count = -1
                    logger.info(f"COUNTER RESETED: {self.count}")
                    break
            else:
                if data.event_type.value == "retrieve" and wait_flag:
                    logger.info("GOT WAITED")
                    yield "PLESE WAIT WE GONNA RETRIVE IMFORMATION FROM SOURCE", True, latency, False
                    wait_flag = False
                elif data.event_type.value == "function_call" and wait_flag:
                    if data.payload.get('tool',False):
                        if data.payload.get('tool').name in ["multiply"]:
                            print("FUNCTION")
                            yield f"PLESE WAIT CALLING THE {data.payload.get('tool').name}", True, latency, False
                            wait_flag = False
        # self.count = -999
        logger.info("CANCLING THE MONITOR")
        monitor.cancel()
        self.count = -999
        
        # async for token in token_generator.async_response_gen():

        # for token in token_generator.response_gen:
        # []<- buffer
        #   <- [XYZ]
        # yeild [].get()
        # 
        # [] <- event ot event out,
        # 
        # 
        # def co1:
        #   gen -> mon
        #   event -> queue
        # genrate:
        #   queue -> data
        #   if -> stem ->
        #   if -> filler   
        # async for token in token_generator.async_response_gen():
        #     if latency < 0:
        #         latency = time.time() - start_time
        #     buffer += " " + token
        #     if len(buffer.split()) >= self.buffer:
        #         yield buffer.strip(), False, latency, False
        #         logger.info(f"LLM BUFFER FULL BUFFER OUTPUT: {buffer}, {latency}")
        #         buffer = ""

        # if buffer:
        #     logger.info(f"LLM BUFFER FLUSH BUFFER OUTPUT: {buffer}, {latency}")
        #     yield buffer.strip(), True, latency, False