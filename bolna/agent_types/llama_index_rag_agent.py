from llama_index.llms.openai import OpenAI
from .base_agent import BaseAgent
import dotenv
import os
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from bolna.helpers.logger_config import configure_logger
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from bolna.helpers.data_ingestion_pipe import lance_db
from typing import List, Tuple
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.llms import ChatMessage
from llama_index.agent.openai import OpenAIAgent

dotenv.load_dotenv()
logger = configure_logger(__name__)

class LlamaIndexRag(BaseAgent):
    def __init__(self,vector_id, temperature, model):
        super().__init__()
        self.vector_id = vector_id
        self.temperature = temperature
        self.model = model
        self.OPENAI_KEY = os.getenv('OPENAI_API_KEY')
        self.llm = OpenAI(model=self.model, temperature=self.temperature,api_key=self.OPENAI_KEY)
        self.vector_store = LanceDBVectorStore(lance_db,self.vector_id)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        vector_index = VectorStoreIndex.from_vector_store(storage_context.vector_store)
        # self.query_engine = vector_index.as_query_engine(
        #     similarity_top_k=8,
        #     streaming=True
        # )
        self.tools = [
            QueryEngineTool(
                vector_index.as_query_engine(
                    similarity_top_k=8
                ),
                metadata=ToolMetadata(
                    name="search",
                    description="Search the document, pass the entire user message in the query",
                ),
            )
        ]
        self.agent = OpenAIAgent.from_tools(tools=self.tools, verbose=True)
    def conversion(self,histroy:List[dict]) ->Tuple[ChatMessage,List[ChatMessage]]:
        ret = []
        message = ChatMessage(role=histroy[-1]['role'],content=histroy[-1]['content'])
        histroy = histroy[:-1]
        for mess in histroy:
            ret.append(ChatMessage(role=mess['role'],content=mess['content']))
        return message, ret

    async def generate(self, message, **k):
        """
        {"role":"",con..}
        .
        .
        """
        logger.info(f"LLAMA INDEX: {message}")
        message, his = self.conversion(message)
        # message, history = self.conversion(histroy=history)
        buffer:str = ""
        for token in self.agent.stream_chat(message.content,his).response_gen:
            buffer += " "+token
            if len(buffer.split(" ")) >= 40:
                yield buffer.strip(), False, 0.99
                logger.info(f"LLM RESPONSE: {buffer}")
                buffer = ""
        logger.info(f"LLM RESPONSE: {buffer}")
        yield buffer.strip(), True, 0.99