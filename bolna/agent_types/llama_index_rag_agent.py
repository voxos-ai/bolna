from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from .base_agent import BaseAgent
from llama_index.core import VectorStoreIndex
from bolna.helpers.logger_config import configure_logger
import os
from llama_parse import LlamaParse
from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    SentenceSplitter,
)
from llama_index.core.llms import ChatMessage
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SummaryIndex


logger = configure_logger(__name__)

class LlamaIndexPinecone(BaseAgent):
    def __init__(self,llm):
        super().__init__()
        self._llm = llm
        self.OPENAI_KEY = os.getenv('OPENAI_API_KEY')
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2,api_key=self.OPENAI_KEY)
        self.parser = LlamaParse(
            result_type="markdown",
        )
        nodes = self.processe_documents()
        ## to change the DB you can chnage this section
        self.vector_index = VectorStoreIndex(nodes=nodes)
        self.summary_index = SummaryIndex(nodes=nodes,show_progress=True)

        ## tools use in agents
        self.tools = [
            QueryEngineTool(
                self.vector_index.as_query_engine(
                    similarity_top_k=8
                ),
                metadata=ToolMetadata(
                    name="search",
                    description="Search the document, pass the entire user message in the query",
                ),
            ),
            QueryEngineTool(
                self.summary_index.as_query_engine(),
                metadata=ToolMetadata(
                    name="summarize",
                    description="Summarize the document using the user message",
                ),
            ),
        ]
        self.agent = OpenAIAgent.from_tools(tools=self.tools, verbose=True)
    def processe_documents(self):
        documents = self.parser.load_data("paper.pdf")
        node_parser = MarkdownElementNodeParser(num_workers=8)
        nodes = node_parser.get_nodes_from_documents(documents)
        nodes, objects = node_parser.get_nodes_and_objects(nodes)
        nodes = SentenceSplitter(chunk_size=512, chunk_overlap=20).get_nodes_from_documents(
            nodes
        )
    async def generate(self, history):
        summary = ""
        logger.info("extracting json from the previous conversation data")
        try:
            summary = await self.llm.generate(history, request_json=False)
            logger.info(f"summary {summary}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"error in generating summary: {e}")
        return {"summary": summary}
