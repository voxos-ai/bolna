from llama_index.llms.openai import OpenAI
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger
import dotenv
import os
dotenv.load_dotenv()
logger = configure_logger(__name__)

class LlamaIndex(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40):
        super().__init__(max_tokens, buffer_size)
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = "gpt-3.5-turbo"
        self.temperature = 0
        self.model = OpenAI(self.model_name,self.temperature,max_tokens=self.max_tokens,api_key=self.api_key)
    async def generate(self, messages, stream=False, request_json=False, meta_info = None):
        return None
    async def generate_stream(self, messages, synthesize=True, meta_info = None):
        return None
