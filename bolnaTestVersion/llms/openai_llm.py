import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .llm import BaseLLM
from bolnaTestVersion.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class OpenAiLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, streaming_model="gpt-3.5-turbo-16k",
                 classification_model="gpt-3.5-turbo-1106", temperature= 0.1, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = streaming_model
        self.started_streaming = False
        logger.info(f"Initializing OpenAI LLM with model: {self.model} and maxc tokens {max_tokens}")
        self.max_tokens = max_tokens
        self.classification_model = classification_model
        self.temperature = temperature
        self.vllm_model = "vllm" in self.model
        self.model_args = { "max_tokens": self.max_tokens, "temperature": self.temperature, "model": self.model}

        if self.vllm_model:
            base_url = kwargs.get("base_url", os.getenv("VLLM_SERVER_BASE_URL"))
            api_key=kwargs.get('llm_key', None)
            if len(api_key) > 0:
                api_key = api_key
            else:
                api_key = "EMPTY"
            self.async_client = AsyncOpenAI( base_url=base_url, api_key= api_key)
            self.model = self.model[5:]
            self.model_args["model"] = self.model
            if "top_k" in kwargs:
                self.model_args["top_k"] = kwargs["top_k"]
            logger.info(f"Using VLLM model base_url {base_url} and model {self.model} and api key {api_key}")
        else:
            llm_key = kwargs.get('llm_key', os.getenv('OPENAI_API_KEY'))
            if llm_key != "sk-":
                llm_key = os.getenv('OPENAI_API_KEY')
            else:
                llm_key = kwargs['llm_key']
            self.async_client = AsyncOpenAI(api_key=llm_key)
        
        if "top_p" in kwargs:
            self.model_args["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            self.model_args["stop"] = kwargs["stop"]        
        if "presence_penalty" in kwargs:
            self.model_args["presence_penalty"] = kwargs["presence_penalty"]
        if  "frequency_penalty" in kwargs:
            self.model_args["frequency_penalty"] = kwargs["frequency_penalty"]

    async def generate_stream(self, messages, classification_task=False, synthesize=True, request_json=False):
        if len(messages) == 0:
            raise Exception("No messages provided")
        
        response_format = self.get_response_format(request_json)

        answer, buffer = "", ""
        model = self.classification_model if classification_task is True else self.model
        logger.info(f"request to open ai {messages} max tokens {self.max_tokens} ")
        model_args = self.model_args
        model_args["response_format"] = response_format
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["stop"] = ["User:"]
        async for chunk in await self.async_client.chat.completions.create(**model_args):
            if text_chunk := chunk.choices[0].delta.content:
                answer += text_chunk
                buffer += text_chunk

                if len(buffer) >= self.buffer_size and synthesize:
                    buffer_words = buffer.split(" ")
                    text = ' '.join(buffer_words[:-1])

                    if not self.started_streaming:
                        self.started_streaming = True
                    yield text, False
                    buffer = buffer_words[-1]

        if synthesize: # This is used only in streaming sense 
            yield buffer, True
        else:
            yield answer, True
        self.started_streaming = False

    async def generate(self, messages, classification_task=False, stream=False, synthesize=True, request_json=False):
        response_format = self.get_response_format(request_json)
        logger.info(f"request to open ai {messages}")
        model = self.classification_model if classification_task is True else self.model

        completion = await self.async_client.chat.completions.create(model=model, temperature=0.0, messages=messages,
                                                                     stream=False, response_format=response_format)
        res = completion.choices[0].message.content
        return res

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.classification_model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}