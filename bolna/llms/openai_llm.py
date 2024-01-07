import os
import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


class OpenAiLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, streaming_model="gpt-3.5-turbo-16k",
                 classification_model="gpt-3.5-turbo-1106", temperature= 0.1):
        super().__init__(max_tokens, buffer_size)
        self.model = streaming_model
        self.started_streaming = False
        self.async_client = AsyncOpenAI()
        self.max_tokens = max_tokens
        self.classification_model = classification_model
        self.temperature = temperature

    async def generate_stream(self, messages, classification_task=False, synthesize=True, request_json=False):
        response_format = self.get_response_format(request_json)

        answer, buffer = "", ""
        model = self.classification_model if classification_task is True else self.model
        logger.info(f"request to open ai {messages}")
        async for chunk in await self.async_client.chat.completions.create(model=model, temperature=self.temperature,
                                                                           messages=messages, stream=True,
                                                                           max_tokens=self.max_tokens,
                                                                           response_format=response_format):
            if text_chunk := chunk.choices[0].delta.content:
                answer += text_chunk
                buffer += text_chunk

                if len(buffer) >= self.buffer_size and synthesize:
                    text = ' '.join(buffer.split(" ")[:-1])

                    if synthesize:
                        if not self.started_streaming:
                            self.started_streaming = True
                        yield text
                    buffer = buffer.split(" ")[-1]

        if synthesize:
            if buffer != "":
                yield buffer
        else:
            yield answer
        self.started_streaming = False

    async def generate(self, messages, classification_task=False, stream=False, synthesize=True, request_json=False):
        response_format = self.get_response_format(request_json)
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
