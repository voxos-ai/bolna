import os
import litellm
from dotenv import load_dotenv
from .llm import BaseLLM
from bolna.helpers.utils import json_to_pydantic_schema
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class LiteLLM(BaseLLM):
    def __init__(self, streaming_model, api_key=None, api_base=None, max_tokens=100, buffer_size=40,
                 classification_model=None, temperature=0.0):
        super().__init__(max_tokens, buffer_size)
        self.model = streaming_model
        self.api_key = api_key or os.getenv('LLM_MODEL_API_KEY')
        self.api_base = api_base or os.getenv('LLM_MODEL_API_BASE')
        self.started_streaming = False
        self.max_tokens = max_tokens
        self.classification_model = classification_model
        self.temperature = temperature

    async def generate_stream(self, messages, synthesize=True):
        answer, buffer = "", ""
        logger.info(f"request to model: {self.model}: {messages}")

        async for chunk in await litellm.acompletion(model=self.model, messages=messages, api_key=self.api_key,
                                                     api_base=self.api_base, temperature=0.2,
                                                     max_tokens=self.max_tokens, stream=True):
            if (text_chunk := chunk['choices'][0]['delta'].content) and not chunk['choices'][0].finish_reason:
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
        model = self.classification_model if classification_task is True else self.model
        logger.info(f'Request to litellm {messages}')

        completion_args = {
            "model": model,
            "messages": messages,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "stream": stream
        }

        if request_json is True:
            completion_args['response_format'] = {
                "type": "json_object",
                "schema": json_to_pydantic_schema('{"classification_label": "classification label goes here"}')
            }
        completion = await litellm.acompletion(**completion_args)
        text = completion.choices[0].message.content
        return text
