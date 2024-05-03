import os
import litellm
from dotenv import load_dotenv
from .llm import BaseLLM
from bolnaTestVersion.helpers.utils import json_to_pydantic_schema
from bolnaTestVersion.helpers.logger_config import configure_logger
import time

logger = configure_logger(__name__)
load_dotenv()


class LiteLLM(BaseLLM):
    def __init__(self, streaming_model, max_tokens=30, buffer_size=40,
                 classification_model=None, temperature=0.0, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = streaming_model
        self.started_streaming = False
        self.model_args = {"max_tokens": max_tokens, "temperature": temperature, "model": self.model}

        self.api_key = kwargs.get("llm_key", os.getenv('LITELLM_MODEL_API_KEY'))
        self.api_base = kwargs.get("base_url", os.getenv('LITELLM_MODEL_API_BASE'))
        self.api_version = kwargs.get("api_version", os.getenv('LITELLM_MODEL_API_VERSION'))
        if self.api_key:
            self.model_args["api_key"] = self.api_key
        if self.api_base:
            self.model_args["api_base"] = self.api_base
        if self.api_version:
            self.model_args["api_version"] = self.api_version

        if "top_k" in kwargs:
            self.model_args["top_k"] = kwargs["top_k"]
        if "top_p" in kwargs:
            self.model_args["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            self.model_args["stop"] = kwargs["stop"]
        if "presence_penalty" in kwargs:
            self.model_args["presence_penalty"] = kwargs["presence_penalty"]
        if "frequency_penalty" in kwargs:
            self.model_args["frequency_penalty"] = kwargs["frequency_penalty"]

        if len(kwargs) != 0:
            if "base_url" in kwargs:
                self.model_args["api_base"] = kwargs["base_url"]
            if "llm_key" in kwargs:
                self.model_args["api_key"] = kwargs["llm_key"]
            if "api_version" in kwargs:
                self.model_args["api_version"] = kwargs["api_version"]
        self.classification_model = classification_model

    async def generate_stream(self, messages, synthesize=True):
        answer, buffer = "", ""
        model_args = self.model_args.copy()
        model_args["messages"] = messages
        model_args["stream"] = True

        logger.info(f"request to model: {self.model}: {messages}")
        start_time = time.time()
        async for chunk in await litellm.acompletion(**model_args):
            if (text_chunk := chunk['choices'][0]['delta'].content) and not chunk['choices'][0].finish_reason:
                answer += text_chunk
                buffer += text_chunk

                if len(buffer) >= self.buffer_size and synthesize:
                    text = ' '.join(buffer.split(" ")[:-1])

                    if synthesize:
                        if not self.started_streaming:
                            self.started_streaming = True
                        yield text, False
                    buffer = buffer.split(" ")[-1]

        if synthesize:
            if buffer != "":
                yield buffer, True
        else:
            yield answer, True
        self.started_streaming = False
        logger.info(f"Time to generate response {time.time() - start_time}")

    async def generate(self, messages, classification_task=False, stream=False, synthesize=True, request_json=False):
        text = ""
        model_args = self.model_args.copy()
        model_args["model"] = self.classification_model if classification_task is True else self.model
        model_args["messages"] = messages
        model_args["stream"] = stream

        if request_json is True:
            model_args['response_format'] = {
                "type": "json_object",
                "schema": json_to_pydantic_schema('{"classification_label": "classification label goes here"}')
            }
        logger.info(f'Request to litellm {model_args}')
        try:
            completion = await litellm.acompletion(**model_args)
            text = completion.choices[0].message.content
            logger.error(completion)
        except Exception as e:
            logger.error(f'Error generating response {e}')
        return text
