import os
import litellm
from dotenv import load_dotenv
from .llm import BaseLLM
from bolna.helpers.utils import json_to_pydantic_schema
from bolna.helpers.logger_config import configure_logger
import time
logger = configure_logger(__name__)
load_dotenv()


class LiteLLM(BaseLLM):
    def __init__(self, streaming_model, api_base=None, max_tokens=30, buffer_size=40,
                 classification_model=None, temperature=0.0, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = streaming_model
        self.api_key = kwargs.get("llm_key", os.getenv('LITELLM_MODEL_API_KEY'))
        self.api_base = api_base or os.getenv('LLM_MODEL_API_BASE')
        self.started_streaming = False
        self.max_tokens = max_tokens
        self.classification_model = classification_model
        self.temperature = temperature
        self.model_args = { "max_tokens": self.max_tokens, "temperature": self.temperature, "model": self.model}
        if "top_k" in kwargs:
            self.model_args["top_k"] = kwargs["top_k"]
        logger.info(f"Using VLLM model base_url {api_base} and model {self.model} and api key {self.api_key}")
        if "top_p" in kwargs:
            self.model_args["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            self.model_args["stop"] = kwargs["stop"]        
        if "presence_penalty" in kwargs:
            self.model_args["presence_penalty"] = kwargs["presence_penalty"]
        if  "frequency_penalty" in kwargs:
            self.model_args["frequency_penalty"] = kwargs["frequency_penalty"]



    async def generate_stream(self, messages, synthesize=True):
        answer, buffer = "", ""
        logger.info(f"request to model: {self.model}: {messages}")
        start_time = time.time()
        async for chunk in await litellm.acompletion(model=self.model, messages=messages, api_key=self.api_key,
                                                     api_base=self.api_base, temperature=0.2,
                                                     max_tokens=self.max_tokens, stream=True):
            logger.info(f"Got chunk {chunk}")
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
        model = self.classification_model if classification_task is True else self.model
        logger.info(f'Request to litellm {messages}')

        model_args = self.model_args
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["api_key"] = self.api_key
        model_args["api_base"] = self.api_base

        if request_json is True:
            model_args['response_format'] = {
                "type": "json_object",
                "schema": json_to_pydantic_schema('{"classification_label": "classification label goes here"}')
            }
        completion = await litellm.acompletion(**model_args)
        text = completion.choices[0].message.content
        return text