import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json, requests
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()

async def trigger_api(url, method, param, api_token, req="", **kwargs):
    code = compile(param % kwargs, "<string>", "exec")
    exec(code, globals(), kwargs)
    try:
        headers = {'Content-Type': 'application/json'}
        req=kwargs['req']
        if api_token:
            headers = {'Content-Type': 'application/json', 'Authorization': api_token}
        if method == "get":
            response = requests.get(url, params=req, headers=headers)
            return response.text
        elif method == "post":
            response = requests.post(url, data=json.dumps(req), headers=headers)
            return response.text
    except Exception as e:
        message = str(f"We send {method} request to {url} & it returned us this error:", e)
        return message

class OpenAiLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, model="gpt-3.5-turbo-16k", temperature= 0.1, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        self.trigger_function_call = kwargs.get("trigger_function_call", None)
        self.custom_tools = kwargs.get("tools", None)
        self.url = kwargs.get("url", None)
        self.method = kwargs.get("method", None)
        self.param = kwargs.get("param", None)
        self.api_token = kwargs.get("api_token", None)
        self.started_streaming = False
        logger.info(f"Initializing OpenAI LLM with model: {self.model} and maxc tokens {max_tokens}")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.vllm_model = "vllm" in self.model
        self.model_args = { "max_tokens": self.max_tokens, "temperature": self.temperature, "model": self.model}
        if model == "Krutrim-spectre-v2":
            logger.info(f"Connecting to Ola's krutrim model")
            base_url = kwargs.get("base_url", os.getenv("OLA_KRUTRIM_BASE_URL"))
            api_key=kwargs.get('llm_key', None)
            if api_key is not None and len(api_key) > 0:
                api_key = api_key
            self.async_client = AsyncOpenAI( base_url=base_url, api_key= api_key)
        else:
            llm_key = kwargs.get('llm_key', os.getenv('OPENAI_API_KEY'))
            if llm_key != "sk-":
                llm_key = os.getenv('OPENAI_API_KEY')
            else:
                llm_key = kwargs['llm_key']
            self.async_client = AsyncOpenAI(api_key=llm_key)
            
    async def generate_stream(self, messages, synthesize=True, request_json=False):
        if len(messages) == 0:
            raise Exception("No messages provided")
        
        response_format = self.get_response_format(request_json)

        answer, buffer, resp, called_fun, i = "", "", "", "", 0
        logger.info(f"request to open ai {messages} max tokens {self.max_tokens} ")
        model_args = self.model_args
        model_args["response_format"] = response_format
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["stop"] = ["User:"]
        trigger_function_call = self.trigger_function_call
        if trigger_function_call:
            tool = json.loads(self.custom_tools)
            model_args["functions"]=tool
            model_args["function_call"]="auto"
        async for chunk in await self.async_client.chat.completions.create(**model_args):
            if trigger_function_call and dict(chunk.choices[0].delta).get('function_call'):
                if chunk.choices[0].delta.function_call.name:
                    called_fun = str(chunk.choices[0].delta.function_call.name)
                    i = [i for i in range(len(tool)) if called_fun == tool[i]["name"]][0]
                if (text_chunk := chunk.choices[0].delta.function_call.arguments):
                    resp += text_chunk
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

        if trigger_function_call and (all(key in resp for key in tool[i]["parameters"]["properties"].keys())):
            resp  = json.loads(resp)
            response = await trigger_api(url=self.url, method=self.method, param=self.param, api_token=self.api_token, **resp)
            content = f"We did made a function calling for user. We hit the function : {called_fun}, we hit the url {self.url} and send a {self.method} request and it returned us the response as given below: {str(response)} \n\n . Kindly understand the above response and convey this response in a conextual to user."
            model_args["messages"].append({"role":"system","content":content})
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

    async def generate(self, messages, request_json=False):
        response_format = self.get_response_format(request_json)
        logger.info(f"request to open ai {messages}")

        completion = await self.async_client.chat.completions.create(model=self.model, temperature=0.0, messages=messages,
                                                                     stream=False, response_format=response_format)
        res = completion.choices[0].message.content
        return res

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}