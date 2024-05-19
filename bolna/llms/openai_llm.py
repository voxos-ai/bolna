import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json, requests
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()

# async def trigger_api(url, method, param, api_token, req="", **kwargs):
#     response = await trigger_api(url=self.url, method=self.method.lower(), param=self.param, api_token=self.api_token, **resp)
#     code = compile(param % kwargs, "<string>", "exec")
#     exec(code, globals(), kwargs)
#     req=kwargs['req']

#     try:
#         headers = {'Content-Type': 'application/json'}

#         logger.info(f"Request to API {req} {code} {method} {url}")
#         if api_token:
#             headers = {'Content-Type': 'application/json', 'Authorization': f"Bearer {api_token}"}
#         if method == "get":
#             response = requests.get(url, params=req, headers=headers)
#             return response.text
#         elif method == "post":
#             logger.info(f"Sending to SERVER")
#             response = requests.post(url, data=json.dumps(req), headers=headers)
#             logger.info(f"Response from The sercvers {response.text}")
#             return response.text
#     except Exception as e:
#         message = str(f"We send {method} request to {url} & it returned us this error:", e)
#         logger.error(message)
#         return message


async def trigger_api(url, method, param, api_token, **kwargs):
    try:
        code = compile(param % kwargs, "<string>", "exec")
        exec(code, globals(), kwargs)
        req = param % kwargs
        logger.info(f"PArams {param % kwargs}")

        headers = {'Content-Type': 'application/json'}
        if api_token:
            headers = {'Content-Type': 'application/json', 'Authorization': api_token}
        if method == "get":
            response = requests.get(url, params=req, headers=headers)
            return response.text
        elif method == "post":
            logger.info(f"Sending request {req}, {url}")
            response = requests.post(url, data=json.dumps(req), headers=headers)
            logger.info(f"Response from The sercvers {response.text}")
            return response.text
    except Exception as e:
        message = str(f"We send {method} request to {url} & it returned us this error:", e)
        logger.error(message)
        return message
    

class OpenAiLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, model="gpt-3.5-turbo-16k", temperature= 0.1, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        self.custom_tools = kwargs.get("api_tools", None)
        logger.info(f"API Tools {self.custom_tools}")
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            self.tools = self.custom_tools['tools']
        else:
            self.trigger_function_call = False

        self.started_streaming = False
        logger.info(f"Initializing OpenAI LLM with model: {self.model} and maxc tokens {max_tokens}")
        self.max_tokens = max_tokens
        self.temperature = temperature
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

        answer, buffer, resp, called_fun, api_params, i = "", "", "", "", "", 0
        logger.info(f"request to open ai {messages} max tokens {self.max_tokens} ")
        model_args = self.model_args
        model_args["response_format"] = response_format
        model_args["messages"] = messages
        model_args["stream"] = True
        model_args["stop"] = ["User:"]
        if self.trigger_function_call:
            tools = json.loads(self.tools)
            model_args["functions"]=tools
            model_args["function_call"]="auto"
        async for chunk in await self.async_client.chat.completions.create(**model_args):
            if self.trigger_function_call and dict(chunk.choices[0].delta).get('function_call'):
                if chunk.choices[0].delta.function_call.name:
                    logger.info(f"Should do a function call {chunk.choices[0].delta.function_call.name}")
                    called_fun = str(chunk.choices[0].delta.function_call.name)
                    i = [i for i in range(len(tools)) if called_fun == tools[i]["name"]][0]
                if (text_chunk := chunk.choices[0].delta.function_call.arguments):
                    resp += text_chunk
            elif text_chunk := chunk.choices[0].delta.content:
                answer += text_chunk
                buffer += text_chunk

                if len(buffer) >= self.buffer_size and synthesize:
                    buffer_words = buffer.split(" ")
                    text = ' '.join(buffer_words[:-1])

                    if not self.started_streaming:
                        self.started_streaming = True
                    yield text, False
                    buffer = buffer_words[-1]

        if self.trigger_function_call and (all(key in resp for key in tools[i]["parameters"]["properties"].keys())) and (called_fun in self.api_params):
            resp  = json.loads(resp)
            logger.info(f"PAyload to send {resp}")
            func_dict = self.api_params[called_fun]
            url = func_dict['url']
            method = func_dict['method']
            param = func_dict['param']
            api_token = func_dict['api_token']
            response = await trigger_api(url= url, method=method.lower(), param= param, api_token= api_token, **resp)
            content = f"We did made a function calling for user. We hit the function : {called_fun}, we hit the url {url} and send a {method} request and it returned us the response as given below: {str(response)} \n\n . Kindly understand the above response and convey this response in a conextual to user."
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