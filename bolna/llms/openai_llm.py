import os
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import json, requests, time

from bolna.constants import PRE_FUNCTIONAL_CALL_MESSAGE
from bolna.helpers.utils import convert_to_request_log, format_messages
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()
    

class OpenAiLLM(BaseLLM):
    def __init__(self, max_tokens=100, buffer_size=40, model="gpt-3.5-turbo-16k", temperature= 0.1, **kwargs):
        super().__init__(max_tokens, buffer_size)
        self.model = model
        self.custom_tools = kwargs.get("api_tools", None)
        logger.info(f"API Tools {self.custom_tools}")
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            logger.info(f"Function dict {self.api_params}")
            self.tools = self.custom_tools['tools']
        else:
            self.trigger_function_call = False

        self.started_streaming = False
        logger.info(f"Initializing OpenAI LLM with model: {self.model} and maxc tokens {max_tokens}")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_args = { "max_tokens": self.max_tokens, "temperature": self.temperature, "model": self.model}
        provider = kwargs.get("provider", "openai")
        if  provider == "ola":
            logger.info(f"Connecting to Ola's krutrim model")
            base_url = kwargs.get("base_url", os.getenv("OLA_KRUTRIM_BASE_URL"))
            api_key=kwargs.get('llm_key', None)
            if api_key is not None and len(api_key) > 0:
                api_key = api_key
            self.async_client = AsyncOpenAI( base_url=base_url, api_key= api_key)
        elif kwargs.get("provider", "openai") == "custom":
            base_url = kwargs.get("base_url")
            api_key=kwargs.get('llm_key', None)
            self.async_client = AsyncOpenAI(base_url=base_url, api_key= api_key)
        else:
            llm_key = kwargs.get('llm_key', os.getenv('OPENAI_API_KEY'))
            if llm_key != "sk-":
                llm_key = os.getenv('OPENAI_API_KEY')
            else:
                llm_key = kwargs['llm_key']
            self.async_client = AsyncOpenAI(api_key=llm_key)
            api_key = llm_key
        self.assistant_id = kwargs.get("assistant_id", None)
        if self.assistant_id:
            logger.info(f"Initializing OpenAI assistant with assistant id {self.assistant_id}")
            self.openai = OpenAI(api_key=api_key)
            self.thread_id = self.openai.beta.threads.create().id
            self.model_args = {"max_completion_tokens": self.max_tokens, "temperature": self.temperature, "model": self.model}
            my_assistant = self.openai.beta.assistants.retrieve(self.assistant_id)
            if my_assistant.tools is not None:
                self.tools = [i for i in my_assistant.tools if i.type == "function"]
            logger.info(f'thread id : {self.thread_id}')
        self.run_id = kwargs.get("run_id", None)
        self.gave_out_prefunction_call_message = False
    
    async def trigger_api(self, url, method, param, api_token, **kwargs):
        try:
            code = compile(param % kwargs, "<string>", "exec")
            exec(code, globals(), kwargs)
            req = param % kwargs
            logger.info(f"Params {param % kwargs} \n {type(req)} \n {param} \n {kwargs} \n\n {req}")

            headers = {'Content-Type': 'application/json'}
            if api_token:
                headers = {'Content-Type': 'application/json', 'Authorization': api_token}
            if method == "get":
                logger.info(f"Sending request {req}, {url}, {headers}")
                response = requests.get(url, params=json.loads(req), headers=headers)
                logger.info(f"Response from The servers {response.text}")
                return response.text
            elif method == "post":
                logger.info(f"Sending request {json.loads(req)}, {url}, {headers}")
                response = requests.post(url, json=json.loads(req), headers=headers)
                logger.info(f"Response from The server {response.text}")
                return response.text
        except Exception as e:
            message = str(f"We send {method} request to {url} & it returned us this error:", e)
            logger.error(message)
            return message
    
    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info = None):
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
        model_args["user"] = f"{self.run_id}#{meta_info['turn_id']}"
        
        latency = False
        start_time = time.time()
        if self.trigger_function_call:
            tools = json.loads(self.tools)
            model_args["functions"]=tools
            model_args["function_call"]="auto"
        textual_response = False
        async for chunk in await self.async_client.chat.completions.create(**model_args):
            if not self.started_streaming:
                first_chunk_time = time.time()
                latency = first_chunk_time - start_time
                logger.info(f"LLM Latency: {latency:.2f} s")
                self.started_streaming = True
            if self.trigger_function_call and dict(chunk.choices[0].delta).get('function_call'):
                if not self.gave_out_prefunction_call_message and not textual_response:
                    yield PRE_FUNCTIONAL_CALL_MESSAGE, False, latency
                    self.gave_out_prefunction_call_message = True
                if len(buffer) > 0:
                    yield buffer, False, latency
                    buffer = ''
                logger.info(f"Response from LLM {resp}")
                yield buffer, False, latency
                buffer = ''
                if chunk.choices[0].delta.function_call.name:
                    logger.info(f"Should do a function call {chunk.choices[0].delta.function_call.name}")
                    called_fun = str(chunk.choices[0].delta.function_call.name)
                    i = [i for i in range(len(tools)) if called_fun == tools[i]["name"]][0]
                if (text_chunk := chunk.choices[0].delta.function_call.arguments):
                    resp += text_chunk
            elif text_chunk := chunk.choices[0].delta.content:
                textual_response = True
                answer += text_chunk    
                buffer += text_chunk
                if len(buffer) >= self.buffer_size and synthesize:
                    buffer_words = buffer.split(" ")
                    text = ' '.join(buffer_words[:-1])

                    if not self.started_streaming:
                        self.started_streaming = True
                    yield text, False, latency
                    buffer = buffer_words[-1]

        if self.trigger_function_call and (all(key in resp for key in tools[i]["parameters"]["properties"].keys())) and (called_fun in self.api_params):
            self.gave_out_prefunction_call_message = False
            logger.info(f"Function call paramaeters {resp}")
            convert_to_request_log(resp, meta_info, self.model, "llm", direction = "response", is_cached= False, run_id = self.run_id)
            resp  = json.loads(resp)
            func_dict = self.api_params[called_fun]
            logger.info(f"PAyload to send {resp} func_dict {func_dict}")

            url = func_dict['url']
            method = func_dict['method']
            param = func_dict['param']
            api_token = func_dict['api_token']
            response = await self.trigger_api(url= url, method=method.lower(), param= param, api_token= api_token, **resp)
            content = f"We did made a function calling for user. We hit the function : {called_fun}, we hit the url {url} and send a {method} request and it returned us the response as given below: {str(response)} \n\n . Kindly understand the above response and convey this response in a conextual to user."
            model_args["messages"].append({"role":"system","content":content})
            logger.info(f"Logging function call parameters ")
            convert_to_request_log(format_messages(model_args['messages'], True), meta_info, self.model, "llm", direction = "request", is_cached= False, run_id = self.run_id)
            async for chunk in await self.async_client.chat.completions.create(**model_args):
                if text_chunk := chunk.choices[0].delta.content:
                    answer += text_chunk
                    buffer += text_chunk

                    if len(buffer) >= self.buffer_size and synthesize:
                        buffer_words = buffer.split(" ")
                        text = ' '.join(buffer_words[:-1])

                        if not self.started_streaming:
                            self.started_streaming = True
                        yield text, False, latency
                        buffer = buffer_words[-1]

        if synthesize: # This is used only in streaming sense 
            yield buffer, True, latency
        else:
            yield answer, True, latency
        self.started_streaming = False

    async def generate(self, messages, request_json=False):
        response_format = self.get_response_format(request_json)
        logger.info(f"request to open ai {messages}")

        completion = await self.async_client.chat.completions.create(model=self.model, temperature=0.0, messages=messages,
                                                                     stream=False, response_format=response_format)
        res = completion.choices[0].message.content
        return res

    async def generate_assistant_stream(self, message, synthesize=True, request_json=False, meta_info=None):
        if len(message) == 0:
            raise Exception("No messages provided")

        response_format = self.get_response_format(request_json)

        answer, buffer, resp, called_fun, api_params, i = "", "", "", "", "", 0
        logger.info(f"request to open ai {message} max tokens {self.max_tokens} ")
        model_args = self.model_args
        model_args["thread_id"] = self.thread_id
        model_args["assistant_id"] = self.assistant_id
        model_args["stream"] = True
        model_args["response_format"] = response_format
        logger.info(f"request to open ai with thread {self.thread_id} & asst. id {self.assistant_id}")

        latency = False
        start_time = time.time()
        textual_response = False

        if self.trigger_function_call:
            tools = self.tools

        runs = await self.async_client.beta.threads.runs.list(thread_id=model_args["thread_id"])
        if runs.data and runs.data[0].status in ["in_progress", "queued", "requires_action"]:
            await self.async_client.beta.threads.runs.cancel(thread_id=model_args["thread_id"], run_id=runs.data[0].id)

        await self.async_client.beta.threads.messages.create(thread_id=model_args["thread_id"], role="user", content=message)

        async for chunk in await self.async_client.beta.threads.runs.create(**model_args):
            logger.info(f"chunk received : {chunk}")
            if self.trigger_function_call and chunk.event == "thread.run.step.delta":
                textual_response = False
                if not self.started_streaming:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    logger.info(f"LLM Latency: {latency:.2f} s")
                    self.started_streaming = True
                if not self.gave_out_prefunction_call_message and not textual_response:
                    yield PRE_FUNCTIONAL_CALL_MESSAGE, False, latency
                    self.gave_out_prefunction_call_message = True
                if len(buffer) > 0:
                    yield buffer, False, latency
                    buffer = ''
                yield buffer, False, latency
                buffer = ''
                if chunk.data.delta.step_details.tool_calls[0].function.name and chunk.data.delta.step_details.tool_calls[0].function.arguments is not None:
                    logger.info(f"Should do a function call {chunk.data.delta.step_details.tool_calls[0].function.name}")
                    called_fun = str(chunk.data.delta.step_details.tool_calls[0].function.name)
                    i = [i for i in range(len(tools)) if called_fun == tools[i].function.name][0]
                if (text_chunk := chunk.data.delta.step_details.tool_calls[0].function.arguments):
                    resp += text_chunk
                    logger.info(f"Response from LLM {resp}")
            elif chunk.event == 'thread.message.delta':
                if not self.started_streaming:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    logger.info(f"LLM Latency: {latency:.2f} s")
                    self.started_streaming = True
                textual_response = True
                text_chunk = chunk.data.delta.content[0].text.value
                answer += text_chunk
                buffer += text_chunk
                if len(buffer) >= self.buffer_size and synthesize:
                    buffer_words = buffer.split(" ")
                    text = ' '.join(buffer_words[:-1])

                    if not self.started_streaming:
                        self.started_streaming = True
                    yield text, False, latency
                    buffer = buffer_words[-1]

        if self.trigger_function_call and not textual_response and (all(key in resp for key in tools[i].function.parameters['properties'].keys())) and (called_fun in self.api_params):
            self.gave_out_prefunction_call_message = False
            logger.info(f"Function call parameters {resp}")
            convert_to_request_log(resp, meta_info, self.model, "llm", direction="response", is_cached=False, run_id=self.run_id)
            resp = json.loads(resp)
            func_dict = self.api_params[called_fun]
            logger.info(f"Payload to send {resp} func_dict {func_dict}")

            url = func_dict['url']
            method = func_dict['method']
            param = func_dict['param']
            api_token = func_dict['api_token']
            response = await self.trigger_api(url=url, method=method.lower(), param=param, api_token=api_token, **resp)
            content = f"We did made a function calling for user. We hit the function : {called_fun}, we hit the url {url} and send a {method} request and it returned us the response as given below: {str(response)} \n\n . Kindly understand the above response and convey this response in a contextual form to user."
            logger.info(f"Logging function call parameters ")
            runs = await self.async_client.beta.threads.runs.list(thread_id=model_args["thread_id"])
            if runs.data and runs.data[0].status in ["in_progress", "queued", "requires_action"]:
                await self.async_client.beta.threads.runs.cancel(thread_id=model_args["thread_id"], run_id=runs.data[0].id)

            await self.async_client.beta.threads.messages.create(thread_id=model_args["thread_id"], role="assistant", content=content)
            async for chunk in await self.async_client.beta.threads.runs.create(**model_args):
                logger.info(f"chunk received : {chunk}")
                if chunk.event == 'thread.message.delta':
                    text_chunk = chunk.data.delta.content[0].text.value
                    answer += text_chunk
                    buffer += text_chunk
                    if len(buffer) >= self.buffer_size and synthesize:
                        buffer_words = buffer.split(" ")
                        text = ' '.join(buffer_words[:-1])

                        if not self.started_streaming:
                            self.started_streaming = True
                        yield text, False, latency
                        buffer = buffer_words[-1]

        if synthesize:  # This is used only in streaming sense
            yield buffer, True, latency
        else:
            yield answer, True, latency
        self.started_streaming = False

    def get_response_format(self, is_json_format: bool):
        if is_json_format and self.model in ('gpt-4-1106-preview', 'gpt-3.5-turbo-1106'):
            return {"type": "json_object"}
        else:
            return {"type": "text"}