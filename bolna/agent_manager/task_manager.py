import asyncio
import traceback
import time
import json
from .base_manager import BaseManager
from bolna.agent_types import *
from bolna.providers import *
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, is_valid_md5, get_raw_audio_bytes_from_base64, \
    get_required_input_types, format_messages, get_prompt_responses, merge_wav_bytes, pcm_to_wav_bytes, update_prompt_with_context, get_md5_hash, clean_json_string, wav_bytes_to_pcm, yield_chunks_from_memory
from bolna.helpers.logger_config import configure_logger

asyncio.get_event_loop().set_debug(True)
logger = configure_logger(__name__)


class TaskManager(BaseManager):
    def __init__(self, assistant_name, task_id, task, ws, input_parameters=None, context_data=None,
                 assistant_id=None, run_id=None, connected_through_dashboard=False, 
                 cache =  None, input_queue = None, conversation_history = None, output_queue = None, **kwargs):
        super().__init__()
        logger.info(f"doing task {task}")
        self.task_id = task_id
        self.assistant_name = assistant_name
        self.tools = {}
        self.websocket = ws
        self.task_config = task
        self.context_data = context_data
        self.connected_through_dashboard = connected_through_dashboard

        # Set up communication queues between processes
        self.audio_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.synthesizer_queue = asyncio.Queue()
        self.transcriber_output_queue = asyncio.Queue()

        self.pipelines = task['toolchain']['pipelines']
        self.textual_chat_agent = False
        if task['toolchain']['pipelines'][0] == "llm" and task["tools_config"]["llm_agent"][
            "agent_task"] == "conversation":
            self.textual_chat_agent = False

        self.start_time = time.time()

        # Assistant persistance stuff
        self.assistant_id = assistant_id
        self.run_id = run_id
        self.mark_set = set()

        self.conversation_ended = False

        # Prompts
        self.prompts, self.system_prompt = {}, {}

        self.input_parameters = input_parameters

        self.queues = {
            "transcriber": self.audio_queue,
            "llm": self.llm_queue,
            "synthesizer": self.synthesizer_queue
        }

        if task_id == 0:
            if self.task_config["tools_config"]["input"]["provider"] in SUPPORTED_INPUT_HANDLERS.keys():
                logger.info(f"Connected through dashboard {connected_through_dashboard}")
                input_kwargs = {"queues": self.queues,
                                "websocket": self.websocket,
                                "input_types": get_required_input_types(task),
                                "mark_set": self.mark_set,
                                "connected_through_dashboard": self.connected_through_dashboard}  
                                                          
                if connected_through_dashboard:
                    logger.info("Connected through dashboard and hence using default input handler")
                    # If connected through dashboard get basic dashboard class
                    input_handler_class = SUPPORTED_INPUT_HANDLERS.get("default")
                    input_kwargs['queue'] = input_queue
                else:
                    input_handler_class = SUPPORTED_INPUT_HANDLERS.get(
                        self.task_config["tools_config"]["input"]["provider"])

                    if self.task_config['tools_config']['input']['provider'] == 'default':
                        input_kwargs['queue'] = input_queue
                self.tools["input"] = input_handler_class(**input_kwargs)
            else:
                raise "Other input handlers not supported yet"
        output_kwargs = {"websocket": self.websocket}  
        
        if self.task_config["tools_config"]["output"] is None:
            logger.info("Not setting up any output handler as it is none")
        elif self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_HANDLERS.keys():
            if connected_through_dashboard:
                logger.info("Connected through dashboard and hence using default output handler")
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get("default")
                output_kwargs['queue'] = output_queue
            else:
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get(self.task_config["tools_config"]["output"]["provider"])
            
                if self.task_config["tools_config"]["output"]["provider"] == "twilio":
                    output_kwargs['mark_set'] = self.mark_set
                    logger.info(f"Making sure that the sampling rate for output handler is 8000")
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 8000
                    self.task_config['tools_config']['synthesizer']['audio_format'] = 'pcm'
                else:
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 24000
                    output_kwargs['queue'] = output_queue

            self.tools["output"] = output_handler_class(**output_kwargs)
        else:
            raise "Other input handlers not supported yet"

        # Current conversation state
        self.current_request_id = None
        self.previous_request_id = None
        self.llm_rejected_request_ids = set()
        self.llm_processed_request_ids = set()

        # Agent stuff
        # Need to maintain current conversation history and overall persona/history kinda thing. 
        # Soon we will maintain a seperate history for this 
        self.history = [] if conversation_history is None else conversation_history 
        logger.info(f'History {self.history}')
        self.label_flow = []

        # Setup IO SERVICE, TRANSCRIBER, LLM, SYNTHESIZER
        self.llm_task = None
        self.synthesizer_tasks = []

        # state of conversation
        self.was_long_pause = False

        # Call conversations
        self.call_sid = None
        self.stream_sid = None

        # metering
        self.transcriber_duration = 0
        self.synthesizer_characters = 0
        self.ended_by_assistant = False

        self.extracted_data = None
        self.summarized_data = None
        logger.info(f"TASK CONFIG {self.task_config['tools_config'] }")
        self.stream = ( self.task_config["tools_config"]['synthesizer'] is not None and self.task_config["tools_config"]["synthesizer"]["stream"]) and not connected_through_dashboard
        #self.stream = not connected_through_dashboard #Currently we are allowing only realtime conversation based usecases. Hence it'll always be true unless connected through dashboard
        self.is_local = False

        # Memory
        self.cache = cache
        logger.info("task initialization completed")

        # Sequence id for interruption
        self.curr_sequence_id = 0
        self.sequence_ids = set()
        llm_config = {
            "streaming_model": self.task_config["tools_config"]["llm_agent"]["streaming_model"],
            "classification_model": self.task_config["tools_config"]["llm_agent"]["classification_model"]
        }

        # setting transcriber
        if self.task_config["tools_config"]["transcriber"] is not None:
            provider = "playground" if self.connected_through_dashboard else self.task_config["tools_config"]["input"][
                "provider"]
            self.task_config["tools_config"]["transcriber"]["input_queue"] = self.audio_queue
            self.task_config['tools_config']["transcriber"]["output_queue"] = self.transcriber_output_queue
            if self.task_config["tools_config"]["transcriber"]["model"] in SUPPORTED_TRANSCRIBER_MODELS.keys():
                if self.connected_through_dashboard:
                    self.task_config["tools_config"]["transcriber"]["stream"] = False
                transcriber_class = SUPPORTED_TRANSCRIBER_MODELS.get(
                    self.task_config["tools_config"]["transcriber"]["model"])
                self.tools["transcriber"] = transcriber_class(provider, **self.task_config["tools_config"]["transcriber"], **kwargs)
        # setting synthesizer
        logger.info(f"Synthesizer config: {self.task_config['tools_config']['synthesizer']}")
        if self.task_config["tools_config"]["synthesizer"] is not None:
            self.synthesizer_provider = self.task_config["tools_config"]["synthesizer"].pop("provider")
            synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(self.synthesizer_provider)
            provider_config = self.task_config["tools_config"]["synthesizer"].pop("provider_config")
            if self.connected_through_dashboard:
                self.task_config["tools_config"]["synthesizer"]["audio_format"] = "mp3" # Hard code mp3 if we're connected through dashboard
                self.task_config["tools_config"]["synthesizer"]["stream"] = False #Hardcode stream to be False as we don't want to get blocked by a __listen_synthesizer co-routine
            self.tools["synthesizer"] = synthesizer_class(**self.task_config["tools_config"]["synthesizer"], **provider_config, **kwargs)
            llm_config["max_tokens"] = self.task_config["tools_config"]["synthesizer"].get('max_tokens')
            llm_config["buffer_size"] = self.task_config["tools_config"]["synthesizer"].get('buffer_size')

        # setting llm
        if self.task_config["tools_config"]["llm_agent"]["family"] in SUPPORTED_LLM_MODELS.keys():
            llm_class = SUPPORTED_LLM_MODELS.get(self.task_config["tools_config"]["llm_agent"]["family"])
            llm = llm_class(**llm_config, **kwargs)
        else:
            raise Exception(f'LLM {self.task_config["tools_config"]["llm_agent"]["family"]} not supported')

        if self.task_config["task_type"] == "conversation":
            if self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] == "streaming":
                self.tools["llm_agent"] = StreamingContextualAgent(llm)
            elif self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] in ("preprocessed", "formulaic"):
                preprocessed = self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] == "preprocessed"
                logger.info(f"LLM TYPE {type(llm)}")
                self.tools["llm_agent"] = GraphBasedConversationAgent(llm, context_data=self.context_data,
                                                                      prompts=self.prompts, preprocessed=preprocessed)
        elif self.task_config["task_type"] == "extraction":
            logger.info("Setting up extraction agent")
            self.tools["llm_agent"] = ExtractionContextualAgent(llm, prompt=self.system_prompt)
            self.extracted_data = None
        elif self.task_config["task_type"] == "summarization":
            logger.info("Setting up summarization agent")
            self.tools["llm_agent"] = SummarizationContextualAgent(llm, prompt=self.system_prompt)
            self.summarized_data = None

        logger.info("prompt and config setup completed")
    
    ########################
    # Load prompts
    ########################
        
    async def load_prompt(self, assistant_name, task_id, local, **kwargs):
        logger.info("prompt and config setup started")
        self.is_local = local
        if "prompt" in self.task_config["tools_config"]["llm_agent"]:
            self.prompts = {
                "system_prompt": self.task_config["tools_config"]["llm_agent"]["prompt"]
            }
            logger.info(f"Prompt given in llm_agent and hence storing the prompt")
        else:
            prompt_responses = await get_prompt_responses(assistant_id=self.assistant_id,local=self.is_local)
            self.prompts = prompt_responses["task_{}".format(task_id + 1)]

        if "system_prompt" in self.prompts:
            # This isn't a graph based agent
            enriched_prompt = self.prompts["system_prompt"]
            if self.context_data is not None:
                enriched_prompt = update_prompt_with_context(self.prompts["system_prompt"], self.context_data)
            self.system_prompt = {
                'role': "system",
                'content': enriched_prompt
            }
        else:
            self.system_prompt = {
                'role': "system",
                'content': ""
            }
        
        if len(self.system_prompt['content']) == 0:
            self.history =  [] if len(self.history) == 0 else self.history
        else:
            self.history =  [self.system_prompt] if len(self.history) == 0 else [self.system_prompt] + self.history

    ########################
    # LLM task
    ########################

    async def _handle_llm_output(self, next_step, text_chunk, should_bypass_synth, meta_info):
        logger.info("received text from LLM for output processing: {}".format(text_chunk))
        if next_step == "synthesizer" and not should_bypass_synth:
            task = asyncio.gather(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
            self.synthesizer_tasks.append(asyncio.ensure_future(task))
        elif self.tools["output"] is not None:
            await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))

    def _get_next_step(self, sequence, origin):
        try:
            return next((self.pipelines[sequence][i + 1] for i in range(len(self.pipelines[sequence]) - 1) if
                         self.pipelines[sequence][i] == origin), "output")
        except Exception as e:
            logger.error(f"Error getting next step: {e}")

    def _set_call_details(self, message):
        if self.call_sid is not None and self.stream_sid is not None and "call_sid" not in message['meta_info'] and "stream_sid" not in message['meta_info']:
            return

        if "call_sid" in message['meta_info']:
            self.call_sid = message['meta_info']["call_sid"]
        if "stream_sid" in message:
            self.stream_sid = message['meta_info']["stream_sid"]

    async def _process_followup_task(self, message, sequence, meta_info):
        message = format_messages(self.input_parameters["messages"])  # Remove the initial system prompt
        self.history.append({
            'role': 'user',
            'content': message
        })

        json_data = await self.tools["llm_agent"].generate(self.history)
        if "summary" in json_data:
            logger.info(f'Summary {json_data["summary"]}')
            self.summarized_data = json_data["summary"]
        else:
            json_data = clean_json_string(json_data)
            logger.info(f"After replacing {json_data}")
            json_data = json.loads(json_data)
            self.extracted_data = json_data
        logger.info("Done")

    async def _process_conversation_preprocessed_task(self, message, sequence, meta_info):
        if self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
            llm_response = ""
            self.history.append({
                'role': 'user',
                'content': message['data']
            })
            start_time = time.time()

            async for text_chunk in self.tools['llm_agent'].generate(self.history, stream=True, synthesize=True,
                                                                     label_flow=self.label_flow):
                if text_chunk == "<end_of_conversation>":
                    logger.info("Got end of conversation. I'm stopping now")
                    self.conversation_ended = True
                    await asyncio.sleep(5) #Make sure that the message is passed over and complete before cutting the handler
                    await self.tools["input"].stop_handler()
                    logger.info("Stopped input handler")
                    if "transcriber" in self.tools and not self.connected_through_dashboard:
                        logger.info("Stopping transcriber")
                        await self.tools["transcriber"].toggle_connection()
                        await asyncio.sleep(5)  # Making sure whatever message was passed is over
                    return
                logger.info(f"Text chunk {text_chunk}")
                if is_valid_md5(text_chunk):
                    self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=True))))
                else:
                    self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=False))))

    async def _process_conversation_formulaic_task(self, message, sequence, meta_info):
        self.history.append({
            'role': 'user',
            'content': message['data']
        })
        start_time = time.time()
        llm_response = ""
        logger.info("Agent flow is formulaic and hence moving smoothly")
        async for text_chunk in self.tools['llm_agent'].generate(self.history, stream=True, synthesize=True):
            if is_valid_md5(text_chunk):
                self.synthesizer_tasks.append(asyncio.create_task(
                    self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=True))))
            else:
                # TODO Make it more modular
                llm_response += " " +text_chunk
                next_step = self._get_next_step(sequence, "llm")
                if next_step == "synthesizer":
                    task = asyncio.gather(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
                    self.synthesizer_tasks.append(asyncio.ensure_future(task))
                else:
                    logger.info(f"Sending output text {sequence}")
                    await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                    self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=False))))

    async def _process_conversation_task(self, message, sequence, meta_info):
        next_step = None
        logger.info("agent flow is not preprocessed")
        llm_response = ""
        self.history.append({
            'role': 'user',
            'content': message['data']
        })
        start_time = time.time()
        should_bypass_synth = 'bypass_synth' in meta_info and meta_info['bypass_synth'] == True
        next_step = self._get_next_step(sequence, "llm")
        self.curr_sequence_id +=1
        meta_info["sequence_id"] = self.curr_sequence_id
        cache_response =  self.cache.get(get_md5_hash(message['data'])) if self.cache is not None else None
        if cache_response is not None:
            logger.info("It was a cache hit and hence simply returning")
            await self._handle_llm_output(next_step, cache_response, should_bypass_synth, meta_info)
        else:
            async for llm_message in self.tools['llm_agent'].generate(self.history, synthesize=True):
                text_chunk, end_of_llm_stream = llm_message
                logger.info(f"###### time to get the first chunk {time.time() - start_time} {text_chunk}")
                llm_response += " " + text_chunk
                if self.stream:
                    if end_of_llm_stream:
                        meta_info["end_of_llm_stream"] = True
                    await self._handle_llm_output(next_step, text_chunk, should_bypass_synth, meta_info)
                    
            if not self.stream:
                meta_info["end_of_llm_stream"] = True
                await self._handle_llm_output(next_step, llm_response, should_bypass_synth, meta_info)
        if self.current_request_id in self.llm_rejected_request_ids:
            logger.info("User spoke while LLM was generating response")
        else:
            self.history.append({"role": "assistant", "content": llm_response})

            # TODO : Write a better check for completion prompt 
            #answer = await self.tools["llm_agent"].check_for_completion(self.history)
            answer = False
            if answer:
                logger.info("Got end of conversation. I'm stopping now")
                self.conversation_ended = True
                self.ended_by_assistant = True
                await self.tools["input"].stop_handler()
                logger.info("Stopped input handler")
                if "transcriber" in self.tools and not self.connected_through_dashboard:
                    logger.info("Stopping transcriber")
                    await self.tools["transcriber"].toggle_connection()
                    await asyncio.sleep(5)  # Making sure whatever message was passed is over
                return

            self.llm_processed_request_ids.add(self.current_request_id)
            llm_response = ""


    def _extract_sequence_and_meta(self, message):
        sequence, meta_info = None, None
        if isinstance(message, dict) and "meta_info" in message:
            self._set_call_details(message)
            sequence = message["meta_info"]["sequence"]
            meta_info = message["meta_info"]
        return sequence, meta_info

    def _is_extraction_task(self):
        return self.task_config["task_type"] == "extraction"

    def _is_summarization_task(self):
        return self.task_config["task_type"] == "summarization"

    def _is_conversation_task(self):
        return self.task_config["task_type"] == "conversation"

    def _is_preprocessed_flow(self):
        return self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed"

    def _is_formulaic_flow(self):
        return self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "formulaic"

    # This is used only in the case it's a text based chatbot
    async def _listen_llm_input_queue(self):
        logger.info(
            f"Starting listening to LLM queue as either Connected to dashboard = {self.connected_through_dashboard} or  it's a textual chat agent {self.textual_chat_agent}")
        while True:
            try:
                ws_data_packet = await self.queues["llm"].get()
                logger.info(f"ws_data_packet {ws_data_packet}")
                bos_packet = create_ws_data_packet("<beginning_of_stream>", ws_data_packet['meta_info'])
                await self.tools["output"].handle(bos_packet)
                await self._run_llm_task(
                    ws_data_packet)  # In case s3 is down and it's an audio processing job, this might produce blank message on the frontend of playground.
                eos_packet = create_ws_data_packet("<end_of_stream>", ws_data_packet['meta_info'])
                await self.tools["output"].handle(eos_packet)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Something went wrong with LLM queue {e}")
                break

    async def _run_llm_task(self, message):
        logger.info("running llm based agent")
        sequence, meta_info = self._extract_sequence_and_meta(message)

        try:
            if self._is_extraction_task() or self._is_summarization_task():
                await self._process_followup_task(message, sequence, meta_info)
            elif self._is_conversation_task():
                if self._is_preprocessed_flow():
                    await self._process_conversation_preprocessed_task(message, sequence, meta_info)

                elif self._is_formulaic_flow():
                    await self._process_conversation_formulaic_task(message, sequence, meta_info)
                else:
                    await self._process_conversation_task(message, sequence, meta_info)
            else:
                logger.error("unsupported task type: {}".format(self.task_config["task_type"]))
            self.llm_task = None
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Something went wrong in llm: {e}")

    async def process_transcriber_request(self, meta_info):
        if not self.current_request_id or self.current_request_id != meta_info["request_id"]:
            self.previous_request_id, self.current_request_id = self.current_request_id, meta_info["request_id"]

        sequence = meta_info["sequence"]

        # check if previous request id is not in transmitted request id
        if self.previous_request_id is None:
            is_first_message = True
        elif self.previous_request_id not in self.llm_processed_request_ids:
            self.llm_rejected_request_ids.add(self.previous_request_id)
        else:
            skip_append_to_data = False
        return sequence

    async def process_interruption(self):
        logger.info("Handling interruption")
        self.sequence_ids = set() #Remove all the sequence ids so subsequent won't be processed
        await self.tools["output"].handle_interruption()
        
        if self.llm_task is not None:
            self.llm_task.cancel()
            self.llm_task = None
            self.was_long_pause = True        

    ########################
    # Transcriber task
    ########################

    async def _handle_transcriber_output(self, next_task, transcriber_message, meta_info):
        if next_task == "llm":
            meta_info["origin"] = "transcriber"
            self.llm_task = asyncio.create_task(
                self._run_llm_task(create_ws_data_packet(transcriber_message, meta_info)))
        elif next_task == "synthesizer":
            self.synthesizer_tasks.append(asyncio.create_task(
                self._synthesize(create_ws_data_packet(transcriber_message, meta_info))))
        else:
            logger.info(f"Need to separate out output task")

    async def _listen_transcriber(self):
        transcriber_message = ""
        logger.info(f"Starting transcriber task")
        try:
            while True:
                if self.stream:
                    message = await self.transcriber_output_queue.get()
                    if message['data'] == "transcriber_connection_closed":
                        self.transcriber_duration += message['meta_info']["transcriber_duration"]
                        logger.info("transcriber connection closed")
                        break

                    self._set_call_details(message)
                    meta_info = message["meta_info"]
                    sequence = await self.process_transcriber_request(meta_info)

                    if message['data'] == "INTERRUPTION":
                        await self.process_interruption()
                    if message['data'] == "TRANSCRIBER_BEGIN":
                        logger.info("starting transcriber stream")
                        #await self.process_interruption()
                        continue
                    elif message['data'] == "TRANSCRIBER_END":
                        logger.info("transcriber stream and preparing the next step")
                        next_task = self._get_next_step(sequence, "transcriber")
                        logger.info(f'got the next task {next_task}')
                        if self.was_long_pause:
                            logger.info(
                                f"Seems like there was a long pause {self.history[-1]['content']} , {transcriber_message}")
                            message = self.history[-1]['content'] + " " + transcriber_message
                            self.history = self.history[:-1]
                            self.was_long_pause = False

                        logger.info(f'invoking next_task {next_task} with transcriber_message: {transcriber_message}')
                        await self._handle_transcriber_output(next_task, transcriber_message, meta_info)
                        transcriber_message = ""
                        continue
                    else:
                        logger.info("processed text from transcriber: {}".format(message['data']))
                        transcriber_message += message['data']
                else:
                    logger.info("Not a streaming conversation. Hence getting a full blown transcript")
                    message = await self.transcriber_output_queue.get()
                    logger.info(f"message from transcriber {message}")
                    if message['data'] == "transcriber_connection_closed":
                        self.transcriber_duration += message['meta_info']["transcriber_duration"]
                        logger.info("transcriber connection closed")
                        break
                    sequence = message["meta_info"]["sequence"]
                    next_task = self._get_next_step(sequence, "transcriber")
                    self.transcriber_duration += message["meta_info"]["transcriber_duration"] if "transcriber_duration" in message["meta_info"] else 0
                    await self._handle_transcriber_output(next_task, message['data'], message["meta_info"])
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in transcriber {e}")

    async def __listen_synthesizer(self):
        try:
            if self.stream and self.synthesizer_provider != "polly":
                logger.info("Opening websocket connection to synthesizer")
                await self.tools["synthesizer"].open_connection()
            while True:
                logger.info("Listening to synthesizer")
                async for message in self.tools["synthesizer"].generate():
                    if not self.conversation_ended and message["meta_info"]["sequence_id"] in self.sequence_ids:
                        logger.info(f"{message['meta_info']['sequence_id'] } is in sequence ids  {self.sequence_ids} and hence removing the sequence ids ")
                        if self.task_config["tools_config"]["output"]["provider"] == "twilio" and not self.connected_through_dashboard and self.synthesizer_provider == "elevenlabs":
                                message['data'] = wav_bytes_to_pcm(message['data'])
                        await self.tools["output"].handle(message)
                    else:
                        logger.info(f"{message['meta_info']['sequence_id']} is not in sequence ids  {self.sequence_ids} and hence not sending to output")                
                        await asyncio.sleep(0.5)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer {e}")

    async def _synthesize(self, message):
        meta_info = message["meta_info"]
        text = message["data"]
        meta_info["type"] = "audio"
        try:
            if meta_info["is_md5_hash"]:
                logger.info('sending preprocessed audio response to {}'.format(
                    self.task_config["tools_config"]["output"]["provider"]))
                
                #TODO: Either load IVR audio into memory before call or user s3 iter_cunks
                # This will help with interruption in IVR
                if self.connected_through_dashboard or self.task_config['tools_config']['output'] == "default":
                    audio_chunk = await get_raw_audio_bytes_from_base64(self.assistant_name, text,
                                                                    self.task_config["tools_config"]["output"][
                                                                        "format"], local=self.is_local,
                                                                    assistant_id=self.assistant_id)
                    await self.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))
                else:
                    audio_chunk = await get_raw_audio_bytes_from_base64(self.assistant_name, text,
                                                                    'pcm', local=self.is_local,
                                                                    assistant_id=self.assistant_id)
                    #await self.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))
                    for chunk in  yield_chunks_from_memory(audio_chunk, chunk_size=16384):
                        await self.tools["output"].handle(create_ws_data_packet(chunk, meta_info))
            elif self.synthesizer_provider in SUPPORTED_SYNTHESIZER_MODELS.keys():
                self.sequence_ids.add(meta_info["sequence_id"])
                logger.info(f"After adding into sequence id {self.sequence_ids}")
                logger.info('sending text to {} for generation: {} '.format(self.synthesizer_provider, text))
                self.synthesizer_characters += len(text)
                await self.tools["synthesizer"].push(message)
            else:
                logger.info("other synthesizer models not supported yet")
        except Exception as e:
            logger.error(f"Error in synthesizer: {e}")

    async def run(self):
        """
        Run will start a listener that will continuously listen to the websocket
        - If type is "audio": it'll pass it to transcriber
            - Transcriber will pass it deepgram
            - Deepgram will respond
    
        """
        try:
            if self.task_id == 0:
                # Create transcriber and synthesizer tasks
                logger.info("starting task_id {}".format(self.task_id))
                tasks = [asyncio.create_task(self.tools['input'].handle())]
                if "transcriber" in self.tools:
                    tasks.append(asyncio.create_task(self._listen_transcriber()))
                    self.transcriber_task = asyncio.create_task(self.tools["transcriber"].run())

                if self.connected_through_dashboard and self.task_config['task_type'] == "conversation":
                    logger.info(
                        "Since it's connected through dashboard, I'll run listen_llm_tas too in case user wants to simply text")
                    self.llm_queue_task = asyncio.create_task(self._listen_llm_input_queue())
                
                if "synthesizer" in self.tools and self._is_conversation_task():
                    logger.info("Starting synthesizer task")
                    try:
                        self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
                    except asyncio.CancelledError as e:
                        logger.error(f'Synth task got cancelled {e}')
                        traceback.print_exc()                    
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError as e:
                    logger.error(f'task got cancelled {e}')
                    traceback.print_exc()
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error: {e}")

                logger.info("Conversation completed")
            else:
                # Run agent followup tasks
                try:
                    await self._run_llm_task(self.input_parameters)
                except Exception as e:
                    logger.error(f"Could not do llm call: {e}")
                    raise Exception(e)

        except asyncio.CancelledError as e:
            # Cancel all tasks on cancel
            traceback.print_exc()
            self.transcriber_task.cancel()
            self.handle_cancellation(f"Websocket got cancelled {self.task_id}")

        except Exception as e:
            # Cancel all tasks on error
            self.handle_cancellation(f"Exception occurred {e}")
            raise Exception(e)

        finally:
            # Construct output
            if "synthesizer" in self.tools and self.synthesizer_task is not None:   
                self.synthesizer_task.cancel()
            if self.task_id == 0:
                output = {"messages": self.history, "conversation_time": time.time() - self.start_time,
                          "label_flow": self.label_flow, "call_sid": self.call_sid, "stream_sid": self.stream_sid,
                          "transcriber_duration": self.transcriber_duration,
                          "synthesizer_characters": self.synthesizer_characters, "ended_by_assistant": self.ended_by_assistant}
            else:
                output = self.input_parameters
                if self.task_config["task_type"] == "extraction":
                    output = { "extracted_data" : self.extracted_data, "task_type": "extraction"}
                elif self.task_config["task_type"] == "summarization":
                    output = {"summary" : self.summarized_data, "task_type": "summarization"}

            return output

    def handle_cancellation(self, message):
        try:
            # Cancel all tasks on cancellation
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if "synthesizer" in self.tools and self.synthesizer_task:
                self.synthesizer_task.cancel()
            logger.info(f"tasks {len(tasks)}")
            for task in tasks:
                logger.info(f"Cancelling task {task.get_name()}")
                task.cancel()
            logger.info(message)
        except Exception as e:
            traceback.print_exc()
            logger.info(e)