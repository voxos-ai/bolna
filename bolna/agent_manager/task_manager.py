import asyncio
from collections import defaultdict
import os
import random
import traceback
import time
import json
import uuid
import copy
from datetime import datetime

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from bolna.memory.cache.vector_cache import VectorCache
from .base_manager import BaseManager
from bolna.agent_types import *
from bolna.providers import *
from bolna.helpers.utils import calculate_audio_duration, create_ws_data_packet, get_file_names_in_directory, get_raw_audio_bytes, is_valid_md5, \
    get_required_input_types, format_messages, get_prompt_responses, list_number_of_wav_files_in_directory, resample, save_audio_file_to_s3, update_prompt_with_context, get_md5_hash, clean_json_string, wav_bytes_to_pcm, write_request_logs, yield_chunks_from_memory
from bolna.helpers.logger_config import configure_logger
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import FastEmbedEncoder

asyncio.get_event_loop().set_debug(True)
logger = configure_logger(__name__)


class TaskManager(BaseManager):
    def __init__(self, assistant_name, task_id, task, ws, input_parameters=None, context_data=None,
                 assistant_id=None, run_id=None, connected_through_dashboard=False, cache=None,
                 input_queue=None, conversation_history=None, output_queue=None, yield_chunks=True, **kwargs):
        super().__init__()
        # Latency and logging 
        self.latency_dict = defaultdict(dict)
        self.kwargs = kwargs

        logger.info(f"doing task {task}")
        self.task_id = task_id
        self.assistant_name = assistant_name
        self.tools = {}
        self.websocket = ws
        self.task_config = task
        self.context_data = context_data
        self.connected_through_dashboard = connected_through_dashboard
        self.enforce_streaming = kwargs.get("enforce_streaming", False)
        self.callee_silent = True
        self.yield_chunks = yield_chunks
        self.kwargs["process_interim_results"] = "true" if task.get("optimize_latency", False) is True else "false"
        logger.info(f"Processing interim results {self.kwargs['process_interim_results'] }")
        # Set up communication queues between processes
        self.audio_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.synthesizer_queue = asyncio.Queue()
        self.transcriber_output_queue = asyncio.Queue()
        self.queues = {
            "transcriber": self.audio_queue,
            "llm": self.llm_queue,
            "synthesizer": self.synthesizer_queue
        }
        self.pipelines = task['toolchain']['pipelines']
        self.textual_chat_agent = False
        if task['toolchain']['pipelines'][0] == "llm" and task["tools_config"]["llm_agent"][
            "agent_task"] == "conversation":
            self.textual_chat_agent = False

        # Assistant persistance stuff
        self.assistant_id = assistant_id
        self.run_id = run_id
        self.mark_set = set()
        
        self.conversation_ended = False

        # Prompts
        self.prompts, self.system_prompt = {}, {}
        self.input_parameters = input_parameters
        
        # Recording
        self.should_record = False
        self.conversation_recording= {
            "input": {
                'data': b'',
                'started': time.time()
            },
            "output": [],
            "metadata": {
                "started": 0
            }
        }
        #IO HANDLERS
        if task_id == 0:
            self.should_record = self.task_config["tools_config"]["output"]["provider"] == 'default' and self.enforce_streaming #In this case, this is a websocket connection and we should record 
            self.__setup_input_handlers(connected_through_dashboard, input_queue, self.should_record)
        self.__setup_output_handlers(connected_through_dashboard, output_queue)

        # Agent stuff
        # Need to maintain current conversation history and overall persona/history kinda thing. 
        # Soon we will maintain a seperate history for this 
        self.history = [] if conversation_history is None else conversation_history 
        self.interim_history = copy.deepcopy(self.history.copy())
        logger.info(f'History {self.history}')
        self.label_flow = []

        # Setup IO SERVICE, TRANSCRIBER, LLM, SYNTHESIZER
        self.llm_task = None
        self.synthesizer_tasks = []
        self.synthesizer_task = None

        # state of conversation
        self.current_request_id = None
        self.previous_request_id = None
        self.llm_rejected_request_ids = set()
        self.llm_processed_request_ids = set()
        self.was_long_pause = False
        self.buffers = []
        self.should_respond = False
        self.last_response_time = time.time()
        self.is_an_ivr_call = self._is_conversation_task() and self._is_preprocessed_flow() and not self.connected_through_dashboard
        self.consider_next_transcript_after = time.time()
        self.duration_to_prevent_accidental_interruption = 3 if self.is_an_ivr_call else 0
        self.callee_speaking = False
        self.callee_speaking_start_time = -1

        # Call conversations
        self.call_sid = None
        self.stream_sid = None

        # metering
        self.transcriber_duration = 0
        self.synthesizer_characters = 0
        self.ended_by_assistant = False
        self.start_time = time.time()
        

        #Tasks
        self.extracted_data = None
        self.summarized_data = None
        logger.info(f"TASK CONFIG {self.task_config['tools_config'] }")
        self.stream = (self.task_config["tools_config"]['synthesizer'] is not None and self.task_config["tools_config"]["synthesizer"]["stream"]) and (self.enforce_streaming or not self.connected_through_dashboard)
        #self.stream = not connected_through_dashboard #Currently we are allowing only realtime conversation based usecases. Hence it'll always be true unless connected through dashboard
        self.is_local = False
        llm_config = None
        if self.task_config["tools_config"]["llm_agent"] is not None:
            llm_config = {
                "model": self.task_config["tools_config"]["llm_agent"]["model"],
                "max_tokens": self.task_config["tools_config"]["llm_agent"]["max_tokens"]
            }
        
        # Output stuff
        self.output_task = None
        self.buffered_output_queue = asyncio.Queue()

        # Memory
        self.cache = cache
        logger.info("task initialization completed")

        # Sequence id for interruption
        self.curr_sequence_id = 0
        self.sequence_ids = set()
        
        # setting transcriber
        self.__setup_transcriber()
        # setting synthesizer
        self.__setup_synthesizer(llm_config)
        # setting llm
        llm = self.__setup_llm(llm_config)
        #Setup tasks
        self.__setup_tasks(llm)


        #setup request logs
        self.request_logs = []

        # for long pauses and rushing
        
        if task_id == 0:
            self.output_chunk_size = 16384 if self.sampling_rate == 24000 else 4096 #0.5 second chunk size for calls 
            # For nitro
            self.nitro = True 
            self.minimum_wait_duration = self.task_config["tools_config"]["transcriber"]["endpointing"]
            logger.info(f"minimum wait duration {self.minimum_wait_duration}")
            self.last_spoken_timestamp = time.time() * 1000
            self.incremental_delay = task.get("incremental_delay", 100)
            self.required_delay_before_speaking = max(self.minimum_wait_duration - self.incremental_delay, 0)  #Everytime we get a message we increase it by 100 miliseconds 
            self.time_since_first_interim_result  = -1

        #Cut conversation
        self.hang_conversation_after = task.get("hangup_after_silence", 10)
        self.last_transmitted_timesatamp = 0
        self.let_remaining_audio_pass_through = False #Will be used to let remaining audio pass through in case of utterenceEnd event and there's still audio left to be sent
        self.use_llm_to_determine_hangup = task.get("hangup_after_LLMCall", False)
        self.check_for_completion_prompt = task.get("call_cancellation_prompt", None)
        if self.check_for_completion_prompt is not None:
            completion_json_format = {"answer": "A simple Yes or No based on if you should cut the phone or not"}
            self.check_for_completion_prompt = f"{self.check_for_completion_prompt}\nYour response should be in the following json format\n{completion_json_format}"
        self.check_for_completion_llm = os.getenv("CHECK_FOR_COMPLETION_LLM")
        self.time_since_last_spoken_human_word = 0 

        #Handling accidental interruption
        self.number_of_words_for_interruption = task.get("number_of_words_for_interruption", 3)
        self.started_transmitting_audio = False
        #self.interruption_backoff_period = 1000 #task.get("interruption_backoff_period", 300) #this is the amount of time output loop will sleep before sending next audio
        self.use_llm_for_hanging_up = task.get("hangup_after_LLMCall", False)
        self.allow_extra_sleep = False #It'll help us to back off as soon as we hear interruption for a while

        # Conversation
        if task_id == 0:
            self.routes = task['tools_config']['llm_agent'].get("routes", None)
            self.route_layer = None
            if self.routes:
                self.__setup_routes(self.routes)
        
        #Backchanneling
        self.should_backchannel = task.get("backchanneling", True)
        self.backchanneling_task = None
        self.backchanneling_start_delay = task.get("backchanneling_start_delay", 5)
        self.backchanneling_message_gap = task.get("backchanneling_message_gap", 2) #Amount of duration co routine will sleep
        if self.should_backchannel:
            logger.info(f"Should backchannel")
            self.backchanneling_audios = f'{kwargs.get("backchanneling_audio_location", os.getenv("BACKCHANNELING_PRESETS_DIR"))}/{self.synthesizer_voice.lower()}'
            #self.num_files = list_number_of_wav_files_in_directory(self.backchanneling_audios)
            try:
                self.filenames = get_file_names_in_directory(self.backchanneling_audios)
                logger.info(f"Backchanneling audio location {self.backchanneling_audios}")
            except Exception as e:
                logger.info(f"Something went wrong an putting should backchannel to false")
                self.should_backchannel = False
        else:
            logger.info(f"Not setting up backchanneling")
            self.backchanneling_audio_map = []
            
            
    def __setup_routes(self, routes):
        embedding_model = routes.get("embedding_model", os.getenv("ROUTE_EMBEDDING_MODEL"))
        self.route_encoder = FastEmbedEncoder(name=embedding_model)

        routes_list = []
        self.vector_caches = {}
        self.route_responses_dict = {}
        for route in routes['routes']:
            logger.info(f"Setting up route {route}")
            utterances = route['utterances']
            r = Route(
                name = route['route_name'],
                utterances= utterances,
                score_threshold = route['score_threshold']
            )
            utterance_response_dict = {}
            if type(route['response']) is list and len(route['response']) == len(route['utterances']):
                for i, utterance in enumerate(utterances):
                    utterance_response_dict[utterance] =  route['response'][i]
                self.route_responses_dict[route['route_name']] = utterance_response_dict
            elif type(route['response']) is str:
                self.route_responses_dict[route['route_name']] = route['response']
            else:
                raise Exception("Invalid number of responses for the responses array")

                
            routes_list.append(r)
            
            if type(route['response']) is list:
                logger.info(f"Setting up vector cache for {route} and embedding model {embedding_model}")
                vector_cache = VectorCache(embedding_model = embedding_model)
                vector_cache.set(utterances)
                self.vector_caches[route['route_name']] = vector_cache
            
        self.route_layer = RouteLayer(encoder=self.route_encoder, routes=routes_list)
        logger.info("Routes are set")

    def __setup_output_handlers(self, connected_through_dashboard, output_queue):
        output_kwargs = {"websocket": self.websocket}  
        
        if self.task_config["tools_config"]["output"] is None:
            logger.info("Not setting up any output handler as it is none")
        elif self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_HANDLERS.keys():
            if connected_through_dashboard:
                logger.info("Connected through dashboard and hence using default output handler")
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get("default")
                output_kwargs['queue'] = output_queue
                self.sampling_rate = 24000
            else:
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get(self.task_config["tools_config"]["output"]["provider"])
            
                if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS.keys():
                    output_kwargs['mark_set'] = self.mark_set
                    logger.info(f"Making sure that the sampling rate for output handler is 8000")
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 8000
                    self.task_config['tools_config']['synthesizer']['audio_format'] = 'pcm'
                else:
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 24000
                    output_kwargs['queue'] = output_queue
                self.sampling_rate = self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate']

            self.tools["output"] = output_handler_class(**output_kwargs)
        else:
            raise "Other input handlers not supported yet"

    def __setup_input_handlers(self, connected_through_dashboard, input_queue, should_record):
        if self.task_config["tools_config"]["input"]["provider"] in SUPPORTED_INPUT_HANDLERS.keys():
            logger.info(f"Connected through dashboard {connected_through_dashboard}")
            input_kwargs = {"queues": self.queues,
                            "websocket": self.websocket,
                            "input_types": get_required_input_types(self.task_config),
                            "mark_set": self.mark_set,
                            "connected_through_dashboard": self.connected_through_dashboard}
            if should_record:
                input_kwargs['conversation_recording'] = self.conversation_recording

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

    def __setup_transcriber(self):
        if self.task_config["tools_config"]["transcriber"] is not None:
            provider = "playground" if self.connected_through_dashboard else self.task_config["tools_config"]["input"][
                "provider"]
            self.task_config["tools_config"]["transcriber"]["input_queue"] = self.audio_queue
            self.task_config['tools_config']["transcriber"]["output_queue"] = self.transcriber_output_queue
            if self.task_config["tools_config"]["transcriber"]["model"] in SUPPORTED_TRANSCRIBER_MODELS.keys():
                if self.connected_through_dashboard:
                    self.task_config["tools_config"]["transcriber"]["stream"] = True if self.enforce_streaming else False
                    logger.info(f'self.task_config["tools_config"]["transcriber"]["stream"] {self.task_config["tools_config"]["transcriber"]["stream"]} self.enforce_streaming {self.enforce_streaming}')
                transcriber_class = SUPPORTED_TRANSCRIBER_MODELS.get(
                    self.task_config["tools_config"]["transcriber"]["model"])
                self.tools["transcriber"] = transcriber_class(provider, **self.task_config["tools_config"]["transcriber"], **self.kwargs)

    def __setup_synthesizer(self, llm_config):
        self.synthesizer_cache = InmemoryScalarCache()
        
        logger.info(f"Synthesizer config: {self.task_config['tools_config']['synthesizer']}")
        if self._is_conversation_task():
            self.kwargs["use_turbo"] = self.task_config["tools_config"]["transcriber"]["language"] == "en"
        if self.task_config["tools_config"]["synthesizer"] is not None:
            self.synthesizer_provider = self.task_config["tools_config"]["synthesizer"].pop("provider")
            synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(self.synthesizer_provider)
            provider_config = self.task_config["tools_config"]["synthesizer"].pop("provider_config")
            self.synthesizer_voice = provider_config["voice"]
            if self.connected_through_dashboard:
                self.task_config["tools_config"]["synthesizer"]["audio_format"] = "mp3" # Hard code mp3 if we're connected through dashboard
                self.task_config["tools_config"]["synthesizer"]["stream"] = True if self.enforce_streaming else False #Hardcode stream to be False as we don't want to get blocked by a __listen_synthesizer co-routine
        
            self.tools["synthesizer"] = synthesizer_class(**self.task_config["tools_config"]["synthesizer"], **provider_config, **self.kwargs, cache = self.synthesizer_cache)
            if self.task_config["tools_config"]["llm_agent"] is not None:
                llm_config["buffer_size"] = self.task_config["tools_config"]["synthesizer"].get('buffer_size')

    def __setup_llm(self, llm_config):
        if self.task_config["tools_config"]["llm_agent"] is not None:
            logger.info(f'### PROVIDER {self.task_config["tools_config"]["llm_agent"]["provider"] }')
            if self.task_config["tools_config"]["llm_agent"]["provider"] in SUPPORTED_LLM_PROVIDERS.keys():
                llm_class = SUPPORTED_LLM_PROVIDERS.get(self.task_config["tools_config"]["llm_agent"]["provider"])
                logger.info(f"LLM CONFIG {llm_config}")
                llm = llm_class(**llm_config, **self.kwargs)
                return llm
            else:
                raise Exception(f'LLM {self.task_config["tools_config"]["llm_agent"]["provider"]} not supported')

    def __setup_tasks(self, llm):
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
        elif self.task_config["task_type"] == "webhook":
            zap_url = self.task_config["tools_config"]["api_tools"]["webhookURL"]
            logger.info(f"Zap URL {zap_url}")
            self.tools["webhook_agent"] = ZapierAgent(zap_url=zap_url)

        logger.info("prompt and config setup completed")
        
    ########################
    # Helper methods
    ########################
    async def load_prompt(self, assistant_name, task_id, local, **kwargs):
        logger.info("prompt and config setup started")
        if self.task_config["task_type"] == "webhook":
            return
        self.is_local = local
        if "prompt" in self.task_config["tools_config"]["llm_agent"]:
            today = datetime.now().strftime("%A, %B %d, %Y")
            self.prompts = {
                "system_prompt": f'{self.task_config["tools_config"]["llm_agent"]["prompt"]} \n### Date\n Today\'s Date is {today}'
            }
            logger.info(f"Prompt given in llm_agent and hence storing the prompt")
        else:
            prompt_responses = kwargs.get('prompt_responses', None)
            if not prompt_responses:
                prompt_responses = await get_prompt_responses(assistant_id=self.assistant_id, local=self.is_local)
            self.prompts = prompt_responses["task_{}".format(task_id + 1)]
            if self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
                self.tools["llm_agent"].load_prompts_and_create_graph(self.prompts)

        if "system_prompt" in self.prompts:
            # This isn't a graph based agent
            enriched_prompt = self.prompts["system_prompt"]
            if self.context_data is not None:
                enriched_prompt = update_prompt_with_context(self.prompts["system_prompt"], self.context_data)
                self.prompts["system_prompt"] = enriched_prompt
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
            self.history = [] if len(self.history) == 0 else self.history
        else:
            self.history = [self.system_prompt] if len(self.history) == 0 else [self.system_prompt] + self.history

        self.interim_history = copy.deepcopy(self.history)

    def __process_stop_words(self, text_chunk, meta_info):
         #THis is to remove stop words. Really helpful in smaller 7B models
        if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"] and "user" in text_chunk[-5:].lower():
            if text_chunk[-5:].lower() == "user:":
                text_chunk = text_chunk[:-5]
            elif text_chunk[-4:].lower() == "user":
                text_chunk = text_chunk[:-4]

        index = text_chunk.find("AI")
        if index != -1:
            text_chunk = text_chunk[index+2:]
        return text_chunk
    
    async def process_interruption(self):
        logger.info(f"Handling interruption sequenxce ids {self.sequence_ids}")
        await self.__cleanup_downstream_tasks()    

    async def __cleanup_downstream_tasks(self):
        logger.info(f"Cleaning up downstream task")
        start_time = time.time()
        await self.tools["output"].handle_interruption()
        self.sequence_ids = set()
        
        #Stop the output loop first so that we do not transmit anything else
        if self.output_task is not None:
            logger.info(f"Cancelling output task")
            self.output_task.cancel()

        if self.llm_task is not None:
            logger.info(f"Cancelling LLM Task")
            self.llm_task.cancel()
            self.llm_task = None

        # self.synthesizer_task.cancel()
        # self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
        for task in self.synthesizer_tasks:
            task.cancel()
        
        self.synthesizer_tasks = []

        logger.info(f"Synth Task cancelled seconds")
        if not self.buffered_output_queue.empty():
            logger.info(f"Output queue was not empty and hence emptying it")
            self.buffered_output_queue = asyncio.Queue()
        
        #restart output task
        self.output_task = asyncio.create_task(self.__process_output_loop())
        logger.info(f"Cleaning up downstream tasks sequenxce ids {self.sequence_ids}. Time taken to send a clear message {time.time() - start_time}")

    def __get_updated_meta_info(self, meta_info = None):
        #This is used in case there's silence from callee's side
        if meta_info is None:
            meta_info = self.tools["transcriber"].get_meta_info()
        meta_info_copy = meta_info.copy()
        self.curr_sequence_id +=1
        meta_info_copy["sequence_id"] = self.curr_sequence_id
        self.sequence_ids.add(meta_info_copy["sequence_id"])
        return meta_info_copy
    
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

    async def _process_followup_task(self, message=None):
        logger.info(f" TASK CONFIG  {self.task_config['task_type']}")
        if self.task_config["task_type"] == "webhook":
            logger.info(f"Input patrameters {self.input_parameters}")
            logger.info(f"DOING THE POST REQUEST TO ZAPIER WEBHOOK {self.input_parameters['extraction_details']}")
            self.webhook_response = await self.tools["webhook_agent"].execute(self.input_parameters['extraction_details'])
            logger.info(f"Response from the server {self.webhook_response}")
        else:
            message = format_messages(self.input_parameters["messages"])  # Remove the initial system prompt
            self.history.append({
                'role': 'user',
                'content': message
            })

            today = datetime.now().strftime("%A, %B %d, %Y")
            self.history[0]['content'] += f"\n Today's Date is {today}"

            json_data = await self.tools["llm_agent"].generate(self.history)
            if self.task_config["task_type"] == "summarization":
                logger.info(f'Summary {json_data["summary"]}')
                self.summarized_data = json_data["summary"]
                logger.info(f"self.summarize {self.summarized_data}")
            else:
                json_data = clean_json_string(json_data)
                logger.info(f"After replacing {json_data}")
                if type(json_data) is not dict:
                    json_data = json.loads(json_data)
                self.extracted_data = json_data
        logger.info("Done")

    async def __process_end_of_conversation(self):
        logger.info("Got end of conversation. I'm stopping now")
        self.conversation_ended = True
        self.ended_by_assistant = True
        await self.tools["input"].stop_handler()
        logger.info("Stopped input handler")
        if "transcriber" in self.tools and not self.connected_through_dashboard:
            logger.info("Stopping transcriber")
            await self.tools["transcriber"].toggle_connection()
            await asyncio.sleep(5)  # Making sure whatever message was passed is over


    def __update_preprocessed_tree_node(self):
        logger.info(f"It's a preprocessed flow and hence updating current node")
        self.tools['llm_agent'].update_current_node()
    
    def __convert_to_request_log(self, message, meta_info, model, component = "transcriber", direction = 'response'):
        log = dict()
        log['direction'] = direction
        log['data'] = message
        log['leg_id'] = meta_info['request_id'] if "request_id" in meta_info else "1234"
        log['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log['component'] = component
        log['sequence_id'] = meta_info['sequence_id']
        log['model'] = model
        if component == "transcriber":
            if 'is_final' in meta_info and meta_info['is_final']:
                log['is_final'] = True
        else:
            log['is_final'] = False #This is logged only for users to know final transcript from the transcriber
        asyncio.create_task(write_request_logs(log, self.run_id))

    ##############################################################
    # LLM task
    ##############################################################
    async def _handle_llm_output(self, next_step, text_chunk, should_bypass_synth, meta_info):

        logger.info("received text from LLM for output processing: {} which belongs to sequence id {}".format(text_chunk, meta_info['sequence_id']))
        if "request_id" not in meta_info:
            meta_info["request_id"] = str(uuid.uuid4())
        first_buffer_latency = time.time() - meta_info["llm_start_time"]
        #self.latency_dict[meta_info["request_id"]]["llm"] = first_buffer_latency
        meta_info["llm_first_buffer_generation_latency"] = first_buffer_latency
        if next_step == "synthesizer" and not should_bypass_synth:
            task = asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
            self.synthesizer_tasks.append(asyncio.ensure_future(task))
        elif self.tools["output"] is not None:
            logger.info("Synthesizer not the next step and hence simply returning back")
            #self.history = copy.deepcopy(self.interim_history)
            await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))

    async def _process_conversation_preprocessed_task(self, message, sequence, meta_info):
        if self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
            messages = copy.deepcopy(self.history)
            messages.append({'role': 'user', 'content': message['data']})
            logger.info(f"Starting LLM Agent {messages}")
            #Expose get current classification_response method from the agent class and use it for the response log
            self.__convert_to_request_log(message = format_messages(messages, use_system_prompt= True), meta_info= meta_info, component="llm", direction="request", model=self.task_config["tools_config"]["llm_agent"]["model"])
            async for next_state in self.tools['llm_agent'].generate(messages, label_flow=self.label_flow):
                if next_state == "<end_of_conversation>":
                    meta_info["end_of_conversation"] = True
                    self.buffered_output_queue.put_nowait(create_ws_data_packet("<end_of_conversation>", meta_info))
                    return
                
                logger.info(f"Text chunk {next_state['text']}")
                messages.append({'role': 'assistant', 'content': next_state['text']})
                self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(next_state['audio'], meta_info, is_md5_hash=True))))
            logger.info(f"Interim history after the LLM task {messages}")
            self.llm_response_generated = True
            self.interim_history = copy.deepcopy(messages)
            if self.callee_silent:
                logger.info("When we got utterance end, maybe LLM was still generating response. So, copying into history")
                self.history = copy.deepcopy(self.interim_history)


    async def _process_conversation_formulaic_task(self, message, sequence, meta_info):
        llm_response = ""
        logger.info("Agent flow is formulaic and hence moving smoothly")
        async for text_chunk in self.tools['llm_agent'].generate(self.history):
            if is_valid_md5(text_chunk):
                self.synthesizer_tasks.append(asyncio.create_task(
                    self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=True))))
            else:
                # TODO Make it more modular
                llm_response += " " +text_chunk
                next_step = self._get_next_step(sequence, "llm")
                if next_step == "synthesizer":
                    self.synthesizer_tasks.append(asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk, meta_info))))
                else:
                    logger.info(f"Sending output text {sequence}")
                    await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                    self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=False))))

    async def _process_conversation_task(self, message, sequence, meta_info):
        next_step = None
        
        logger.info("agent flow is not preprocessed")
        llm_response = ""

        start_time = time.time()
        should_bypass_synth = 'bypass_synth' in meta_info and meta_info['bypass_synth'] == True
        next_step = self._get_next_step(sequence, "llm")        
        meta_info['llm_start_time'] = time.time()
        route = None
        if self.route_layer is not None:
            route = self.route_layer(message['data']).name
            logger.info(f"Got route name {route}")
        
        if route is not None:
            logger.info(f"It was a route hit and we've got to respond from cache hence simply returning and the route is {route}")
            # Check if for the particular route if there's a vector store
            # If not send the response else get the response from the vector store
            logger.info(f"Vector caches {self.vector_caches}")
            if route in self.vector_caches:
                logger.info(f"Route {route} has a vector cache")
                relevant_utterance = self.vector_caches[route].get(message['data'])
                cache_response = self.route_responses_dict[route][relevant_utterance]
                self.__convert_to_request_log(message = message['data'], meta_info= meta_info, component="llm", direction="request", model=self.task_config["tools_config"]["llm_agent"]["model"])
                self.__convert_to_request_log(message = message['data'], meta_info= meta_info, component="llm", direction="response", model=self.task_config["tools_config"]["llm_agent"]["model"])
                messages = copy.deepcopy(self.history)
                messages += [{'role': 'user', 'content': message['data']},{'role': 'assistant', 'content': cache_response}]
                self.interim_history = copy.deepcopy(messages)
                self.llm_response_generated = True
                if self.callee_silent:
                    logger.info("##### When we got utterance end, maybe LLM was still generating response. So, copying into history")
                    self.history = copy.deepcopy(self.interim_history)

            else:
                logger.info(f"Route doesn't have a vector cache, and hence simply returning back a given response")
                cache_response = self.route_responses_dict[route]
            
            logger.info(f"Cached response {cache_response}")
            meta_info['cached'] = True
            meta_info["end_of_llm_stream"] = True
                
            await self._handle_llm_output(next_step, cache_response, should_bypass_synth, meta_info)
            self.llm_processed_request_ids.add(self.current_request_id)
        else:
            messages = copy.deepcopy(self.history)
            messages.append({'role': 'user', 'content': message['data']})
            ### TODO CHECK IF THIS IS EVEN REQUIRED
            self.__convert_to_request_log(message=format_messages(messages, use_system_prompt= True), meta_info= meta_info, component="llm", direction="request", model=self.task_config["tools_config"]["llm_agent"]["model"])
            
            async for llm_message in self.tools['llm_agent'].generate(messages, synthesize=True):
                text_chunk, end_of_llm_stream = llm_message
                llm_response += " " + text_chunk
                if self.stream:
                    if end_of_llm_stream:
                        meta_info["end_of_llm_stream"] = True
                    text_chunk = self.__process_stop_words(text_chunk, meta_info)
                    logger.info(f"##### O/P from LLM {text_chunk} {llm_response}")
                    await self._handle_llm_output(next_step, text_chunk, should_bypass_synth, meta_info)
                    
            if not self.stream:
                meta_info["end_of_llm_stream"] = True
                messages.append({"role": "assistant", "content": llm_response})
                self.history = copy.deepcopy(messages)
                await self._handle_llm_output(next_step, llm_response, should_bypass_synth, meta_info)
                self.__convert_to_request_log(message = llm_response, meta_info= meta_info, component="llm", direction="response", model=self.task_config["tools_config"]["llm_agent"]["model"])
            else:    
                if self.current_request_id in self.llm_rejected_request_ids:
                    logger.info("##### User spoke while LLM was generating response")
                else:
                    messages.append({"role": "assistant", "content": llm_response})
                    self.__convert_to_request_log(message=llm_response, meta_info= meta_info, component="llm", direction="response", model=self.task_config["tools_config"]["llm_agent"]["model"])
                    self.interim_history = copy.deepcopy(messages)
                    self.llm_response_generated = True
                    if self.callee_silent:
                        logger.info("##### When we got utterance end, maybe LLM was still generating response. So, copying into history")
                        self.history = copy.deepcopy(self.interim_history)
                    #self.__update_transcripts()

            # TODO : Write a better check for completion prompt 
            if self.use_llm_to_determine_hangup and not self.connected_through_dashboard:
                answer = await self.tools["llm_agent"].check_for_completion(self.history, self.check_for_completion_prompt)
                should_hangup = answer['answer'].lower() == "yes"
                prompt = [
                        {'role': 'system', 'content': self.check_for_completion_prompt},
                        {'role': 'user', 'content': format_messages(self.history, use_system_prompt= True)}]
                logger.info(f"##### Answer from the LLM {answer}")
                self.__convert_to_request_log(message=format_messages(prompt, use_system_prompt= True), meta_info= meta_info, component="llm", direction="request", model=self.task_config["tools_config"]["llm_agent"]["model"])
                self.__convert_to_request_log(message=answer, meta_info= meta_info, component="llm", direction="response", model= self.check_for_completion_llm)
                
                if should_hangup:
                    await self.__process_end_of_conversation()
                    return

            self.llm_processed_request_ids.add(self.current_request_id)
            llm_response = ""
    
    async def _listen_llm_input_queue(self):
        logger.info(
            f"Starting listening to LLM queue as either Connected to dashboard = {self.connected_through_dashboard} or  it's a textual chat agent {self.textual_chat_agent}")
        while True:
            try:
                ws_data_packet = await self.queues["llm"].get()
                logger.info(f"ws_data_packet {ws_data_packet}")
                meta_info = self.__get_updated_meta_info(ws_data_packet['meta_info'])
                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tools["output"].handle(bos_packet)
                # self.interim_history = self.history.copy()
                # self.history.append({'role': 'user', 'content': ws_data_packet['data']})
                await self._run_llm_task(
                    create_ws_data_packet(ws_data_packet['data'], meta_info))
                if self._is_preprocessed_flow():
                    self.__update_preprocessed_tree_node()
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tools["output"].handle(eos_packet)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Something went wrong with LLM queue {e}")
                break

    async def _run_llm_task(self, message):
        sequence, meta_info = self._extract_sequence_and_meta(message)
        logger.info(f"After adding {self.curr_sequence_id} into sequence id {self.sequence_ids} for message {message}")

        try:
            if self._is_extraction_task() or self._is_summarization_task():
                await self._process_followup_task(message)
            elif self._is_conversation_task():
                if self._is_preprocessed_flow():
                    if time.time() < self.consider_next_transcript_after:
                        logger.info("Not considering transcript as we're still in cool down period")
                        await asyncio.sleep(self.consider_next_transcript_after - time.time())
                    logger.info(f"Running preprocessedf task")
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


    #################################################################
    # Transcriber task
    #################################################################
    async def process_transcriber_request(self, meta_info):
        if not self.current_request_id or self.current_request_id != meta_info["request_id"]:
            self.previous_request_id, self.current_request_id = self.current_request_id, meta_info["request_id"]

        sequence = meta_info["sequence"]

        # check if previous request id is not in transmitted request id
        if self.previous_request_id is None:
            is_first_message = True
        elif self.previous_request_id not in self.llm_processed_request_ids:
            logger.info(f"Adding previous request id to LLM rejected request if")
            self.llm_rejected_request_ids.add(self.previous_request_id)
        else:
            skip_append_to_data = False
        return sequence

    async def _handle_transcriber_output(self, next_task, transcriber_message, meta_info):
        self.__convert_to_request_log(message=transcriber_message, meta_info= meta_info, model = "deepgram")
        if next_task == "llm":
            logger.info(f"Running llm Tasks")
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
        response_started = False
        num_words = 0
        try:
            while True:
                message = await self.transcriber_output_queue.get()
                logger.info(f"##### Message from the transcriber class {message}")
                if message["data"].strip() == "":
                    continue
                if message['data'] == "transcriber_connection_closed":
                    self.transcriber_duration += message['meta_info']["transcriber_duration"]
                    logger.info("transcriber connection closed")
                    break

                if self.stream:
                    self._set_call_details(message)
                    meta_info = message["meta_info"]
                    sequence = await self.process_transcriber_request(meta_info)
                    next_task = self._get_next_step(sequence, "transcriber")
                    num_words = 0
                    if message['data'] == "TRANSCRIBER_BEGIN":
                        response_started = False #This signifies if we've gotten the first bit of interim text for the given response or not
                        self.callee_silent = False
                        if self.nitro:
                            logger.info(f"Just a nitro thingy")
                            should_interrupt = meta_info.get("should_interrupt", True)
                            # if not should_interrupt and self.started_transmitting_audio:
                            #     # Ideally is we are transmitting, we want to wait for x seconds here to make sure if we interrupt or not 
                            #     # So, we send a clear message for sure but use a variable to make sure that we wait 
                            #     # then if we haven't received interruption signal, we simply continue
                            #     # If we have, we interrupt  
                            #     # Send a clear message
                            #     await self.tools["output"].handle_interruption()
                            #     self.backoff_until = (time.time() * 1000) + self.interruption_backoff_period
                            #     self.allow_extra_sleep = True
                            #     logger.info(f"###### Sending interrupt to clear and allowing extra sleep to wait for more messages as we are transmitting audio right now. {self.backoff_until}")
                        
                        else:
                            logger.info(f"###### Processing interruption from TRANSCRIBER_BEGIN for non nitro mode")
                            await self.process_interruption()
                    elif "speech_final" in meta_info and meta_info['speech_final'] and message['data'] != "":
                        logger.info(f"Starting the TRANSCRIBER_END TASK")
                        self.callee_speaking = False
                        self.callee_silent = True
                        if self.output_task is None:
                            logger.info(f"Output task was none and hence starting it")
                            self.output_task = asyncio.create_task(self.__process_output_loop())

                        if self._is_preprocessed_flow():
                            self.__update_preprocessed_tree_node()
                        
                        logger.info(f"INTERIM TRANSCRIPT WHEN EVERYTING IS OVER {self.interim_history}")
                        if self.llm_response_generated:
                            logger.info(f"LLM RESPONSE WAS GENERATED AND HENCE MOVING INTERIM HISTORY TO HISTORY")
                            self.history = copy.deepcopy(self.interim_history)
                        meta_info = message['meta_info']
                        transcriber_message = ""
                        self.let_remaining_audio_pass_through = True 

                        if self.nitro:
                            self.time_since_first_interim_result = -1
                            self.required_delay_before_speaking = max(self.minimum_wait_duration - self.incremental_delay, 0)
                            logger.info(f"#### Resetting time since first interim result and resetting required delay {self.required_delay_before_speaking}")
                        
                    else:
                        self.time_since_last_spoken_human_word = time.time()
                        logger.info(f'invoking next_task {next_task} with transcriber_message: {message["data"]}')
                        if transcriber_message.strip() == message['data'].strip():
                            logger.info(f"###### Transcriber message and message data are same and hence not changing anything else. Probably just an is_final thingy. {message}")
                            continue

                        elif len(message['data'].strip()) != 0:
                            #Currently simply cancel the next task
                            num_words += len(message['data'].split(" "))
                            if self.callee_speaking is False:
                                self.callee_speaking_start_time = time.time()
                                self.callee_speaking = True
                            # if self.started_transmitting_audio:
                            #     # Ideally is we are transmitting, we want to wait for x seconds here to make sure if we interrupt or not 
                            #     # So, we send a clear message for sure but use a variable to make sure that we wait 
                            #     # then if we haven't received interruption signal, we simply continue
                            #     # If we have, we interrupt  
                            #     # Send a clear message
                            #     await self.tools["output"].handle_interruption()
                            #     self.backoff_until = (time.time() * 1000) + self.interruption_backoff_period
                            #     self.allow_extra_sleep = True
                            #     logger.info(f"Sending interrupt to clear and allowing extra sleep to wait for more messages as we are transmitting audio right now. {self.backoff_until}")

                            
                            # This means we are generating response from an interim transcript 
                            # Hence we transmit quickly 
                            if not self.started_transmitting_audio:
                                logger.info("##### Haven't started transmitting audio and hence cleaning up downstream tasks")
                                await self.__cleanup_downstream_tasks()
                            
                            # If we've started transmitting audio this is probably an interruption, so calculate number of words
                            if self.nitro and self.started_transmitting_audio:
                                if num_words > self.number_of_words_for_interruption:
                                    #Process interruption only if number of words is higher than the threshold 
                                    logger.info(f"###### Number of words {num_words} is higher than the required number of words for interruption, hence, definitely interrupting")
                                    await self.__cleanup_downstream_tasks()
                                else:
                                    logger.info(f"Not starting a cleanup because {num_words} number of words are lesser {self.number_of_words_for_interruption} and hence continuing,")
                                    continue
                                    
                            self.last_response_time = time.time()
                            transcriber_message = message['data']
                            
                            # Use last spoken timestamp to give out endpointing in nitro
                            logger.info(f"###### last spoken timestamp before changing {self.last_spoken_timestamp}")
                            self.last_spoken_timestamp = time.time() * 1000

                            if not response_started:
                                response_started = True
                            elif self.nitro:
                                self.let_remaining_audio_pass_through = False
                                logger.info(f"Increase the incremental delay time")
                                self.required_delay_before_speaking += self.incremental_delay
                                if self.time_since_first_interim_result == -1:
                                    self.time_since_first_interim_result = time.time() * 1000
                                    logger.info(f"###### Updating Time since first interim result {self.time_since_first_interim_result}")
                                #In this case user has already started speaking
                                # Hence check the previous message if it's user or assistant
                                # If it's user, simply change user's message
                                # If it's assistant remover assistant message and append user
                                self.callee_silent = False
                            self.llm_response_generated = False
                            logger.info("###### Current transcript: {} Predicting next few tokens and changing last spoken timestampt to {}".format(transcriber_message, self.last_spoken_timestamp))
                            meta_info = self.__get_updated_meta_info(meta_info)
                            await self._handle_transcriber_output(next_task, transcriber_message, meta_info)

                        else:
                            logger.info(f"Got a null message")
                else:
                    logger.info(f"Processing http transcription for message {message}")
                    await self.__process_http_transcription(message)
        
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in transcriber {e}")
    
    
    async def __process_http_transcription(self, message):
        meta_info = self.__get_updated_meta_info(message["meta_info"])
        include_latency = meta_info.get("include_latency", False)
        if include_latency:
            self.latency_dict[meta_info['request_id']]["transcriber"] = {"total_latency":  meta_info["transcriber_latency"], "audio_duration": meta_info["audio_duration"], "last_vocal_frame_timestamp": meta_info["last_vocal_frame_timestamp"] }

        sequence = message["meta_info"]["sequence"]
        next_task = self._get_next_step(sequence, "transcriber")
        self.transcriber_duration += message["meta_info"]["transcriber_duration"] if "transcriber_duration" in message["meta_info"] else 0
        #self.history.append({'role': 'user', 'content': message['data']})
        if self._is_preprocessed_flow():
            self.__update_preprocessed_tree_node()

        await self._handle_transcriber_output(next_task, message['data'], meta_info)


    #################################################################
    # Synthesizer task
    #################################################################
    def __enqueue_chunk(self, chunk, i, number_of_chunks, meta_info):
        logger.info(f"Meta_info of chunk {meta_info} {i} {number_of_chunks}")
        if i == 0 and "is_first_chunk" in meta_info and meta_info["is_first_chunk"]:
            copied_meta_info = copy.deepcopy(meta_info)
            logger.info(f"##### Sending first chunk")
            copied_meta_info["is_first_chunk_of_entire_response"] = True
            self.buffered_output_queue.put_nowait(create_ws_data_packet(chunk, copied_meta_info))
        elif i == number_of_chunks and "end_of_synthesizer_stream" in meta_info and meta_info['end_of_synthesizer_stream']:
            logger.info(f"##### Truly a final chunk")
            copied_meta_info = meta_info.copy()
            copied_meta_info["is_final_chunk_of_entire_response"] = True
            self.buffered_output_queue.put_nowait(create_ws_data_packet(chunk, copied_meta_info))
        else:
            self.buffered_output_queue.put_nowait(create_ws_data_packet(chunk, meta_info))

    async def __listen_synthesizer(self):
        try:
            if self.stream and self.synthesizer_provider != "polly" and not self.is_an_ivr_call: 
                logger.info("Opening websocket connection to synthesizer")
                await self.tools["synthesizer"].open_connection()
            while True:
                logger.info("Listening to synthesizer")
                async for message in self.tools["synthesizer"].generate():
                    meta_info = message["meta_info"]
                    if not self.conversation_ended and message["meta_info"]["sequence_id"] in self.sequence_ids:
                        logger.info(f"{message['meta_info']['sequence_id'] } is in sequence ids  {self.sequence_ids} and hence removing the sequence ids ")
                        if self.stream:   
                            if self.synthesizer_provider == "polly":
                                if message['meta_info']['is_first_chunk']:
                                    first_chunk_generation_timestamp = time.time()
                                    meta_info["synthesizer_first_chunk_latency"] = first_chunk_generation_timestamp - message['meta_info']['synthesizer_start_time']
                                logger.info(f"Simply Storing in buffered output queue for now")

                                if self.yield_chunks:
                                    logger.info(f"Yielding chunks")
                                    number_of_chunks = (len(message['data'])//self.output_chunk_size)
                                    i = 0
                                    for chunk in yield_chunks_from_memory(message['data'], chunk_size=self.output_chunk_size):
                                        self.__enqueue_chunk(chunk, i, number_of_chunks, meta_info)
                                        i +=1
                                else:
                                    self.buffered_output_queue.put_nowait(message)
                                
                            else:
                                if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_INPUT_TELEPHONY_HANDLERS.keys() and not self.connected_through_dashboard and self.synthesizer_provider == "elevenlabs":
                                    if meta_info.get('format', '') != 'mulaw':
                                        message['data'] = wav_bytes_to_pcm(message['data'])
                                
                                if "is_first_chunk" in message['meta_info'] and message['meta_info']['is_first_chunk']:
                                    first_chunk_generation_timestamp = time.time()
                                    meta_info["synthesizer_first_chunk_latency"] = first_chunk_generation_timestamp - message['meta_info']['synthesizer_start_time']
                                    #self.latency_dict[message['meta_info']["request_id"]]['synthesizer'] = {"first_chunk_generation_latency": first_chunk_generation_timestamp - message['meta_info']['synthesizer_start_time'], "first_chunk_generation_timestamp": first_chunk_generation_timestamp}
                                
                                if self.yield_chunks:
                                    number_of_chunks = (len(message['data'])/self.output_chunk_size)
                                    i = 0
                                    for chunk in yield_chunks_from_memory(message['data'], chunk_size=self.output_chunk_size):
                                        i+=1
                                        self.__enqueue_chunk(chunk, i, number_of_chunks, meta_info)
                                else:
                                    self.buffered_output_queue.put_nowait(message)
                            
                        else:
                            logger.info("Stream is not enabled and hence sending entire audio")
                            first_chunk_generation_timestamp = time.time()
                            self.latency_dict[message['meta_info']["request_id"]]['synthesizer'] = {"first_chunk_generation_latency": first_chunk_generation_timestamp - message['meta_info']['synthesizer_start_time'], "first_chunk_generation_timestamp": first_chunk_generation_timestamp}
                            #self.history = copy.deepcopy(self.interim_history)
                            logger.info(f"Changing history")
                            await self.tools["output"].handle(message)
                    else:
                        logger.info(f"{message['meta_info']['sequence_id']} is not in sequence ids  {self.sequence_ids} and hence not sending to output")                
                    logger.info(f"Sleeping for 100 ms")
                    await asyncio.sleep(0.3) #Sleeping for 100ms after receiving every chunk so other tasks can execute

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer {e}")

    async def __send_preprocessed_audio(self, meta_info, text):
        #TODO: Either load IVR audio into memory before call or user s3 iter_cunks
        # This will help with interruption in IVR
        if self.connected_through_dashboard or self.task_config['tools_config']['output'] == "default":
            audio_chunk = await get_raw_audio_bytes(text, self.assistant_name,
                                                            self.task_config["tools_config"]["output"][
                                                                "format"], local=self.is_local,
                                                            assistant_id=self.assistant_id)
            logger.info("Sending preprocessed audio")
            await self.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))
        else:
            audio_chunk = await get_raw_audio_bytes( text, self.assistant_name,
                                                            'pcm', local=self.is_local,
                                                            assistant_id=self.assistant_id)
            
            if not self.buffered_output_queue.empty():
                logger.info(f"Output queue was not empty and hence emptying it")
                self.buffered_output_queue = asyncio.Queue()

            if self.yield_chunks:
                for chunk in yield_chunks_from_memory(audio_chunk, chunk_size=self.output_chunk_size):
                    logger.debug("Sending chunk to output queue")
                    message = create_ws_data_packet(chunk, meta_info)
                    self.buffered_output_queue.put_nowait(message)
            else:
                message = create_ws_data_packet(audio_chunk, meta_info)
                logger.info(f"Yield in chunks is false and hence sending a full")
                self.buffered_output_queue.put_nowait(message)

    async def _synthesize(self, message):
        meta_info = message["meta_info"]
        text = message["data"]
        meta_info["type"] = "audio"
        meta_info["synthesizer_start_time"] = time.time()
        try:
            if not self.conversation_ended and message["meta_info"]["sequence_id"] in self.sequence_ids:
                if meta_info["is_md5_hash"]:
                    logger.info('sending preprocessed audio response to {}'.format(self.task_config["tools_config"]["output"]["provider"]))
                    await self.__send_preprocessed_audio(meta_info, text)
                    
                elif self.synthesizer_provider in SUPPORTED_SYNTHESIZER_MODELS.keys():
                    # self.sequence_ids.add(meta_info["sequence_id"])
                    # logger.info(f"After adding into sequence id {self.sequence_ids}")
                    self.__convert_to_request_log(message = text, meta_info= meta_info, component="synthesizer", direction="request", model = self.synthesizer_provider)
                    logger.info('##### sending text to {} for generation: {} '.format(self.synthesizer_provider, text))
                    if 'cached' in message['meta_info'] and meta_info['cached'] == True:
                        logger.info(f"Cached response and hence sending preprocessed text")
                        await self.__send_preprocessed_audio(meta_info, get_md5_hash(text))
                    else:
                        self.synthesizer_characters += len(text)
                        await self.tools["synthesizer"].push(message)
                else:
                    logger.info("other synthesizer models not supported yet")
            else:
                logger.info(f"{message['meta_info']['sequence_id']} is not in sequence ids  {self.sequence_ids} and hence not synthesizing this")                

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer: {e}")

    ############################################################
    # Output handling
    ############################################################
    async def __handle_initial_silence(self):
        logger.info(f"Checking for initial silence")
        await asyncio.sleep(5)
        if self.callee_silent and len(self.history) == 1 and len(self.interim_history) == 1:
            logger.info(f"Calee was silent and hence speaking Hello on callee's behalf")
            meta_info = self.__get_updated_meta_info()
            sequence = meta_info["sequence"]
            next_task = self._get_next_step(sequence, "transcriber")
            await self._handle_transcriber_output(next_task, "Hello", meta_info)
            self.time_since_first_interim_result = (time.time() * 1000) - 1000
            
    #Currently this loop only closes in case of interruption 
    # but it shouldn't be the case. 
    async def __process_output_loop(self):
        prev_message = None
        current_message = None
        try:
            while True:
                # Allow extra sleep allows us to have real time impact in when uer starts speaking
                # if self.nitro and self.allow_extra_sleep and time.time() *1000 < self.backoff_until:
                #     logger.info(f"##### sleeping for extra backoff period to see if user will start speaking something new or not after {self.interruption_backoff_period/1000}")
                #     await asyncio.sleep(self.interruption_backoff_period/1000)
                #     self.allow_extra_sleep = False
                #     prev_message = current_message

                if self.nitro and not self.let_remaining_audio_pass_through:
                    time_since_first_interim_result = (time.time() *1000)- self.time_since_first_interim_result if self.time_since_first_interim_result != -1 else -1
                    if  time_since_first_interim_result != -1 and time_since_first_interim_result < self.required_delay_before_speaking:
                        logger.info(f"##### It's been {time_since_first_interim_result} ms since first  interim result and required time to wait for it is {self.required_delay_before_speaking}. Hence sleeping for 100ms. self.time_since_first_interim_result {self.time_since_first_interim_result}")
                        await asyncio.sleep(0.1) #sleep for 100ms and continue 
                        continue
                    else:
                        logger.info(f"First interim result hasn't been gotten yet and hence sleeping ")
                        await asyncio.sleep(0.1)

                    logger.info(f"##### Got to wait {self.required_delay_before_speaking} ms before speaking and alreasy waited {time_since_first_interim_result} since the first interim result")
                else:
                    logger.info(f"Started transmitting at {time.time()}")
                if prev_message is None:
                    message = await self.buffered_output_queue.get()   
                    current_message = message 
                else:
                    logger.info(f'prev message is not none and hence getting prev message')
                    message = prev_message
                    prev_message = None
                logger.info("##### Start response is True and hence starting to speak {} Current sequence ids".format(message['meta_info'], self.sequence_ids))
                if "end_of_conversation" in message['meta_info']:
                    await self.__process_end_of_conversation()
                
                if 'sequence_id' in message['meta_info'] and message["meta_info"]["sequence_id"] in self.sequence_ids:
                    await self.tools["output"].handle(message)                    
                    duration = calculate_audio_duration(len(message["data"]), self.sampling_rate)
                    logger.info(f"Duration of the byte {duration}")
                    self.conversation_recording['output'].append({'data': message['data'], "start_time": time.time(), "duration": duration})
                else:
                    logger.info(f'{message["meta_info"]["sequence_id"]} is not in {self.sequence_ids} and hence not speaking')
                    continue
                
                if "is_final_chunk_of_entire_response" in message['meta_info'] and message['meta_info']['is_final_chunk_of_entire_response']:
                    self.started_transmitting_audio = False
                    logger.info("##### End of synthesizer stream and ")                    

                if "is_first_chunk_of_entire_response" in message['meta_info'] and message['meta_info']['is_first_chunk_of_entire_response']:
                    logger.info(f"First chunk stuff")
                    self.started_transmitting_audio = True
                    meta_info = message['meta_info']
                    self.consider_next_transcript_after = time.time() + self.duration_to_prevent_accidental_interruption
                    utterance_end = meta_info.get("utterance_end", None)
                    overall_first_byte_latency = time.time() - message['meta_info']['utterance_end'] if utterance_end is not None else 0
                    transcriber_latency = message["meta_info"]["transcriber_latency"] if utterance_end is not None else 0
                    first_llm_buffer_latency = message["meta_info"]["llm_first_buffer_generation_latency"] if utterance_end is not None else 0
                    synthesizer_first_chunk_latency = message["meta_info"]["synthesizer_first_chunk_latency"] if utterance_end is not None else 0

                    if utterance_end is None:
                        logger.info(f"First chunk is none")

                    latency_metrics = {
                        "transcriber": {
                            "utterance_end": utterance_end,
                            "latency": transcriber_latency
                            },
                        "llm": {
                            "first_llm_buffer_latency" : first_llm_buffer_latency
                            },
                        "synthesizer": {
                            "synthesizer_first_chunk_latency": synthesizer_first_chunk_latency
                            },
                        "overall_first_byte_latency": overall_first_byte_latency,
                        
                        }

                    if message['meta_info']["request_id"] not in self.latency_dict:
                        self.latency_dict[message['meta_info']["request_id"]] = latency_metrics
                        logger.info("LATENCY METRICS FOR {} are {}".format(message['meta_info']["request_id"], latency_metrics))
                    
                    await asyncio.sleep(duration + 0.1) 
                else:
                    # Sleep until this particular audio frame is spoken only if the duration for the frame is atleast 500ms
                    if duration > 0:
                        logger.info(f"##### Sleeping for {duration} to maintain quueue on our side {self.sampling_rate}")
                        await asyncio.sleep(duration - 0.010) #10 milliseconds less
                        

                    
                self.last_transmitted_timesatamp = time.time()
                logger.info(f"##### Updating Last transmitted timestamp to {self.last_transmitted_timesatamp}")
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f'Error in processing message output')

    async def __check_for_completion(self):
        while True:
            await asyncio.sleep(2)

            if self.last_transmitted_timesatamp == 0:
                logger.info(f"Last transmitted timestamp is simply 0 and hence continuing")
                continue

            time_since_last_spoken_AI_word = (time.time() - self.last_transmitted_timesatamp) 
            if time_since_last_spoken_AI_word > self.hang_conversation_after and self.time_since_last_spoken_human_word < self.last_transmitted_timesatamp:
                logger.info(f"{time_since_last_spoken_AI_word} seconds since last spoken time stamp and hence cutting the phone call and last transmitted timestampt ws {self.last_transmitted_timesatamp} and time since last spoken human word {self.time_since_last_spoken_human_word}")
                await self.__process_end_of_conversation()
                break
            else:
                logger.info(f"Only {time_since_last_spoken_AI_word} seconds since last spoken time stamp and hence cutting the phone call and hence not cutting the phone call")
    
    async def __check_for_backchanneling(self):
        while True:
            if self.callee_speaking and time.time() - self.callee_speaking_start_time > self.backchanneling_start_delay:
                filename = random.choice(self.filenames)
                logger.info(f"Should send a random backchanneling words and sending them {filename}")
                audio = await get_raw_audio_bytes(f"{self.backchanneling_audios}/{filename}", local= True, is_location=True)
                if not self.connected_through_dashboard and self.task_config['tools_config']['output'] != "default":
                    audio = resample(audio, target_sample_rate= 8000, format="wav")
                await self.tools["output"].handle(create_ws_data_packet(audio, self.__get_updated_meta_info())) 
            else:
                logger.info(f"Callee isn't speaking and hence not sending or {time.time() - self.callee_speaking_start_time} is not greater than {self.backchanneling_start_delay}") 
            await asyncio.sleep(self.backchanneling_message_gap) 
    async def run(self):
        try:
            if self.task_id == 0:
                # Create transcriber and synthesizer tasks
                logger.info("starting task_id {}".format(self.task_id))
                tasks = [asyncio.create_task(self.tools['input'].handle())]
                if not self.connected_through_dashboard:
                    self.background_check_task = asyncio.create_task(self.__handle_initial_silence())
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
                if self._is_conversation_task():
                    self.output_task = asyncio.create_task(self.__process_output_loop())
                    if not self.use_llm_to_determine_hangup and not self.connected_through_dashboard :
                        self.hangup_task = asyncio.create_task(self.__check_for_completion())
                    if not self.connected_through_dashboard and self.should_backchannel:
                        self.backchanneling_task = asyncio.create_task(self.__check_for_backchanneling())
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
                    if self.task_config["task_type"] == "webhook":
                        await self._process_followup_task()
                    else:
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
            if self.use_llm_to_determine_hangup is False and self._is_conversation_task() and not self.connected_through_dashboard:
                self.hangup_task.cancel()
            
            if self._is_conversation_task():
                self.output_task.cancel()
            
            if self._is_conversation_task() and self.backchanneling_task is not None:
                self.backchanneling_task.cancel()
            
            if self.task_id == 0:
                output = {"messages": self.history, "conversation_time": time.time() - self.start_time,
                          "label_flow": self.label_flow, "call_sid": self.call_sid, "stream_sid": self.stream_sid,
                          "transcriber_duration": self.transcriber_duration,
                          "synthesizer_characters": self.tools['synthesizer'].get_synthesized_characters(), "ended_by_assistant": self.ended_by_assistant,
                          "latency_dict": self.latency_dict}

                if self.should_record:
                    output['recording_url'] = await save_audio_file_to_s3(self.conversation_recording, self.sampling_rate, self.assistant_id, self.run_id)

            else:
                output = self.input_parameters
                if self.task_config["task_type"] == "extraction":
                    output = { "extracted_data" : self.extracted_data, "task_type": "extraction"}
                elif self.task_config["task_type"] == "summarization":
                    logger.info(f"self.summarized_data {self.summarized_data}")
                    output = {"summary" : self.summarized_data, "task_type": "summarization"}
                elif self.task_config["task_type"] == "webhook":
                    output = {"status": self.webhook_response, "task_type": "webhook"}
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