import asyncio
import traceback
from bolna.agent_types import StreamingContextualAgent, GraphBasedConversationAgent, ExtractionContextualAgent, SummarizationContextualAgent
import time
import json
from bolna.helpers.logger_config import configure_logger
from bolna.models import *
from bolna.helpers.utils import create_ws_data_packet, is_valid_md5, get_raw_audio_bytes_from_base64, \
    get_required_input_types, format_messages, get_prompt_responses, update_prompt_with_context

logger = configure_logger(__name__, True)


class TaskManager:
    def __init__(self, assistant_name, task_id, task, ws, input_parameters=None, context_data=None, user_id=None,
                 assistant_id=None, run_id=None, connected_through_dashboard = False):
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

        self.pipelines = task['toolchain']['pipelines']
        self.textual_chat_agent = True if (task['toolchain']['pipelines'][0] == "llm" and task["tools_config"]["llm_agent"]["agent_task"] == "conversation") else False

        self.start_time = time.time()

        # Assistant persistance stuff
        self.user_id = user_id
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
                if connected_through_dashboard:
                    input_handler_class = SUPPORTED_INPUT_HANDLERS.get("default") #If connected through dashboard get basic dashboard class
                else:
                    input_handler_class = SUPPORTED_INPUT_HANDLERS.get(self.task_config["tools_config"]["input"]["provider"])
                self.tools["input"] = input_handler_class(self.queues, self.websocket, get_required_input_types(task), self.mark_set, self.connected_through_dashboard)
            else:
                raise "Other input handlers not supported yet"

        if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_HANDLERS.keys():
            if self.task_config["tools_config"]["output"]["provider"] == 'default':
                self.tools["output"] = DefaultOutputHandler(self.websocket)
            elif self.task_config["tools_config"]["output"]["provider"] == "database":
                self.tools["output"] = DatabaseOutputHandler(self.user_id, self.assistant_id, self.run_id)
            elif self.task_config["tools_config"]["output"]["provider"] == "twilio":
                self.tools["output"] = TwilioOutputHandler(self.websocket, self.mark_set)
        else:
            raise "Other input handlers not supported yet"

        # Current conversation state
        self.current_request_id = None
        self.previous_request_id = None
        self.llm_rejected_request_ids = set()
        self.llm_processed_request_ids = set()

        # Agent stuff
        self.history = []
        self.label_flow = []

        # Setup IO SERVICE, TRANSCRIBER, LLM, SYNTHESIZER
        self.llm_task = None
        self.synthesizer_tasks = []

        # state of conversation
        self.was_long_pause = False

        # Call conversations
        self.call_sid = None
        self.stream_sid = None

        # chharacters
        self.transcription_characters = 0
        self.synthesizer_characters = 0

        self.extracted_data = None
        self.summarized_data = None

        self.stream = not connected_through_dashboard
        self.is_local = False

    async def load_prompt(self, assistant_name, task_id, is_local):
        self.is_local = is_local
        prompt_responses = await get_prompt_responses(assistant_name, assistant_id=self.assistant_id, user_id=self.user_id, local=self.is_local)
        self.prompts = prompt_responses["task_{}".format(task_id + 1)]

        if "system_prompt" in self.prompts:
            # This isn't a graph based agent
            enriched_prompt = update_prompt_with_context(self.prompts["system_prompt"], self.context_data) if self.context_data is not None else self.prompts["system_prompt"]
            self.system_prompt = {
                'role': "system",
                'content': enriched_prompt
            }
        else:
            self.system_prompt = {
                'role': "system",
                'content': ""
            }

        self.history = [self.system_prompt]

        llm_config = {"streaming_model": self.task_config["tools_config"]["llm_agent"]["streaming_model"]}

        if self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] == "preprocessed":
            llm_config["classification_model"] = self.task_config["tools_config"]["llm_agent"]["classification_model"]

        # setting transcriber
        if self.task_config["tools_config"]["transcriber"] is not None:
            provider = "playground" if self.connected_through_dashboard else self.task_config["tools_config"]["input"]["provider"]
            self.task_config["tools_config"]["transcriber"]["input_queue"] = self.audio_queue
            if self.task_config["tools_config"]["transcriber"]["model"] in SUPPORTED_TRANSCRIBER_MODELS.keys():
                if self.connected_through_dashboard:
                    self.task_config["tools_config"]["transcriber"]["stream"] = False
                transcriber_class = SUPPORTED_TRANSCRIBER_MODELS.get(
                    self.task_config["tools_config"]["transcriber"]["model"])
                self.tools["transcriber"] = transcriber_class(provider, **self.task_config["tools_config"]["transcriber"])

        # setting synthesizer
        if self.task_config["tools_config"]["synthesizer"] is not None:
            synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(
                self.task_config["tools_config"]["synthesizer"]["model"])
            self.tools["synthesizer"] = synthesizer_class(**self.task_config["tools_config"]["synthesizer"])

            llm_config["max_tokens"] = self.task_config["tools_config"]["synthesizer"].get('max_tokens')
            llm_config["buffer_size"] = self.task_config["tools_config"]["synthesizer"].get('buffer_size')

        # setting llm
        if self.task_config["tools_config"]["llm_agent"]["family"] in SUPPORTED_LLM_MODELS.keys():
            llm_class = SUPPORTED_LLM_MODELS.get(self.task_config["tools_config"]["llm_agent"]["family"])
            llm = llm_class(**llm_config)
        else:
            raise Exception(f'LLM {self.task_config["tools_config"]["llm_agent"]["family"]} not supported')

        if self.task_config["task_type"] == "conversation":
            if self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] == "streaming":
                self.tools["llm_agent"] = StreamingContextualAgent(llm)
            elif self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] in ("preprocessed", "formulaic"):
                preprocessed = self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] == "preprocessed"
                # TODO START WITH LOOKING INTO PROMPTS
                self.tools["llm_agent"] = GraphBasedConversationAgent(llm, context_data=self.context_data,
                                                                      prompts=self.prompts,
                                                                      preprocessed=preprocessed)
        elif self.task_config["task_type"] == "extraction":
            logger.info("Setting up extraction agent")
            self.tools["llm_agent"] = ExtractionContextualAgent(llm, prompt=self.system_prompt)
            self.extracted_data = None
        elif self.task_config["task_type"] == "summarization":
            logger.info("Setting up summarization agent")
            self.tools["llm_agent"] = SummarizationContextualAgent(llm, prompt=self.system_prompt)
            self.summarized_data = None

    ########################
    # LLM task
    ########################

    async def _handle_llm_output(self, next_step, text_chunk, should_bypass_synth, meta_info):
        if next_step == "synthesizer" and not should_bypass_synth:
            task = asyncio.gather(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
            self.synthesizer_tasks.append(asyncio.ensure_future(task))
        else:
            logger.info(f"Sending output text {text_chunk}")
            await self.tools["output"].handle(create_ws_data_packet(text_chunk,meta_info))

    def _get_next_step(self, sequence, origin):
        try:
            return next((self.pipelines[sequence][i + 1] for i in range(len(self.pipelines[sequence]) - 1) if
                         self.pipelines[sequence][i] == origin), "output")
        except Exception as e:
            logger.error(f"Error getting next step: {e}")

    def _set_call_details(self, message):
        if self.call_sid is not None and self.stream_sid is not None and "call_sid" not in message[
            'meta_info'] and "stream_sid" not in message['meta_info']:
            return

        if "call_sid" in message['meta_info']:
            self.call_sid = message['meta_info']["call_sid"]
        if "stream_sid" in message:
            self.stream_sid = message['meta_info']["stream_sid"]

    async def _process_followup_task(self, message, sequence, meta_info):
        message = format_messages(self.input_parameters["messages"]) #Remove the initial system prompt
        self.history.append({
            'role': 'user',
            'content': message
        })

        json_data = await self.tools["llm_agent"].generate(self.history)

        #TODO validation if the required data is correct
        if self.task_config["tools_config"]["output"]["provider"] == "database":
            self.extracted_data = json.loads(json_data)
            self.input_parameters = {**self.input_parameters, **self.extracted_data}
            logger.info(f"Saving data in DB {json_data}")
            await self.tools["output"].handle(self.input_parameters)
        else:
            await self.tools["output"].handle(self.input_parameters)

    async def _process_conversation_preprocessed_task(self, message, sequence, meta_info):
        if self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
            llm_response = ""
            self.history.append({
                'role': 'user',
                'content': message['data']
            })
            start_time = time.time()
            
            async for text_chunk in self.tools['llm_agent'].generate(self.history, stream=True, synthesize=True, label_flow = self.label_flow):
                if text_chunk == "<end_of_conversation>":
                    logger.info("Got end of conversation. I'm stopping now")
                    self.conversation_ended = True
                    await self.tools["input"].stop_handler()
                    logger.info("Stopped input handler")
                    if "transcriber" in self.tools and not self.connected_through_dashboard:
                        logger.info("Stopping transcriber")
                        await self.tools["transcriber"].toggle_connection()
                        await asyncio.sleep(5) #Making sure whatever message was passed is over
                    return
                logger.info(f"Text chunk {text_chunk}")
                if is_valid_md5(text_chunk):
                    self.synthesizer_tasks.append(asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk,meta_info, is_md5_hash = True))))
                else:
                    self.synthesizer_tasks.append(asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk,meta_info, is_md5_hash = False))))

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
                self.synthesizer_tasks.append(asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk,meta_info, is_md5_hash = True))))
            else:
                #TODO Make it more modular
                llm_response += text_chunk
                next_step = self._get_next_step(sequence, "llm")
                if next_step == "synthesizer":
                    task = asyncio.gather(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
                    self.synthesizer_tasks.append(asyncio.ensure_future(task))
                else:
                    logger.info(f"Sending output text {sequence}")
                    await self.tools["output"].handle(create_ws_data_packet(text_chunk,meta_info))
                    self.synthesizer_tasks.append(asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk,meta_info, is_md5_hash = False))))

    async def _process_conversation_task(self, message, sequence, meta_info):
        next_step = None
        logger.info("Agent flow is not preprocesessed and hence moving smoothly")
        llm_response = ""
        self.history.append({
            'role': 'user',
            'content': message['data']
        })
        start_time = time.time()
        should_bypass_synth = 'bypass_synth' in meta_info and meta_info['bypass_synth'] == True
        
        async for text_chunk in self.tools['llm_agent'].generate(self.history, synthesize=True):
            logger.info(f"###### time to get the first chunk {time.time() - start_time} {text_chunk}")
            llm_response += text_chunk
            next_step = self._get_next_step(sequence, "llm")
            if self.stream:
                await self._handle_llm_output(next_step, text_chunk, should_bypass_synth, meta_info)

        if not self.stream:
            await self._handle_llm_output(next_step, llm_response, should_bypass_synth, meta_info)

        if self.current_request_id in self.llm_rejected_request_ids:
            logger.info("User spoke while LLM was generating response")
        else:
            logger.info(f"Since we do not have to discard this response adding it {self.current_request_id}")
            self.history.append({"role": "assistant", "content": llm_response})
            
            answer = await self.tools["llm_agent"].check_for_completion(self.history)
            if answer:
                logger.info("Got end of conversation. I'm stopping now")
                self.conversation_ended = True
                await self.tools["input"].stop_handler()
                logger.info("Stopped input handler")
                if "transcriber" in self.tools and not  self.connected_through_dashboard:
                    logger.info("Stopping transcriber")
                    await self.tools["transcriber"].toggle_connection()
                    await asyncio.sleep(5) #Making sure whatever message was passed is over
                return

            self.llm_processed_request_ids.add(self.current_request_id)
            llm_response = ""
        
        logger.info(f"time to get all the chunks {time.time() - start_time}")

    
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
        logger.info(f"Starting listening to LLM queue as either Connected to dashboard = {self.connected_through_dashboard} or  it's a textual chat agent {self.textual_chat_agent}")
        while True:
            try:
                ws_data_packet = await self.queues["llm"].get()
                logger.info(f"ws_data_packet {ws_data_packet}")
                bos_packet = create_ws_data_packet("<beginning_of_stream>", ws_data_packet['meta_info'])
                await self.tools["output"].handle(bos_packet)
                await self._run_llm_task(ws_data_packet) #In case s3 is down and it's an audio processing job, this might produce blank message on the frontend of playground.
                eos_packet = create_ws_data_packet("<end_of_stream>",ws_data_packet['meta_info'])
                await self.tools["output"].handle(eos_packet)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Something went wrong with LLM queue {e}")
                break

    async def _run_llm_task(self, message):
        sequence, meta_info = self._extract_sequence_and_meta(message)
        
        logger.info("Running llm based agent")

        try:
            if self._is_extraction_task()or self._is_summarization_task():
                await self._process_followup_task(message, sequence, meta_info)
            elif self._is_conversation_task():
                if self._is_preprocessed_flow():
                    await self._process_conversation_preprocessed_task(message, sequence, meta_info)

                elif self._is_formulaic_flow(): 
                    await self._process_conversation_formulaic_task(message, sequence, meta_info)
                else:
                    await self._process_conversation_task(message, sequence, meta_info)
            else:
                logger.error("Unknown task type")
            self.llm_task = None
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Something went wrong in llm: {e}")
    

    async def process_transcriber_request(self, meta_info):
        if not self.current_request_id or self.current_request_id != meta_info[
            "request_id"]:
            self.previous_request_id, self.current_request_id = self.current_request_id, meta_info[
                "request_id"]

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
        await self.tools["output"].handle_interruption()
        if self.llm_task is not None:
            self.llm_task.cancel()
            self.llm_task = None
            self.was_long_pause = True

        if len(self.synthesizer_tasks) > 0:
            for synth_task in self.synthesizer_tasks:
                synth_task.cancel()
            self.synthesizer_tasks = []

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
            logger.info(f"Need to seperate out output task")

    async def _listen_transcriber(self):
        transcriber_message = ""
        start_time = None
        try:
            if self.stream:
                async for message in self.tools["transcriber"].transcribe():

                    if message['data'] == "transcriber_connection_closed":
                        logger.info("transcriber connection closed")
                        return

                    self._set_call_details(message)
                    meta_info = message["meta_info"]
                    self.previous_request_id, self.current_request_id = self.current_request_id, message["meta_info"][
                        "request_id"]
                    sequence = message["meta_info"]["sequence"]

                    # check if previous request id is not in transmitted request id
                    if self.previous_request_id is None:
                        logger.info("Previous request id is none and hence this must be the first message")
                    elif self.previous_request_id not in self.llm_processed_request_ids:
                        logger.info(
                            f"request id {meta_info['previous_request_id']} not in self.llm_processed_request_ids. Hence it must be transmitting")
                        self.llm_rejected_request_ids.add(self.previous_request_id)
                        logger.info(
                            f"processessed request ids {self.llm_processed_request_ids} rejected request ids {self.llm_rejected_request_ids}")
                    else:
                        logger.info(f"No need to append data")

                    if message['data'] == "TRANSCRIBER_BEGIN":
                        logger.info("Starting transcriber stream")
                        start_time = time.time()
                        await self.tools["output"].handle_interruption()
                        if self.llm_task is not None:
                            logger.info("Cancelling LLM Task as it's on")
                            self.llm_task.cancel()
                            self.llm_task = None
                            self.was_long_pause = True

                        if len(self.synthesizer_tasks) > 0:
                            logger.info("Cancelling Synthesizer tasks")
                            for synth_task in self.synthesizer_tasks:
                                synth_task.cancel()
                            self.synthesizer_tasks = []
                        continue
                    elif message['data'] == "TRANSCRIBER_END":
                        self.transcription_characters += len(transcriber_message)
                        logger.info("END and gerring the next step")
                        next_task = self._get_next_step(sequence, "transcriber")
                        logger.info(f'got the next task {next_task}')
                        if self.was_long_pause:
                            logger.info(
                                f"Seems like there was a long, long pause {self.history[-1]['content']} , {transcriber_message}")
                            message = self.history[-1]['content'] + " " + transcriber_message
                            self.history = self.history[:-1]
                            self.was_long_pause = False
                        await self._handle_transcriber_output(next_task, transcriber_message, meta_info)
                        transcriber_message = ""
                        continue
                    else:
                        logger.info("data")
                        transcriber_message += message['data']
            else:
                logger.info("Not a streaming conversation. Hence getting a full blown transcript")
                async for message in self.tools["transcriber"].transcribe():
                    logger.info(f"message from transcriber {message}")
                    sequence = message["meta_info"]["sequence"]
                    next_task = self._get_next_step(sequence, "transcriber")
                    await self._handle_transcriber_output(next_task, message['data'], message["meta_info"])

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in transcriber {e}")

    async def _send_to_synthesizer(self):
        pass

    async def _receive_from_synthesizer(self):
        async for packet in self.tools["synthesizer"].receive():
            await self.tools["output"].handle(packet)

    async def _synthesize(self, message):
        meta_info = message["meta_info"]
        text = message["data"]
        meta_info["type"] = "audio"
        try:
            if meta_info["is_md5_hash"]:
                logger.info('Sending preprocessed audio response to {}'.format(self.task_config["tools_config"]["output"]["provider"]))
                audio_chunk = await get_raw_audio_bytes_from_base64(self.assistant_name, text,
                        self.task_config["tools_config"]["output"]["format"], local=self.is_local, user_id=self.user_id, assistant_id = self.assistant_id)
                await self.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))

            elif self.task_config["tools_config"]["synthesizer"]["model"] in SUPPORTED_SYNTHESIZER_MODELS.keys():
                logger.info(
                    'Synthesizing chunk via {}'.format(self.task_config["tools_config"]["synthesizer"]["model"]))
                self.synthesizer_characters += len(text)
                async for audio_chunk in self.tools["synthesizer"].generate(text):
                    if not self.conversation_ended:
                        logger.info('Sending synthesized audio chunk to {}'.format(
                            self.task_config["tools_config"]["output"]["provider"]))
                        await self.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))
            else:
                logger.info("other models haven't been done yet")
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
                tasks = [asyncio.create_task(self.tools['input'].handle())]
                if "transcriber" in self.tools:
                    tasks.append(asyncio.create_task(self._listen_transcriber()))
                if "synthesizer" in self.tools and self.task_config["tools_config"]["synthesizer"]["model"] == "xtts":
                    tasks.append(asyncio.create_task(self._receive_from_synthesizer()))

                if self.connected_through_dashboard and self.task_config['task_type'] == "conversation":
                    logger.info("Since it's connected through dashboard, I'll run listen_llm_tas too in case user wants to simply text")
                    self.llm_queue_task = asyncio.create_task(self._listen_llm_input_queue())
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    logger.error(f"Error: {e}")

                # Close connections
                # if "transcriber" in self.tools:
                #     logger.info(f"Closing transcriber")
                #     await self.tools["transcriber"].toggle_connection()
                #     await asyncio.sleep(5) #Making sure whatever message was passed is over


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
            handle_cancellation(f"Websocket got cancelled {self.task_id}")

        except Exception as e:
            # Cancel all tasks on error
            handle_cancellation(f"Exception occurred {e}")
            raise Exception(e)

        finally:
            # Construct output
            if self.task_id == 0:
                output = {"messages": self.history, "conversation_time": time.time() - self.start_time,
                          "label_flow": self.label_flow, "call_sid": self.call_sid, "stream_sid": self.stream_sid,
                          "transcriber_characters": self.transcription_characters,
                          "synthesizer_characters": self.synthesizer_characters}
            else:
                output = self.input_parameters
                if self.task_config["task_type"] == "extraction":
                    output["extracted_data"] = self.extracted_data
                elif self.task_config["task_type"] == "summarization":
                    output["summarization"] = self.summarized_data

            return output


def handle_cancellation(message):
    try:    
    # Cancel all tasks on cancellation
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        logger.info(f"tasks {len(tasks)}")
        for task in tasks:
            logger.info(f"Cancelling task {task.get_name()}")
            task.cancel()
        logger.info(message)
    except Exception as e:
        traceback.print_exc()
        logger

