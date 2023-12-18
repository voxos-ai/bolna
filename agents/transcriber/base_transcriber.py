import asyncio
import json
from dotenv import load_dotenv
import numpy as np
import time
import traceback
import uuid
import torch
from agents.helpers.utils import create_ws_data_packet, int2float
from agents.helpers.logger_config import configure_logger

torch.set_num_threads(1)
load_dotenv()


logger = configure_logger(__name__)


class BaseTranscriber:
    def __init__(self, input_queue=None, model='deepgram', stream=True):
        self.input_queue = input_queue
        self.model = model
        self.connection_on = True
        self.callee_speaking = False
        self.caller_speaking = False
        self.meta_info = None
        self.transcription_start_time = 0
        self.last_vocal_frame_time = None
        self.previous_request_id = None
        self.current_request_id = None
        # self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad',
        #                                                 force_reload=False)
        # (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator,
        #  self.collect_chunks) = self.vad_utils

    async def get_audio_confidence(self, audio_data):
        t1 = time.time()
        # logger.info('[START] computing confidence for audio at {}'.format(t1))
        # confidence check of incoming audio
        audio_int16 = np.frombuffer(audio_data, np.int16)
        audio_float32 = int2float(audio_int16)
        vad_time = time.time()
        confidence = self.vad_model(torch.from_numpy(audio_float32), 16000).item()
        #logger.info(f"VAD inference time {time.time() - vad_time} and vad confidence {confidence}")
        if confidence > 0.5:
            self.last_vocal_frame_time = time.time()
        t2 = time.time()

    async def send_heartbeat(self, ws):
        try:
            while True:
                data = {'type': 'KeepAlive'}
                await ws.send(json.dumps(data))
                await asyncio.sleep(5)  # Send a heartbeat message every 5 seconds
        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong while sending heartbeats to {}".format(self.model))

    async def sender(self, ws):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                audio_data = ws_data_packet.get('data')
                self.meta_info = ws_data_packet.get('meta_info')
                await asyncio.gather(ws.send(audio_data))
        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong")

    def update_meta_info(self, transcript):
        self.meta_info['request_id'] = self.current_request_id if self.current_request_id else None
        self.meta_info['previous_request_id'] = self.previous_request_id
        self.meta_info['origin'] = "transcriber"
        logger.debug(f"received message with request id {self.meta_info['request_id']} length of transcript {len(transcript)} ")

    @staticmethod
    def generate_request_id():
        return str(uuid.uuid4())

    async def signal_transcription_begin(self, msg):
        send_begin_packet = False
        if self.current_request_id is None:
            self.current_request_id = self.generate_request_id()
            logger.info(f"Setting current request id to {self.current_request_id}")
            self.meta_info['request_id'] = self.current_request_id

        if not self.callee_speaking:
            self.callee_speaking = True
            logger.debug("Making callee speaking true")
            self.transcription_start_time = time.time() - msg['duration']
            send_begin_packet = True

        return send_begin_packet

    async def log_latency_info(self):
        transcription_completion_time = time.time()
        if self.last_vocal_frame_time:
            logger.info(
                f"################ Time latency: For request {self.meta_info['request_id']}, user started speaking at {self.transcription_start_time}, last audio frame received at {self.last_vocal_frame_time} transcription_completed_at {transcription_completion_time} overall latency {transcription_completion_time - self.last_vocal_frame_time}")
        else:
            logger.info(
                f"No confidence for the last vocal timeframe. Over transcription time {transcription_completion_time - self.transcription_start_time}")

    async def receiver(self, ws):
        curr_message = ""
        async for msg in ws:
            try:
                logger.info(f"Got response from {self.model} {msg}")
                msg = json.loads(msg)
                transcript = msg['channel']['alternatives'][0]['transcript']

                self.update_meta_info(transcript)

                if transcript and len(transcript.strip()) != 0:
                    if await self.signal_transcription_begin(msg):
                        yield create_ws_data_packet("TRANSCRIBER_BEGIN", self.meta_info)

                    curr_message += " " + transcript

                if msg["speech_final"] and self.callee_speaking:
                    yield create_ws_data_packet(curr_message, self.meta_info)
                    curr_message = ""
                    yield create_ws_data_packet("TRANSCRIBER_END", self.meta_info)
                    self.callee_speaking = False
                    self.last_vocal_frame_time = None
                    self.previous_request_id = self.current_request_id
                    self.current_request_id = None
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error while getting transcriptions {e}")
                yield create_ws_data_packet("TRANSCRIBER_END", self.meta_info)

    def toggle_connection(self):
        self.connection_on = False

    @staticmethod
    async def _close(ws, data):
        try:
            await ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error while closing transcriber stream {e}")
