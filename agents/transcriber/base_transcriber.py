import json
from dotenv import load_dotenv
import time
import uuid
from agents.helpers.logger_config import configure_logger

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

    def toggle_connection(self):
        self.connection_on = False

    @staticmethod
    async def _close(ws, data):
        try:
            await ws.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error while closing transcriber stream {e}")
