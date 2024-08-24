import io
from bolna.helpers.logger_config import configure_logger
import asyncio
from pydub import AudioSegment

logger = configure_logger(__name__)


class BaseSynthesizer:
    def __init__(self, stream=True, buffer_size=40, event_loop= None):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()

    def clear_internal_queue(self):
        logger.info(f"Clearing out internal queue")
        self.internal_queue = asyncio.Queue()
        
    def generate(self):
        pass

    def push(self, text):
        pass
    
    def synthesize(self, text):
        pass

    def get_synthesized_characters(self):
        return 0

    def resample(self, audio_bytes):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_frame_rate(8000)
        audio_segment = audio_segment.set_channels(1)
        audio_buffer = io.BytesIO()
        audio_segment.export(audio_buffer, format="wav")
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        return audio_data

    def get_engine(self):
        return "default"

    def supports_websocket(self):
        return True
