import io
from bolna.helpers.logger_config import configure_logger
import asyncio
import numpy as np
import soxr
import soundfile as sf
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

    def resample(audio_bytes, target_sample_rate=8000):
        audio_data, orig_sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        resampler = soxr.resample(audio_data, orig_sample_rate, target_sample_rate, "VHQ")
        audio_buffer = io.BytesIO()
        audio_segment = AudioSegment(
            data=resampler.tobytes(),
            sample_width=2,
            frame_rate=target_sample_rate,
            channels=1
        )
        audio_segment.export(audio_buffer, format="wav")
        return audio_buffer.getvalue()

    def get_engine(self):
        return "default"

    def supports_websocket(self):
        return True
