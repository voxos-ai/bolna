import io
import torchaudio
import uvloop
from bolna.helpers.logger_config import configure_logger
import asyncio

logger = configure_logger(__name__)


class BaseSynthesizer:
    def __init__(self, stream=True, buffer_size=40, event_loop= None):
        self.stream = stream
        self.buffer_size = buffer_size
        self.__event_loop = uvloop.new_event_loop()
        asyncio.set_event_loop(self.__event_loop)
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

    def resample(self, audio_bytes):
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, orig_sample_rate = torchaudio.load(audio_buffer)
        resampler = torchaudio.transforms.Resample(orig_sample_rate, 8000)
        audio_waveform = resampler(waveform)
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, audio_waveform, 8000, format="wav")
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        return audio_data
    def get_event_loop(self):
        return self.__event_loop