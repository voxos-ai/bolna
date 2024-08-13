import io
import torchaudio
from bolna.helpers.logger_config import configure_logger
import asyncio

logger = configure_logger(__name__)


class BaseSynthesizer:
    """
    Initializes the BaseSynthesizer class.
    Args:
        stream (bool, optional): Whether to stream the audio or not. Defaults to True.
        buffer_size (int, optional): The size of the buffer. Defaults to 40.
        event_loop (None, optional): The event loop to use. Defaults to None.
    """
    def __init__(self, stream=True, buffer_size=40, event_loop= None):
        self.stream = stream
        self.buffer_size = buffer_size
        self.internal_queue = asyncio.Queue()

    def clear_internal_queue(self):
        """
        Clears out the internal queue.
        """
        logger.info(f"Clearing out internal queue")
        self.internal_queue = asyncio.Queue()
        
    def generate(self):
        """
        Generates the audio.
        """
        pass

    def push(self, text):
        """
        Pushes the given text to the synthesizer.
        Args:
            text (str): The text to be synthesized.
        """
        pass
    
    def synthesize(self, text):
        """
        Synthesizes the given text.
        Args:
            text (str): The text to be synthesized.
        """
        pass

    def get_synthesized_characters(self):
        """
        Returns the number of synthesized characters.
        Returns:
            int: The number of synthesized characters.
        """
        return 0

    def resample(self, audio_bytes):
        """
        Resamples the audio.
        Args:
            audio_bytes (bytes): The audio bytes to be resampled.
        Returns:
            bytes: The resampled audio bytes.
        """
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, orig_sample_rate = torchaudio.load(audio_buffer)
        resampler = torchaudio.transforms.Resample(orig_sample_rate, 8000)
        audio_waveform = resampler(waveform)
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, audio_waveform, 8000, format="wav")
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        return audio_data

    def get_engine(self):
        """
        Returns the engine used for synthesis.
        Returns:
            str: The engine used for synthesis.
        """
        return "default"

    def supports_websocket(self):
        """
        Checks if the synthesizer supports websocket.
        Returns:
            bool: True if the synthesizer supports websocket, False otherwise.
        """
        return True
