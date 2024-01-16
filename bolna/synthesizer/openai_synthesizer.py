from dotenv import load_dotenv
import audioop
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet
from .base_synthesizer import BaseSynthesizer
from openai import AsyncOpenAI
import io

logger = configure_logger(__name__)
load_dotenv()

class OPENAISynthesizer(BaseSynthesizer):
    def __init__(self, voice, audio_format="mp3", model = "tts-1", stream=False, buffer_size=400):
        super().__init__(stream, buffer_size)
        self.format = audio_format.lower()
        self.voice = voice
        self.sample_rate = 24000
        self.async_client = AsyncOpenAI()
        self.model = model
  

    async def synthesize(self, text):
        #This is used for one off synthesis mainly for use cases like voice lab and IVR
        audio = await self.__generate_http(text)
        return audio

    async def __generate_http(self, text):
        spoken_response = await self.async_client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format=self.format,
            input=text
            )

        buffer = io.BytesIO()
        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)
        return buffer.getvalue()
    
    async def __generate_stream(self, text):
        spoken_response = await self.async_client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format=self.format,
            input=text
            )

        for chunk in spoken_response.iter_bytes(chunk_size=4096):
            yield chunk

    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                if self.stream:
                    for chunk in self.__generate_stream(text):
                        yield create_ws_data_packet(chunk, meta_info)
                else:
                    audio = await self.__generate_http(text)
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                    yield create_ws_data_packet(audio, meta_info)
        except Exception as e:
                logger.error(f"Error in eleven labs generate {e}")

    async def open_connection(self):
        pass

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)
