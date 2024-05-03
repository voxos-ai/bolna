import aiohttp
import os
from dotenv import load_dotenv
from bolnaTestVersion.helpers.logger_config import configure_logger
from bolnaTestVersion.helpers.utils import create_ws_data_packet
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"


class DeepgramSynthesizer(BaseSynthesizer):
    def __init__(self, voice, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400,
                 **kwargs):
        super().__init__(stream, buffer_size)
        self.format = "linear16" if audio_format == "pcm" else audio_format
        self.voice = voice
        self.sample_rate = str(sampling_rate)
        self.first_chunk_generated = False
        self.api_key = kwargs.get("transcriber_key", os.getenv('DEEPGRAM_AUTH_TOKEN'))

    async def __generate_http(self, text):
        headers = {
            "Authorization": "Token {}".format(self.api_key),
            "Content-Type": "application/json"
        }
        url = DEEPGRAM_TTS_URL + "?encoding={}&container=none&sample_rate={}&model={}".format(
            self.format, self.sample_rate, self.voice
        )

        payload = {
            "text": text
        }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        chunk = await response.read()
                        yield chunk
            else:
                logger.info("Payload was null")

    async def open_connection(self):
        pass

    async def generate(self):
        while True:
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")

            meta_info, text = message.get("meta_info"), message.get("data")
            async for message in self.__generate_http(text):
                if not self.first_chunk_generated:
                    meta_info["is_first_chunk"] = True
                    self.first_chunk_generated = True
                else:
                    meta_info["is_first_chunk"] = False
                if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                    meta_info["end_of_synthesizer_stream"] = True
                    self.first_chunk_generated = False

                meta_info['text'] = text
                meta_info['format'] = self.format
                yield create_ws_data_packet(message, meta_info)

    async def push(self, message):
        logger.info("Pushed message to internal queue")
        self.internal_queue.put_nowait(message)
