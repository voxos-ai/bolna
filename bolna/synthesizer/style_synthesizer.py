import aiohttp
import os
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet
from .base_synthesizer import BaseSynthesizer
import json
import base64


load_dotenv()
logger = configure_logger(__name__)



class StyleSynthesizer(BaseSynthesizer):
    def __init__(self, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400,
                 **kwargs):
        super().__init__(stream, buffer_size)
        self.format = "linear16" if audio_format == "pcm" else audio_format
        self.sample_rate = int(sampling_rate)
        self.first_chunk_generated = False
        self.url = os.getenv('BOLNA_TTS')

        self.voice = kwargs.get('voice')
        self.sample_rate = kwargs.get('sample_rate')
        self.embedding_scale = kwargs.get('embedding_scale')
        self.diffusion_steps=kwargs.get('diffusion_steps')

        self.counter = 0
       

    async def __generate_http(self, text):
        payload = {
        "model": "StyleTTS",
        "config": {
            "text": text,
            "sr": self.sample_rate,
            "diffusion_steps": self.diffusion_steps,
            "embedding_scale" : self.diffusion_steps
            }
        }
       
        headers = {
            'Content-Type': 'application/json'
        }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(self.url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        res_json:dict = json.loads(await response.text())
                        chunk = base64.b64decode(res_json["audio"])
                        with open(f"../venv/audio/{self.counter}.wav",'wb') as file:
                            file.write(chunk)
                            self.counter+=1
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