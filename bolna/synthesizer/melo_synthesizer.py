
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



class MeloSynthesizer(BaseSynthesizer):
    def __init__(self, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400,
                 **kwargs):
        super().__init__(stream, buffer_size)
        self.format = "linear16" if audio_format == "pcm" else audio_format
        self.sample_rate = int(sampling_rate)
        self.first_chunk_generated = False
        self.url = os.getenv('MELO_TTS')

        self.voice = kwargs.get('voice')
        self.sample_rate = kwargs.get('sample_rate')
        self.sdp_ratio = kwargs.get('sdp_ratio')
        self.noise_scale=kwargs.get('noise_scale')
        self.noise_scale_w = kwargs.get('noise_scale_w')
        self.speed = kwargs.get('speed')
        # self.voice_id

    async def __generate_http(self, text):
        payload = {
            "voice_id": self.voice,
            "text": text,
            "sr": self.sample_rate,
            "sdp_ratio" : self.sdp_ratio,
            "noise_scale" : self.noise_scale,
            "noise_scale_w" :  self.noise_scale_w,
            "speed" : self.speed
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