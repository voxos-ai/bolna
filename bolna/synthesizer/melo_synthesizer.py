
import aiohttp
import os
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, wav_bytes_to_pcm
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
import json
import base64

load_dotenv()
logger = configure_logger(__name__)

class MeloSynthesizer(BaseSynthesizer):
    def __init__(self, audio_format="pcm", sampling_rate="8000", stream=False, buffer_size=400, caching = True,
                 **kwargs):
        super().__init__(stream, buffer_size)
        self.format = "linear16" if audio_format == "pcm" else audio_format
        self.sample_rate = int(sampling_rate)
        self.first_chunk_generated = False
        self.url = os.getenv('MELO_TTS')

        MELOTTS_VOICE_MAP = {
            "Alex": "EN-US",
            "Ariel": "EN-BR",
            "Taylor": "EN-AU",
            "Casey": "EN-Default",
            "Aadi": "EN_INDIA"
        }


        self.voice_name = kwargs.get('voice', "Casey")
        self.voice = MELOTTS_VOICE_MAP.get(self.voice_name)
        self.sample_rate = kwargs.get('sample_rate')
        self.sdp_ratio = kwargs.get('sdp_ratio')
        self.noise_scale=kwargs.get('noise_scale')
        self.noise_scale_w = kwargs.get('noise_scale_w')
        self.speed = kwargs.get('speed')
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

    def get_synthesized_characters(self):
        return self.synthesized_characters

    async def __generate_http(self, text):
        try: 
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
                logger.info(f"Posting {self.url}")
                async with session.post(self.url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        res_json:dict = json.loads(await response.text())
                        chunk = base64.b64decode(res_json["audio"])
                        return chunk
                    
        except Exception as e:
            logger.error(f"Could not synthesizer")

    async def synthesize(self, text):
        # This is used for one off synthesis mainly for use cases like voice lab and IVR
        try:
            logger.info(f"Synthesizeing")
            audio = await self.__generate_http(text)
            return audio
        except Exception as e:
            logger.error(f"Could not synthesize {e}")


    async def open_connection(self):
        pass
    
    def supports_websocket(self):
        return False
    async def generate(self):
        while True:
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")

            if self.caching:
                logger.info(f"Caching is on")
                if self.cache.get(text):
                    logger.info(f"Cache hit and hence returning quickly {text}")
                    audio = self.cache.get(text)
                else:
                    logger.info(f"Not a cache hit {list(self.cache.data_dict)}")
                    self.synthesized_characters += len(text)
                    audio = await self.__generate_http(text)
                    self.cache.set(text, audio)
            else:
                logger.info(f"No caching present")
                self.synthesized_characters += len(text)
                audio = await self.__generate_http(text)
            
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
            if self.sample_rate == 8000:
                audio = wav_bytes_to_pcm(audio)
            logger.info(f"Sending sample rate of {self.sample_rate}")
            yield create_ws_data_packet(audio, meta_info)

    async def push(self, message):
        logger.info("Pushed message to internal queue")
        self.internal_queue.put_nowait(message)