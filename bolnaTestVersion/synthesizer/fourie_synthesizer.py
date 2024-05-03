import json
import os
import aiohttp
from dotenv import load_dotenv
import audioop
from bolnaTestVersion.helpers.logger_config import configure_logger
from bolnaTestVersion.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes
from .base_synthesizer import BaseSynthesizer
from openai import AsyncOpenAI
import io

logger = configure_logger(__name__)
load_dotenv()

class FourieSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_id = None, audio_format="mp3", gender = "male" ,stream=False, buffer_size=400, **kwargs):
        super().__init__(stream, buffer_size)
        self.voice = voice
        self.sample_rate = 48000
        self.voice_id = voice_id
        self.gender = gender
        self.api_key = os.getenv("FOURIE_API_KEY")
  
    async def synthesize(self, text):
        #This is used for one off synthesis mainly for use cases like voice lab and IVR
        audio = await self.__generate_http(text)
        return audio
    

    async def __generate_http(self, text):
        payload = None
        logger.info(f"text {text}")

        payload = {
            "text": text,
            "locale": "en-US",
            "gender": self.gender,
            "speaker_id": self.voice_id,
            "speed": 1,
            "pitch": 0,
            "volume": 1
            }
        url = "https://api.fourie.ai/tts"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
            }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"data {data}")
                        audio_url =  data["audio_clip"]
                        
                        async with session.get(audio_url) as audio_response:
                            if audio_response.status == 200:
                                audio_bytes = await audio_response.read()  # Return raw bytes of the audio file
                                logger.info(f"audio_bytes {len(audio_bytes)}")
                                return audio_bytes
                            else:
                                logger.error(f"Error retrieving audio file: {audio_response.status}")
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")
        return response
    async def generate(self):
        try:
            while True:
                message = await self.internal_queue.get()
                logger.info(f"Generating TTS response for message: {message}")
                meta_info, text = message.get("meta_info"), message.get("data")
                audio = await self.__generate_http(text)
                yield create_ws_data_packet(audio, meta_info)
                if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                    meta_info["end_of_synthesizer_stream"] = True
        except Exception as e:
                logger.error(f"Error in openai generate {e}")

    async def open_connection(self):
        pass

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        self.internal_queue.put_nowait(message)