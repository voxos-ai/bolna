import os
import time
from dotenv import load_dotenv
import requests
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, wav_bytes_to_pcm
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()

class CambAiSynthesizer(BaseSynthesizer):
    def __init__(self, voice_id, language, gender, sampling_rate=16000, buffer_size=400, caching=True, **kwargs):
        super().__init__(False, buffer_size)  # stream is always False
        self.voice_id = voice_id
        self.language = language
        self.gender = gender
        logger.info(f"{self.voice_id} initialized")
        self.sample_rate = str(sampling_rate)
        self.first_chunk_generated = False
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

        # Initialize CambAI API Key
        self.subscription_key = kwargs.get("synthesizer_key", os.getenv("CAMBAI_API_KEY"))
        if not self.subscription_key:
            raise ValueError("CambAI API key must be provided")

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def get_engine(self):
        return "CambAI"

    def supports_websocket(self):
        return False

    async def synthesize(self, text):
        audio = await self.__generate_http(text)
        return audio

    def __send_tts_request(self, text):
        url = "https://client.camb.ai/apis/tts"
        headers = {
            "x-api-key": self.subscription_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "gender": self.gender
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("task_id")
        else:
            logger.error(f"Failed to send TTS request: {response.text}")
            return None
        
    def __poll_tts_status(self, task_id):
        url = f"https://client.camb.ai/apis/tts/{task_id}"
        headers = {
            "x-api-key": self.subscription_key
        }
        while True:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                status = response.json().get("status")
                if status == "SUCCESS":
                    return response.json().get("run_id")
                elif status == "PENDING":
                    time.sleep(2)  # Polling interval
                else:
                    logger.error(f"TTS task failed with status: {status}")
                    return None
            else:
                logger.error(f"Failed to poll TTS status: {response.text}")
                return None

    def __get_tts_result(self, run_id):
        url = f"https://client.camb.ai/apis/tts_result/{run_id}"
        headers = {
            "x-api-key": self.subscription_key
        }
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            audio_data = b""
            for chunk in response.iter_content(chunk_size=1024):
                audio_data += chunk
            return audio_data
        else:
            logger.error(f"Failed to get TTS result: {response.text}")
            return None

    async def __generate_http(self, text):
        task_id = self.__send_tts_request(text)
        if task_id:
            run_id = self.__poll_tts_status(task_id)
            if run_id:
                return self.__get_tts_result(run_id)
        return None

    async def generate(self):
        while True:
            logger.info("Generating TTS response")
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")
            if self.caching:
                logger.info("Caching is on")
                cached_message = self.cache.get(text)
                if cached_message:
                    logger.info(f"Cache hit and hence returning quickly {text}")
                    message = cached_message
                else:
                    logger.info(f"Not a cache hit {list(self.cache.data_dict)}")
                    self.synthesized_characters += len(text)
                    message = await self.__generate_http(text)
                    self.cache.set(text, message)
            else:
                logger.info("No caching present")
                self.synthesized_characters += len(text)
                message = await self.__generate_http(text)

            if not self.first_chunk_generated:
                meta_info["is_first_chunk"] = True
                self.first_chunk_generated = True
            else:
                meta_info["is_first_chunk"] = False
            if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                meta_info["end_of_synthesizer_stream"] = True
                self.first_chunk_generated = False

            meta_info['text'] = text
            meta_info['format'] = 'wav'
            message = wav_bytes_to_pcm(message)

            yield create_ws_data_packet(message, meta_info)
