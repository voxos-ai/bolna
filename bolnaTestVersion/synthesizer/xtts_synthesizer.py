import asyncio
from collections import deque
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed
import json
import os
import audioop
from dotenv import load_dotenv
from .base_synthesizer import BaseSynthesizer
from bolnaTestVersion.helpers.logger_config import configure_logger
from bolnaTestVersion.helpers.utils import create_ws_data_packet
logger = configure_logger(__name__)
load_dotenv()



class XTTSSynthesizer(BaseSynthesizer):
    def __init__(self, audio_format = "wav", stream = False, sampling_rate="24000", buffer_size=400, language = "en", voice = "rohan", **kwargs):
        super().__init__(stream, buffer_size)
        self.buffer = []  # Initialize buffer to make sure we're sending chunks of words instead of token wise
        self.buffered = False
        self.ws_url = os.getenv('TTS_WS')
        self.api_url = os.getenv('TTS_API_URL')
        self.format = self.get_format(audio_format)
        self.stream = stream
        self.language = language
        self.voice = voice
        self.sampling_rate = sampling_rate
        self.websocket_connection = None
        self.audio = b"" 
        self.chunk_count = 1
        self.first_chunk_generated = False
        self.text_queue = deque()
        
    def get_format(self, format):
        return "wav"

    async def _send_payload(self, payload):
        url = self.api_url

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.read() 
                        return data
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")

    async def __generate_http(self, text):
        payload = None
        logger.info(f"text {text}")
        payload = {
            "text": text,
            "model": "xtts",
            "language": self.language,
            "voice": self.voice
        }
        logger.info(f"Sending {payload}")
        response = await self._send_payload(payload)
        return response

    async def _generate(self, text):
        try:
            yield await self._http_tts(text)
        except Exception as e:
            logger.error(f"Error in xtts generate {e}")

    async def sender(self, text, end_of_llm_stream):
        logger.info(f"Sending to the serve {text} which is end_of_llm_stream {end_of_llm_stream}")
        input_message = {
            "text": text,
            "model": "xtts",
            "language": self.language,
            "voice": self.voice,
            "end_of_stream": end_of_llm_stream
        }

        await self.websocket_connection.send(json.dumps(input_message))
        logger.info(f"Sent to the server {input_message}")

    async def receiver(self):
        while True:
            try:
                if self.websocket_connection is not None:
                    chunk = await self.websocket_connection.recv()
                    if not self.buffered and len(self.buffer) < 3:
                        self.buffer.append(chunk)
                        continue
                    if len(self.buffer) == 3:
                        chunk = b''.join(self.buffer)
                        self.buffer = []
                        self.buffered = True

                    if int(self.sampling_rate) != 24000:
                        logger.info(f"Changing the sampling rate to {int(self.sampling_rate)}")
                        chunk = audioop.ratecv(chunk, 2, 1, 24000, int(self.sampling_rate), None)[0]
                        
                    yield chunk


            except ConnectionClosed:
                logger.error("Connection closed")
                break
            except Exception as e:
                logger.error(f"Error in receiving and processing audio bytes {e}")

    async def synthesize(self, text):
        #This is used for one off synthesis mainly for use cases like voice lab and IVR
        audio = await self.__generate_http(text)
        return audio

    async def open_connection(self):
        if self.websocket_connection is None:
                self.websocket_connection = await websockets.connect(self.ws_url)
                logger.info("Connected to the server")
    async def generate(self):
        should_pop_meta_info = True
        try:
            if self.stream:
                async for message in self.receiver():
                    logger.info(f"Received message friom server")
                    yield create_ws_data_packet(message, self.meta_info)
                    
                    if not self.first_chunk_generated:
                        self.meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    if should_pop_meta_info:
                        meta_info = self.text_queue.popleft()
                        should_pop_meta_info = False
                    if message == b'\x00':
                        logger.info("received null byte and hence end of stream")
                        self.meta_info["end_of_synthesizer_stream"] = True
                        yield create_ws_data_packet(message, self.meta_info)
                        self.first_chunk_generated = False
                        should_pop_meta_info = True
            else:
                while True:
                    message = await self.internal_queue.get()
                    logger.info(f"Generating TTS response for message: {message}")
                    meta_info, text = message.get("meta_info"), message.get("data")
                    audio = await self.__generate_http(text)
                    meta_info['text']=  text
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True
                    
                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False

                    yield create_ws_data_packet(audio, meta_info)
                    
                    logger.info(f"Generated TTS response for message: {message}")
        except Exception as e:
                logger.error(f"Error in xtts generate {e}")
    
    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        if self.stream:
            logger.info(f"Pushing message to internal queue {message}")
            meta_info, text = message.get("meta_info"), message.get("data")
            end_of_llm_stream =  "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = meta_info
            self.sender_task = asyncio.create_task(self.sender(text, end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)