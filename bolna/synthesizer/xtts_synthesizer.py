import asyncio
import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed
import json
import os
import audioop
from dotenv import load_dotenv
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class XTTSSynthesizer(BaseSynthesizer):
    def __init__(self, audio_format = "wav", stream = False, sampling_rate="24000", buffer_size=400, language = "en", voice = "rohan"):
        super().__init__(stream, buffer_size)
        self.websocket_connection = None
        self.buffer = []  # Initialize buffer to make sure we're sending chunks of words instead of token wise
        self.buffered = False
        self.ws_url = os.getenv('TTS_WS')
        self.api_url = os.getenv('TTS_API_URL')
        self.format = audio_format
        self.stream = stream
        self.language = language
        self.voice = voice
        self.sampling_rate = sampling_rate

    # async def _connect(self):
    #     if self.websocket_connection is None:
    #         self.websocket_connection = websockets.connect(self.ws_url)

    def get_websocket_connection(self):
        if self.websocket_connection is None:
            self.websocket_connection = websockets.connect(self.ws_url)
        return self.websocket_connection

    # @staticmethod
    # async def sender(ws, text):
    #     if text != "" and text != "LLM_END":
    #         input_message = {
    #             "text": text,
    #             "model": "xtts"
    #         }
    #         await ws.send(json.dumps(input_message))
    #
    # async def receiver(self, ws):
    #     while True:
    #         try:
    #             chunk = await ws.recv()
    #             if not self.buffered and len(self.buffer) < 3:
    #                 self.buffer.append(chunk)
    #                 continue
    #             if len(self.buffer) == 3:
    #                 chunk = b''.join(self.buffer)
    #                 self.buffer = []
    #                 self.buffered = True
    #
    #             if self.format == "pcm":
    #                 chunk = audioop.ratecv(chunk, 2, 1, 24000, 8000, None)[0]
    #
    #             yield chunk
    #
    #         except websockets.exceptions.ConnectionClosed:
    #             logger.error("Connection closed")
    #             break
    #         except Exception as e:
    #             logger.error(f"Error in receiving and processing audio bytes {e}")

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

    async def _http_tts(self, text):
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

    async def sender(self, ws, text):
        if text != "" and text != "LLM_END":
            input_message = {
                "text": text,
                "model": "xtts",
                "language": self.language,
                "voice": self.voice
            }

            await asyncio.gather(ws.send(json.dumps(input_message)))

    async def receiver(self, ws):
        while True:
            try:
                chunk = await ws.recv()
                if not self.buffered and len(self.buffer) < 3:
                    self.buffer.append(chunk)
                    continue
                if len(self.buffer) == 3:
                    chunk = b''.join(self.buffer)
                    self.buffer = []
                    self.buffered = True

                if self.format == "pcm":
                    # for pcm audio
                    chunk = audioop.ratecv(chunk, 2, 1, 24000, 8000, None)[0]

                yield chunk

            except ConnectionClosed:
                logger.error("Connection closed")
                break
            except Exception as e:
                logger.error(f"Error in receiving and processing audio bytes {e}")

    async def generate_stream_response(self, text):
        async with self.get_websocket_connection() as ws:
            self.sender_task = asyncio.create_task(self.sender(ws, text))
            async for message in self.receiver(ws):
                yield message

    async def generate(self, text):
        try:
            if text != "" and text != "LLM_END":
                async for message in self.generate_stream_response(text):
                    logger.info('yielding for {}'.format(text))
                    yield message
        except Exception as e:
            logger.error(f"Error in xtts generate {e}")
