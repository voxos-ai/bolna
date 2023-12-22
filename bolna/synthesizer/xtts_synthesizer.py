import asyncio
import websockets
import json
import os
from bolna.helpers.logger_config import configure_logger
import audioop
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__, True)


class XTTSSynthesizer(BaseSynthesizer):
    def __init__(self, model, audio_format, stream, buffer_size=400):
        super().__init__(stream, buffer_size)
        self.websocket_connection = None
        self.buffer = []  # Initialize buffer to make sure we're sending chunks of words instead of token wise
        self.buffered = False
        self.ws_url = os.getenv('TTS_WS')
        self.format = audio_format

    async def _connect(self):
        if self.websocket_connection is None:
            self.websocket_connection = websockets.connect(self.ws_url)

    def get_websocket_connection(self):
        if self.websocket_connection is None:
            self.websocket_connection = websockets.connect(self.ws_url)
        return self.websocket_connection

    @staticmethod
    async def sender(ws, text):
        if text != "" and text != "LLM_END":
            input_message = {
                "text": text,
                "model": "xtts"
            }
            await ws.send(json.dumps(input_message))

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
                    chunk = audioop.ratecv(chunk, 2, 1, 24000, 8000, None)[0]

                yield chunk

            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection closed")
                break
            except Exception as e:
                logger.error(f"Error in receiving and processing audio bytes {e}")

    async def generate(self, text):
        try:
            async with self.get_websocket_connection() as xtts_ws:
                sender_task = asyncio.create_task(self.sender(xtts_ws, text))

                async for message in self.receiver(xtts_ws):
                    yield message
        except Exception as e:
            logger.error(f"Error in xtts generate {e}")
