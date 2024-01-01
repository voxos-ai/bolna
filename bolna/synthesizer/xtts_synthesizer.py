import aiohttp
import websockets
import json
import os
import audioop
from .base_synthesizer import BaseSynthesizer


class XTTSSynthesizer(BaseSynthesizer):
    def __init__(self, audio_format="wav", stream=False, buffer_size=400, language="en", voice="rohan",
                 log_dir_name=None):
        super().__init__(stream, buffer_size, log_dir_name)
        self.websocket_connection = None
        self.buffer = []  # Initialize buffer to make sure we're sending chunks of words instead of token wise
        self.buffered = False
        self.ws_url = os.getenv('TTS_WS')
        self.format = audio_format.lower()
        self.stream = stream
        self.language = language
        self.voice = voice

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
                self.logger.error("Connection closed")
                break
            except Exception as e:
                self.logger.error(f"Error in receiving and processing audio bytes {e}")

    async def _send_payload(self, payload):
        url = f'http://localhost:8000/generate'

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.read()
                        self.logger.info(f"Received audio chunk {data}")
                        return data
                    else:
                        self.logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                self.logger.info("Payload was null")

    async def _http_tts(self, text):
        self.logger.info(f"text {text}")
        payload = {
            "text": text,
            "model": "xtts",
            "language": self.language,
            "voice": self.voice
        }
        self.logger.info(f"Sending {payload}")
        response = await self._send_payload(payload)
        return response

    async def generate(self, text):
        try:
            yield await self._http_tts(text)
        except Exception as e:
            self.logger.error(f"Error in xtts generate {e}")
