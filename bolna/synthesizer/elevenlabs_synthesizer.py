import asyncio
import websockets
import base64
import json
import aiohttp
import os
import queue
import logging
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ElevenlabsSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_id, model="eleven_multilingual_v1", audio_format = "pcm", sampling_rate = "16000", stream=False, buffer_size=400):
        super().__init__(stream)
        self.api_key = os.environ["ELEVENLABS_API_KEY"]
        self.voice = voice_id
        self.model = model
        self.stream = stream
        self.websocket_connection = None
        self.connection_open = False

    async def _connect(self, ws):
        if self.websocket_connection is None:
            self.websocket_connection = websockets.connect('wss://api.elevenlabs.io/v1/text-to-speech/8NKMgvCBNI0jN8dVStjg/stream-input?model_id=eleven_multilingual_v1')
    
    def _get_websocket_connection(self):
        if self.websocket_connection is None:
            self.websocket_connection = websockets.connect('wss://api.elevenlabs.io/v1/text-to-speech/8NKMgvCBNI0jN8dVStjg/stream-input?model_id=eleven_multilingual_v1')
        return self.websocket_connection
        
    async def _stream_tts(self):
        logger.info("Streaming TTS from eleven labs")
        async with self._get_websocket_connection() as ws:
            async def sender(ws): # sends text to websocket
                while True:
                    if not self.connection_open:
                        bos_message = {
                            "text": " ",
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": True
                            },
                            "xi_api_key": self.api_key,
                        }
                        await ws.send(json.dumps(bos_message))
                        self.connection_open = True
        
                    try:
                        text = self.input_queue.get(timeout = 0.5)
                    except queue.Empty:
                        logger.debug("Empty queue, sleeping")
                        await asyncio.sleep(0.1)
                        continue  

                    if text != "" and text != "EOS":
                        input_message = {
                        "text": f"{text} ",
                        "try_trigger_generation": True
                        }
                        logger.info(f"Got message {text}")
                        await ws.send(json.dumps(input_message))

                    if text == "EOS":
                        eos_message = {
                            "text": ""
                        }
                        await ws.send(json.dumps(eos_message))
                        break

            async def receiver(ws):
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)
                        logger.info("Server response:")
                        if data["audio"]:
                            chunk = base64.b64decode(data["audio"])
                            self.output_queue.put_nowait(chunk)
                        else:
                            logger.info("No audio data in the response")
                    except websockets.exceptions.ConnectionClosed:
                        break
            await asyncio.gather(sender(ws), receiver(ws))

    async def _send_payload(self, payload):
        url = f'https://api.elevenlabs.io/v1/text-to-speech/{self.voice}'

        headers = {
            'xi-api-key': self.api_key,
            'accept':"application/mpeg+base64"
        }

        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.read()
                        return data
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")

    async def _http_tts(self, text):
        payload = None
        logger.info(f"text {text}")
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "optimize_streaming_latency": 3
            }
        }
        response = await self._send_payload(payload)
        return response
                    
    
    async def generate(self, text):
        audio = await self._http_tts(text)
        yield audio