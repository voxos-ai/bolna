import asyncio
import websockets
import os
from dotenv import load_dotenv
import torch
from .base_transcriber import BaseTranscriber
from agents.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()
torch.set_num_threads(1)


class DeepgramTranscriber(BaseTranscriber):
    def __init__(self, provider, input_queue=None, model="deepgram", stream=True, language="en", endpointing="400"):
        super().__init__(input_queue, model, stream)
        self.endpointing = endpointing
        self.language = language
        self.stream = stream
        self.provider = provider

    def get_deepgram_ws_url(self):
        websocket_url = (f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
                         f"&filler_words=true&endpointing={self.endpointing}")

        if self.provider == 'twilio':
            websocket_url = (f"wss://api.deepgram.com/v1/listen?model=nova-2&encoding=mulaw&sample_rate=8000&channels"
                             f"=1&filler_words=true&endpointing={self.endpointing}")

        if "en" not in self.language:
            websocket_url += '&language={}'.format(self.language)
        logger.info('Websocket URL: {}'.format(websocket_url))
        return websocket_url

    def deepgram_connect(self):
        websocket_url = self.get_deepgram_ws_url()
        extra_headers = {
            'Authorization': 'Token {}'.format(os.getenv('DEEPGRAM_AUTH_TOKEN'))
        }
        deepgram_ws = websockets.connect(websocket_url, extra_headers=extra_headers)

        return deepgram_ws

    async def transcribe(self):
        async with self.deepgram_connect() as deepgram_ws:
            sender_task = asyncio.create_task(self.sender(deepgram_ws))
            heartbeat_task = asyncio.create_task(self.send_heartbeat(deepgram_ws))

            async for message in self.receiver(deepgram_ws):
                if self.connection_on:
                    yield message
                else:
                    logger.info("Closing the connection")
                    await self._close(deepgram_ws, data={"type": "CloseStream"})
