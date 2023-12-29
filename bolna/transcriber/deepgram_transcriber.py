import asyncio
import websockets
import os
import json
import aiohttp
import time
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import CustomLogger
from bolna.helpers.utils import create_ws_data_packet

custom_logger = CustomLogger(__name__)
load_dotenv()


class DeepgramTranscriber(BaseTranscriber):
    def __init__(self, provider, input_queue=None, model='deepgram', stream=True, language="en", endpointing="400", log_dir_name=None):
        super().__init__(input_queue, log_dir_name)
        self.endpointing = endpointing
        self.language = language
        self.stream = stream
        self.provider = provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = 'deepgram'
        self.audio_format = "audio/webm"
        if not self.stream:
            self.session = aiohttp.ClientSession()
            self.api_url = f"https://api.deepgram.com/v1/listen?model=nova-2&filler_words=true&language={self.language}"

    def get_deepgram_ws_url(self):
        websocket_url = (f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
                         f"&filler_words=true&endpointing={self.endpointing}")

        if self.provider == 'twilio':
            websocket_url = (f"wss://api.deepgram.com/v1/listen?model=nova-2&encoding=mulaw&sample_rate=8000&channels"
                             f"=1&filler_words=true&endpointing={self.endpointing}")

        if self.provider == "playground":
            websocket_url = (f"wss://api.deepgram.com/v1/listen?model=nova-2&encoding=opus&sample_rate=8000&channels"
                             f"=1&filler_words=true&endpointing={self.endpointing}")
        if "en" not in self.language:
            websocket_url += '&language={}'.format(self.language)
        return websocket_url

    async def send_heartbeat(self, ws):
        try:
            while True:
                data = {'type': 'KeepAlive'}
                await ws.send(json.dumps(data))
                await asyncio.sleep(5)  # Send a heartbeat message every 5 seconds
        except Exception as e:
            self.logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong while sending heartbeats to {}".format(self.model))

    async def toggle_connection(self):
        self.connection_on = False
        if self.heartbeat_task is not None:
            await self.heartbeat_task.cancel()
        await self.sender_task.cancel()

    async def _get_http_transcription(self, audio_data):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        headers = {
        'Authorization': 'Token {}'.format(os.getenv('DEEPGRAM_AUTH_TOKEN')),
        'Content-Type': 'audio/webm' #Currently we are assuming this is via browser
        }
        start_time = time.time()
        async with self.session as session:
            async with session.post(self.api_url, data=audio_data, headers=headers) as response:
                response_data = await response.json()
                self.logger.info(f"response_data {response_data} total time {time.time() - start_time}")
                transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
                self.logger.info(f"transcript {transcript} total time {time.time() - start_time}")
                self.meta_info['transcriber_duration'] =  response_data["metadata"]["duration"]
                return create_ws_data_packet(transcript, self.meta_info)

    async def sender(self, ws=None):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] == True:
                    await self._close(ws, data={"type": "CloseStream"})
                    break

                audio_data = ws_data_packet.get('data')
                self.meta_info = ws_data_packet.get('meta_info')
                if self.stream:
                    await asyncio.gather(ws.send(audio_data))
                else:
                    transcription = await self._get_http_transcription(audio_data)
                    yield transcription
        except Exception as e:
            self.logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong")

    async def receiver(self, ws):
        curr_message = ""
        async for msg in ws:
            try:
                msg = json.loads(msg)
                if msg['type'] == "Metadata":
                    self.logger.info(f"Got a summary object {msg}")
                    self.meta_info["transcriber_duration"] = msg["duration"]
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

                transcript = msg['channel']['alternatives'][0]['transcript']

                self.update_meta_info()

                if transcript and len(transcript.strip()) != 0:
                    if await self.signal_transcription_begin(msg):
                        yield create_ws_data_packet("TRANSCRIBER_BEGIN", self.meta_info)

                    curr_message += " " + transcript

                if (msg["speech_final"] and self.callee_speaking) or not self.stream:
                    yield create_ws_data_packet(curr_message, self.meta_info)
                    self.logger.info('User: {}'.format(curr_message))
                    curr_message = ""
                    yield create_ws_data_packet("TRANSCRIBER_END", self.meta_info)
                    self.callee_speaking = False
                    self.last_vocal_frame_time = None
                    self.previous_request_id = self.current_request_id
                    self.current_request_id = None
            except Exception as e:
                self.logger.error(f"Error while getting transcriptions {e}")
                yield create_ws_data_packet("TRANSCRIBER_END", self.meta_info)

    def deepgram_connect(self):
        websocket_url = self.get_deepgram_ws_url()
        extra_headers = {
            'Authorization': 'Token {}'.format(os.getenv('DEEPGRAM_AUTH_TOKEN'))
        }
        deepgram_ws = websockets.connect(websocket_url, extra_headers=extra_headers)

        self.logger.info('got here')
        return deepgram_ws

    async def transcribe(self):
        async with self.deepgram_connect() as deepgram_ws:
            if self.stream:
                self.sender_task = asyncio.create_task(self.sender(deepgram_ws))
                self.heartbeat_task = asyncio.create_task(self.send_heartbeat(deepgram_ws))
                async for message in self.receiver(deepgram_ws):
                    if self.connection_on:
                        yield message
                    else:
                        self.logger.info("Closing the connection")
                        await self._close(deepgram_ws, data={"type": "CloseStream"})
            else:
                async for message in self.sender():
                    yield message
