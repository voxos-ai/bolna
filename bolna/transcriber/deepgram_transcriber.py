import asyncio
import numpy as np
import torch
import websockets
import os
import json
import aiohttp
import time
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet, int2float
from bolna.helpers.vad import VAD
torch.set_num_threads(1)

logger = configure_logger(__name__)
load_dotenv()


class DeepgramTranscriber(BaseTranscriber):
    def __init__(self, provider, input_queue=None, model='deepgram', stream=True, language="en", endpointing="400",
                 sampling_rate="16000", encoding="linear16", output_queue= None, **kwargs):
        super().__init__(input_queue)
        self.endpointing = endpointing
        self.language = language
        self.stream = stream
        self.provider = provider
        self.heartbeat_task = None
        self.sender_task = None
        self.model = 'deepgram'
        self.sampling_rate = sampling_rate
        self.encoding = encoding
        self.api_key = kwargs.get("transcriber_key", os.getenv('DEEPGRAM_AUTH_TOKEN'))
        self.transcriber_output_queue = output_queue
        self.transcription_task = None
        self.on_device_vad = kwargs.get("on_device_vad", False) if self.stream else False
        logger.info(f"self.stream: {self.stream}")
        if self.on_device_vad:
            self.vad_model = VAD()
            logger.info("on_device_vad is TRue")
            self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.voice_threshold = 0.5
        self.interruption_signalled = False
        self.sampling_rate = 16000
        if not self.stream:
            self.session = aiohttp.ClientSession()
            self.api_url = f"https://api.deepgram.com/v1/listen?model=nova-2&filler_words=true&language={self.language}"

    def get_deepgram_ws_url(self):
        websocket_url = (f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1"
                         f"&filler_words=true&endpointing={self.endpointing}")

        if self.provider == 'twilio':
            websocket_url = (f"wss://api.deepgram.com/v1/listen?model=nova-2&encoding=mulaw&sample_rate=8000&channels"
                             f"=1&filler_words=true&endpointing={self.endpointing}")
            self.sampling_rate = 8000

        if self.provider == "playground":
            websocket_url = (f"wss://api.deepgram.com/v1/listen?model=nova-2&encoding=opus&sample_rate=8000&channels"
                             f"=1&filler_words=true&endpointing={self.endpointing}")
            self.sampling_rate = 8000
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
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong while sending heartbeats to {}".format(self.model))

    async def toggle_connection(self):
        self.connection_on = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
        await self.sender_task.cancel()

    async def _get_http_transcription(self, audio_data):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        headers = {
            'Authorization': 'Token {}'.format(self.api_key),
            'Content-Type': 'audio/webm'  # Currently we are assuming this is via browser
        }
        start_time = time.time()
        async with self.session as session:
            async with session.post(self.api_url, data=audio_data, headers=headers) as response:
                response_data = await response.json()
                logger.info(f"response_data {response_data} total time {time.time() - start_time}")
                transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
                logger.info(f"transcript {transcript} total time {time.time() - start_time}")
                self.meta_info['transcriber_duration'] = response_data["metadata"]["duration"]
                return create_ws_data_packet(transcript, self.meta_info)

    async def _handle_data_packet(self, ws_data_packet, ws):
        if 'eos' in ws_data_packet['meta_info'] and ws_data_packet['meta_info']['eos'] is True:
            logger.info("First closing transcription websocket")
            await self._close(ws, data={"type": "CloseStream"})
            logger.info("Closed transcription websocket and now closing transcription task")
            return True  # Indicates end of processing

        return False

    async def sender(self, ws=None):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                end_of_stream = await self._handle_data_packet(ws_data_packet, ws)
                if end_of_stream:
                    break

                self.meta_info = ws_data_packet.get('meta_info')
                transcription = await self._get_http_transcription(ws_data_packet.get('data'))
                yield transcription
            if self.transcription_task is not None:
                self.transcription_task.cancel()
        except asyncio.CancelledError:
            logger.info("Cancelled sender task")
            return
        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong")

    async def __check_for_vad(self, data):
        start_time = time.time()
        if data is None:
            return
        audio_int16 = np.frombuffer(data, np.int16)
        frame_np = int2float(audio_int16)
        speech_prob = self.vad_model(torch.from_numpy(frame_np.copy()), self.sampling_rate)
        logger.info(f"Speech probability: {speech_prob}")
        if speech_prob > self.voice_threshold:
            logger.info(f"It's definitely human voice and hence interrupting")
            self.interruption_signalled = True
            self.push_to_transcriber_queue(create_ws_data_packet("INTERRUPTION", self.meta_info))

        logger.info(f"Time to run VAD {time.time() - start_time}")
    async def sender_stream(self, ws=None):
        try:
            while True:
                ws_data_packet = await self.input_queue.get()
                audio_bytes = ws_data_packet['data']
                logger.info(f"On device vad {self.on_device_vad} interruption signalled {self.interruption_signalled}")
                if not self.interruption_signalled and self.on_device_vad:
                    await self.__check_for_vad(audio_bytes)
                end_of_stream = await self._handle_data_packet(ws_data_packet, ws)
                if end_of_stream:
                    break

                self.meta_info = ws_data_packet.get('meta_info')
                await asyncio.gather(ws.send(ws_data_packet.get('data')))

        except Exception as e:
            logger.error('Error while sending: ' + str(e))
            raise Exception("Something went wrong")

    async def receiver(self, ws):
        curr_message = ""
        async for msg in ws:
            try:
                msg = json.loads(msg)
                if msg['type'] == "Metadata":
                    logger.info(f"Got a summary object {msg}")
                    self.meta_info["transcriber_duration"] = msg["duration"]
                    yield create_ws_data_packet("transcriber_connection_closed", self.meta_info)
                    return

                transcript = msg['channel']['alternatives'][0]['transcript']

                self.update_meta_info()

                if transcript and len(transcript.strip()) != 0:
                    if await self.signal_transcription_begin(msg):
                        if not self.on_device_vad:
                            self.meta_info["should_interrupt"] = True
                        yield create_ws_data_packet("TRANSCRIBER_BEGIN", self.meta_info)
                    curr_message += " " + transcript

                if msg["speech_final"]  or not self.stream:
                    self.push_to_transcriber_queue(create_ws_data_packet(curr_message, self.meta_info))
                    logger.info('User: {}'.format(curr_message))
                    curr_message = ""
                    self.interruption_signalled = False
                    yield create_ws_data_packet("TRANSCRIBER_END", self.meta_info)
            except Exception as e:
                logger.error(f"Error while getting transcriptions {e}")
                self.interruption_signalled = False
                yield create_ws_data_packet("TRANSCRIBER_END", self.meta_info)

    def push_to_transcriber_queue(self, data_packet):
        self.transcriber_output_queue.put_nowait(data_packet)

    def deepgram_connect(self):
        websocket_url = self.get_deepgram_ws_url()
        extra_headers = {
            'Authorization': 'Token {}'.format(os.getenv('DEEPGRAM_AUTH_TOKEN'))
        }
        deepgram_ws = websockets.connect(websocket_url, extra_headers=extra_headers)

        return deepgram_ws

    async def run(self):
        self.transcription_task = asyncio.create_task(self.transcribe())

    async def transcribe(self):
        try:
            async with self.deepgram_connect() as deepgram_ws:
                if self.stream:
                    self.sender_task = asyncio.create_task(self.sender_stream(deepgram_ws))
                    self.heartbeat_task = asyncio.create_task(self.send_heartbeat(deepgram_ws))
                    async for message in self.receiver(deepgram_ws):
                        if self.connection_on:
                            self.push_to_transcriber_queue(message)
                        else:
                            logger.info("closing the deepgram connection")
                            await self._close(deepgram_ws, data={"type": "CloseStream"})
                else:
                    async for message in self.sender():
                        self.push_to_transcriber_queue(message)
            
            self.push_to_transcriber_queue(create_ws_data_packet("transcriber_connection_closed", self.meta_info))
        except Exception as e:
            logger.error(f"Error in transcribe: {e}")
