import asyncio
import copy
import websockets
import base64
import json
import aiohttp
import os
import traceback
from collections import deque
from .base_synthesizer import BaseSynthesizer
from bolnaTestVersion.helpers.logger_config import configure_logger
from bolnaTestVersion.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample
import pandas as pd
import datetime
from uuid import uuid4
logger = configure_logger(__name__)


class ElevenlabsSynthesizer(BaseSynthesizer):
    def __init__(self, voice, voice_id=None, model="eleven_multilingual_v1", audio_format="mp3", sampling_rate="16000",
                 stream=False, buffer_size=400, synthesier_key=None, **kwargs):
        super().__init__(stream)

        self.LogWebSocket:dict = {"time":[],"event":[],"data":[]}


        self.api_key = os.environ["ELEVENLABS_API_KEY"] if synthesier_key is None else synthesier_key
        # self.api_key = "3550f99d3dc00f4489e72fa29c487a78"
        self.voice = "21m00Tcm4TlvDq8ikWAM"
        self.use_turbo = kwargs.get("use_turbo", False)
        self.model = "eleven_turbo_v2" if self.use_turbo else "eleven_multilingual_v2"
        # self.model = "eleven_turbo_v2"
        self.stream = stream  # Issue with elevenlabs streaming that we need to always send the text quickly
        self.websocket_connection = None
        self.connection_open = False
        self.sampling_rate = sampling_rate
        self.audio_format = "mp3"
        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice}/stream-input?model_id={self.model}&optimize_streaming_latency=2&output_format={self.get_format(self.audio_format, self.sampling_rate)}"
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}?optimize_streaming_latency=2&output_format="
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.text_queue = deque()
        self.meta_info = None
        print(self.ws_url)

    def add_log(self,event:str,data:str=None):
        if data == None:
            self.LogWebSocket["time"].append(str(datetime.datetime.now()))
            self.LogWebSocket["event"].append(event)
            self.LogWebSocket["data"].append(" ")
        else:
            self.LogWebSocket["time"].append(str(datetime.datetime.now))
            self.LogWebSocket["event"].append(event)
            self.LogWebSocket["data"].append(data)
    # Ensuring we only do wav output for now
    def get_format(self, format, sampling_rate):
        # Eleven labs only allow mp3_44100_64, mp3_44100_96, mp3_44100_128, mp3_44100_192, pcm_16000, pcm_22050,
        # pcm_24000, ulaw_8000
        if self.use_mulaw:
            return "ulaw_8000"
        return f"mp3_44100_128"

    # Don't send EOS signal. Let        
    async def sender(self, text, end_of_llm_stream=False):  # sends text to websocket
        if self.websocket_connection is not None and not self.websocket_connection.open:
            self.add_log("connection is closed")
            self.connection_open = False
            logger.info(f"Connection was closed and hence opening connection")
            await self.open_connection()

        if not self.connection_open:
            logger.info("Connecting to elevenlabs websocket...")
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                },
                "xi_api_key": self.api_key,
            }
            self.add_log("send auth token")
            print(bos_message)
            print(self.websocket_connection)
            await self.websocket_connection.send(json.dumps(bos_message))
            print(text)
            self.connection_open = True
        
        if text != "":
            logger.info(f"Sending message {text}")

            input_message = {
                "text": f"{text} ",
                "try_trigger_generation": True,
                "flush": True
            }

            self.add_log(event="send data",data=text)
            print(input_message)
            try:
                await self.websocket_connection.send(json.dumps(input_message))
                print("data sended")
            except Exception as e:
                print(e)
            if end_of_llm_stream:
                self.last_text_sent = True

            # self.connection_open = False

    async def receiver(self):
        while True:
            if not self.connection_open:
                logger.info("Since eleven labs always closes the connection after every leg, simply open it...")
                await self.open_connection()
            try:
                response = await self.websocket_connection.recv()
                data = json.loads(response)
                # logger.info(f"check data {data}")
                # i added
                logger.info("response for isFinal: {}".format(data.get('isFinal', False)))
                if "audio" in data and data["audio"]:
                    self.add_log("get audio")
                    chunk = base64.b64decode(data["audio"])
                    print(chunk)
                    if len(chunk) % 2 == 1:
                        chunk += b'\x00'
                    # @TODO make it better - for example sample rate changing for mp3 and other formats  
                    yield chunk

                    if "isFinal" in data and data["isFinal"]:
                        self.connection_open = False
                        yield b'\x00'
                else:
                    self.add_log("didn't get audio")
                    logger.info("No audio data in the response")
            except websockets.exceptions.ConnectionClosed:
                break

    async def __send_payload(self, payload, format=None):
        headers = {
            'xi-api-key': self.api_key
        }
        url = f"{self.api_url}{self.get_format(self.audio_format, self.sampling_rate)}" if format is None else f"{self.api_url}{format}"
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

    async def synthesize(self, text):
        audio = await self.__generate_http(text, format="mp3_44100_128")
        return audio

    async def __generate_http(self, text, format=None):
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
        response = await self.__send_payload(payload, format=format)
        return response

    # Currently we are only supporting wav output but soon we will incorporate conver
    async def generate(self):
        try:
            if self.stream:
                async for message in self.receiver():
                    logger.info(f"Received message from server")

                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()
                        data = pd.DataFrame(self.LogWebSocket)
                        data.to_csv(f"{uuid4()}.csv")
                        print(self.meta_info)
                    audio = ""

                    if self.use_mulaw:
                        self.meta_info['format'] = 'mulaw'
                        audio = message
                    else:
                        self.meta_info['format'] = "wav"
                        audio = resample(convert_audio_to_wav(message, source_format="mp3"), int(self.sampling_rate),
                                             format="wav")

                    yield create_ws_data_packet(audio, self.meta_info)
                    if not self.first_chunk_generated:
                        self.meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True

                    if self.last_text_sent:
                        # Reset the last_text_sent and first_chunk converted to reset synth latency
                        self.first_chunk_generated = False
                        self.last_text_sent = True

                    if message == b'\x00':
                        logger.info("received null byte and hence end of stream")
                        self.meta_info["end_of_synthesizer_stream"] = True
                        yield create_ws_data_packet(resample(message, int(self.sampling_rate)), self.meta_info)
                        self.first_chunk_generated = False

            else:
                while True:
                    message = await self.internal_queue.get()
                    logger.info(f"Generating TTS response for message: {message}")
                    meta_info, text = message.get("meta_info"), message.get("data")
                    audio = await self.__generate_http(text)

                    meta_info['text'] = text
                    if not self.first_chunk_generated:
                        meta_info["is_first_chunk"] = True
                        self.first_chunk_generated = True

                    if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                        meta_info["end_of_synthesizer_stream"] = True
                        self.first_chunk_generated = False

                    if self.use_mulaw:
                        meta_info['format'] = "mulaw"
                        # not aviable
                    else:
                        meta_info['format'] = "wav"
                        wav_bytes = convert_audio_to_wav(audio, source_format="mp3")
                        audio = resample(wav_bytes, int(self.sampling_rate), format="wav")
                    yield create_ws_data_packet(audio, meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in eleven labs generate {e}")

    async def open_connection(self):
        if self.websocket_connection is None or self.connection_open is False:
            self.websocket_connection = await websockets.connect(self.ws_url)
            logger.info("Connected to the server")
            self.add_log("open connection")
        else:
            self.add_log("call open connection but connection is opened already")
            logger.info("connectin is opened")

    async def push(self, message):
        logger.info(f"Pushed message to internal queue {message}")
        try:
            print(self.stream)
            if self.stream:
                meta_info, text = message.get("meta_info"), message.get("data")
                end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
                self.meta_info = copy.deepcopy(meta_info)
                meta_info["text"] = text
                self.sender_task = asyncio.create_task(self.sender(text, end_of_llm_stream))
                self.text_queue.append(meta_info)
            else:
                self.internal_queue.put_nowait(message)
        except Exception as e:
            print(e)
