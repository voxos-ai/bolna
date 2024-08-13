import asyncio
import copy
import websockets
import base64
import json
import aiohttp
import os
import traceback
from collections import deque

from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample


logger = configure_logger(__name__)


class ElevenlabsSynthesizer(BaseSynthesizer):
    """
    A class representing a synthesizer for Elevenlabs text-to-speech API.
    Args:
        voice (str): The voice to be used for synthesis.
        voice_id (str): The ID of the voice to be used for synthesis.
        model (str, optional): The model to be used for synthesis. Defaults to "eleven_turbo_v2_5".
        audio_format (str, optional): The audio format of the synthesized output. Defaults to "mp3".
        sampling_rate (str, optional): The sampling rate of the synthesized output. Defaults to "16000".
        stream (bool, optional): Whether to stream the synthesized output. Defaults to False.
        buffer_size (int, optional): The buffer size for streaming. Defaults to 400.
        temperature (float, optional): The stability of the synthesized output. Defaults to 0.9.
        similarity_boost (float, optional): The similarity boost of the synthesized output. Defaults to 0.5.
        synthesier_key (str, optional): The API key for the synthesizer. Defaults to None.
        caching (bool, optional): Whether to enable caching of synthesized outputs. Defaults to True.
    Attributes:
        api_key (str): The API key for the synthesizer.
        voice (str): The ID of the voice to be used for synthesis.
        use_turbo (bool): Whether to use turbo mode for synthesis.
        model (str): The model to be used for synthesis.
        stream (bool): Whether to stream the synthesized output.
        websocket_connection: The WebSocket connection for streaming synthesis.
        connection_open (bool): Whether the WebSocket connection is open.
        sampling_rate (str): The sampling rate of the synthesized output.
        audio_format (str): The audio format of the synthesized output.
        use_mulaw (bool): Whether to use mulaw encoding for the synthesized output.
        ws_url (str): The WebSocket URL for streaming synthesis.
        api_url (str): The API URL for non-streaming synthesis.
        first_chunk_generated (bool): Whether the first chunk of synthesized output has been generated.
        last_text_sent (bool): Whether the last text has been sent for synthesis.
        text_queue (deque): A queue for storing texts to be synthesized.
        meta_info (None): Information about the synthesized output.
        temperature (float): The stability of the synthesized output.
        similarity_boost (float): The similarity boost of the synthesized output.
        caching (bool): Whether caching of synthesized outputs is enabled.
        cache: The cache for storing synthesized outputs.
        synthesized_characters (int): The number of synthesized characters.
        previous_request_ids: The IDs of previous synthesis requests.
    Methods:
        get_format(format, sampling_rate): Get the format for the synthesized output.
        get_engine(): Get the model used for synthesis.
        sender(text, end_of_llm_stream): Send text to the WebSocket connection.
        receiver(): Receive synthesized output from the WebSocket connection.
        __send_payload(payload, format): Send a payload for non-streaming synthesis.
        synthesize(text): Synthesize text using non-streaming synthesis.
        __generate_http(text, format): Generate synthesized output using non-streaming synthesis.
        get_synthesized_characters(): Get the number of synthesized characters.
        generate(): Generate synthesized output.
    """
    # Rest of the code...
    def __init__(self, voice, voice_id, model="eleven_turbo_v2_5", audio_format="mp3", sampling_rate="16000",
                 stream=False, buffer_size=400, temperature = 0.9, similarity_boost = 0.5, synthesier_key=None, 
                 caching=True, **kwargs):
        super().__init__(stream)
        self.api_key = os.environ["ELEVENLABS_API_KEY"] if synthesier_key is None else synthesier_key
        self.voice_id = voice_id
        self.use_turbo = kwargs.get("use_turbo", False)
        self.model = model
        logger.info(f"Using turbo or not {self.model}")
        self.stream = stream  # Issue with elevenlabs streaming that we need to always send the text quickly
        self.websocket_connection = None
        self.connection_open = False
        self.sampling_rate = sampling_rate
        self.audio_format = "mp3"
        self.use_mulaw = kwargs.get("use_mulaw", False)
        self.ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model}&optimize_streaming_latency=2&output_format={self.get_format(self.audio_format, self.sampling_rate)}"
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}?optimize_streaming_latency=2&output_format="
        self.first_chunk_generated = False
        self.last_text_sent = False
        self.text_queue = deque()
        self.meta_info = None
        self.temperature = 0.8
        self.similarity_boost = similarity_boost
        self.caching = caching
        if self.caching:
            self.cache = InmemoryScalarCache()
        self.synthesized_characters = 0
        self.previous_request_ids = []

    # Ensuring we only do wav output for now
    def get_format(self, format, sampling_rate):
        """
        Returns the audio format based on the given format and sampling rate.

        Parameters:
        - format (str): The desired audio format.
        - sampling_rate (int): The desired sampling rate.

        Returns:
        - str: The audio format based on the given format and sampling rate.
        """
        # Eleven labs only allow mp3_44100_64, mp3_44100_96, mp3_44100_128, mp3_44100_192, pcm_16000, pcm_22050,
        # pcm_24000, ulaw_8000
        if self.use_mulaw:
            return "ulaw_8000"
        return f"mp3_44100_128"

    def get_engine(self):
        """
        Get the model used for synthesis.

        Returns:
        - str: The model used for synthesis.
        """
        return self.model

    # Don't send EOS signal. Let
    async def sender(self, text, end_of_llm_stream=False):  # sends text to websocket
        async def sender(self, text, end_of_llm_stream=False):
            """
            Sends text to the websocket.
            Parameters:
            - text (str): The text to be sent.
            - end_of_llm_stream (bool): Indicates whether it is the end of the LLM stream. Default is False.
            """
        if self.websocket_connection is not None and not self.websocket_connection.open:
            self.connection_open = False
            logger.info(f"Connection was closed and hence opening connection")
            await self.open_connection()

        if not self.connection_open:
            logger.info("Connecting to elevenlabs websocket...")
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": self.temperature,
                    "similarity_boost": self.similarity_boost
                },
                "xi_api_key": self.api_key,
            }
            
            await self.websocket_connection.send(json.dumps(bos_message))
            self.connection_open = True

        if text != "":
            logger.info(f"Sending message {text}")

            input_message = {
                "text": f"{text} ",
                "try_trigger_generation": True,
                "flush": True
            }
            await self.websocket_connection.send(json.dumps(input_message))
            if end_of_llm_stream:
                self.last_text_sent = True

            # self.connection_open = False

    async def receiver(self):
        """
        Asynchronous method that receives audio data from the websocket connection.

        Yields:
            bytes: Audio data chunks received from the websocket connection.

        Raises:
            ConnectionClosed: If the websocket connection is closed unexpectedly.
        """
        # code implementation
        while True:
            if not self.connection_open:
                logger.info("Since eleven labs always closes the connection after every leg, simply open it...")
                await self.open_connection()
            try:
                response = await self.websocket_connection.recv()
                data = json.loads(response)
                logger.info("response for isFinal: {}".format(data.get('isFinal', False)))
                if "audio" in data and data["audio"]:
                    chunk = base64.b64decode(data["audio"])
                    if len(chunk) % 2 == 1:
                        chunk += b'\x00'
                    # @TODO make it better - for example sample rate changing for mp3 and other formats  
                    yield chunk

                    if "isFinal" in data and data["isFinal"]:
                        self.connection_open = False
                        yield b'\x00'
                else:
                    logger.info("No audio data in the response")
            except websockets.exceptions.ConnectionClosed:
                break

    async def __send_payload(self, payload, format=None):
        """
        Sends a payload to the API endpoint.

        Args:
            payload: The payload to be sent as JSON.
            format (optional): The format of the audio file. Defaults to None.

        Returns:
            The response data from the API.

        Raises:
            aiohttp.ClientError: If there is an error in the HTTP request.
        """
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
        """
        Synthesizes the given text into audio.

        Parameters:
        - text (str): The text to be synthesized.

        Returns:
        - audio (bytes): The synthesized audio in MP3 format with a sample rate of 44100 Hz and a bitrate of 128 kbps.
        """
        audio = await self.__generate_http(text, format="mp3_44100_128")
        return audio

    async def __generate_http(self, text, format=None):
        """
        Generates HTTP request payload for text synthesis.

        Args:
            text (str): The input text to be synthesized.
            format (str, optional): The format of the synthesized output. Defaults to None.

        Returns:
            dict: The response payload from the HTTP request.
        """
        payload = None
        logger.info(f"text {text}")
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.temperature,
                "similarity_boost": self.similarity_boost,
                "optimize_streaming_latency": 3
            }
        }
        response = await self.__send_payload(payload, format=format)
        return response

    def get_synthesized_characters(self):
        """
        Returns the synthesized characters.

        Returns:
            str: The synthesized characters.
        """
        return self.synthesized_characters

    # Currently we are only supporting wav output but soon we will incorporate conver
    async def generate(self):
        """
        Generates audio data for streaming.
        Yields:
            bytes: Audio data packets for streaming.
        """
            # code omitted for brevity
        try:
            if self.stream:
                async for message in self.receiver():
                    logger.info(f"Received message from server")

                    if len(self.text_queue) > 0:
                        self.meta_info = self.text_queue.popleft()
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
                    logger.info(f"Generating TTS response for message: {message}, using mulaw {self.use_mulaw}")
                    meta_info, text = message.get("meta_info"), message.get("data")
                    audio = None
                    if self.caching:
                        if self.cache.get(text):
                            logger.info(f"Cache hit and hence returning quickly {text}")
                            audio = self.cache.get(text)
                            meta_info['is_cached'] = True
                        else:
                            c = len(text)
                            self.synthesized_characters += c
                            logger.info(f"Not a cache hit {list(self.cache.data_dict)} and hence increasing characters by {c}")
                            meta_info['is_cached'] = False
                            audio = await self.__generate_http(text)
                            self.cache.set(text, audio)
                    else:
                        meta_info['is_cached'] = False
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
                    else:
                        meta_info['format'] = "wav"
                        wav_bytes = convert_audio_to_wav(audio, source_format="mp3")
                        logger.info(f"self.sampling_rate {self.sampling_rate}")
                        audio = resample(wav_bytes, int(self.sampling_rate), format="wav")
                    yield create_ws_data_packet(audio, meta_info)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in eleven labs generate {e}")

    async def open_connection(self):
        """
        Opens a websocket connection to the server.

        Returns:
            None

        Raises:
            None
        """
        if self.websocket_connection is None or self.connection_open is False:
            self.websocket_connection = await websockets.connect(self.ws_url)
            logger.info("Connected to the server")

    async def push(self, message):
        """
        Pushes a message to the internal queue.

        Parameters:
        - message: The message to be pushed to the queue.

        Returns:
        None
        """
        logger.info(f"Pushed message to internal queue {message}")
        if self.stream:
            meta_info, text = message.get("meta_info"), message.get("data")
            end_of_llm_stream = "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]
            self.meta_info = copy.deepcopy(meta_info)
            meta_info["text"] = text
            self.sender_task = asyncio.create_task(self.sender(text, end_of_llm_stream))
            self.text_queue.append(meta_info)
        else:
            self.internal_queue.put_nowait(message)
