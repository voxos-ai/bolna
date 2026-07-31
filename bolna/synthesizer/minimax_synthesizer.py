import copy
import os

import aiohttp
from dotenv import load_dotenv

from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes
from bolna.memory.cache.inmemory_scalar_cache import InmemoryScalarCache
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()

MINIMAX_TTS_ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/t2a_v2",
    "cn_zh": "https://api.minimaxi.com/v1/t2a_v2",
}
MINIMAX_TTS_MODELS = {
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
}
MINIMAX_AUDIO_FORMATS = {"mp3", "wav", "flac", "pcm"}


class MiniMaxSynthesizer(BaseSynthesizer):
    def __init__(self, voice, region, model="speech-2.8-hd", audio_format="pcm",
                 sampling_rate=24000, stream=False, buffer_size=400, caching=True,
                 language_boost=None, pronunciation_dict=None, voice_setting=None,
                 audio_setting=None, voice_modify=None, subtitle_enable=None, **kwargs):
        super().__init__(stream, buffer_size)
        if region not in MINIMAX_TTS_ENDPOINTS:
            raise ValueError("Unsupported MiniMax TTS region: {}".format(region))
        if model not in MINIMAX_TTS_MODELS:
            raise ValueError("Unsupported MiniMax TTS model: {}".format(model))

        audio_format = audio_format.lower()
        if audio_format not in MINIMAX_AUDIO_FORMATS:
            raise ValueError("Unsupported MiniMax audio format: {}".format(audio_format))

        self.voice = voice
        self.region = region
        self.model = model
        self.format = audio_format
        self.sample_rate = int(sampling_rate)
        self.url = MINIMAX_TTS_ENDPOINTS[region]
        self.api_key = kwargs.get("synthesizer_key") or os.getenv("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError("MiniMax API key is required")

        self.language_boost = language_boost
        self.pronunciation_dict = pronunciation_dict
        self.voice_setting = dict(voice_setting or {})
        self.voice_setting["voice_id"] = voice
        self.audio_setting = dict(audio_setting or {})
        self.audio_setting.update({
            "format": self.format,
            "sample_rate": self.sample_rate,
        })
        self.voice_modify = voice_modify
        self.subtitle_enable = subtitle_enable
        self.first_chunk_generated = False
        self.synthesized_characters = 0
        self.caching = caching
        if caching:
            self.cache = InmemoryScalarCache()

    def get_synthesized_characters(self):
        return self.synthesized_characters

    def get_engine(self):
        return self.model

    def supports_websocket(self):
        return False

    def _build_payload(self, text):
        payload = {
            "model": self.model,
            "text": text,
            "stream": False,
            "output_format": "hex",
            "voice_setting": self.voice_setting,
            "audio_setting": self.audio_setting,
        }
        optional_fields = {
            "language_boost": self.language_boost,
            "pronunciation_dict": self.pronunciation_dict,
            "voice_modify": self.voice_modify,
            "subtitle_enable": self.subtitle_enable,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        return payload

    async def _generate_http(self, text):
        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=headers, json=self._build_payload(text)) as response:
                response_body = await response.json(content_type=None)

        base_status = response_body.get("base_resp", {}).get("status_code")
        data = response_body.get("data", {})
        if response.status != 200 or base_status != 0:
            raise RuntimeError(
                "MiniMax speech synthesis failed with HTTP status {} and API status {}".format(
                    response.status, base_status
                )
            )

        audio_hex = data.get("audio")
        if not audio_hex:
            raise RuntimeError(
                "MiniMax speech synthesis returned no audio with data status {}".format(
                    data.get("status")
                )
            )
        try:
            return bytes.fromhex(audio_hex)
        except ValueError as error:
            raise RuntimeError("MiniMax speech synthesis returned invalid hex audio") from error

    def _to_wav(self, audio):
        if self.format == "wav":
            return audio
        if self.format == "pcm":
            return pcm_to_wav_bytes(audio, sample_rate=self.sample_rate)
        return convert_audio_to_wav(audio, source_format=self.format)

    async def synthesize(self, text):
        audio = await self._generate_http(text)
        return self._to_wav(audio)

    async def generate(self):
        while True:
            message = await self.internal_queue.get()
            meta_info, text = message.get("meta_info"), message.get("data")
            audio = self.cache.get(text) if self.caching else None
            if audio is None:
                self.synthesized_characters += len(text)
                audio = await self.synthesize(text)
                if self.caching:
                    self.cache.set(text, audio)

            meta_info["is_first_chunk"] = not self.first_chunk_generated
            self.first_chunk_generated = True
            if meta_info.get("end_of_llm_stream"):
                meta_info["end_of_synthesizer_stream"] = True
                self.first_chunk_generated = False
            meta_info["text"] = text
            meta_info["format"] = "wav"
            yield create_ws_data_packet(audio, meta_info)

    async def open_connection(self):
        pass

    async def push(self, message):
        logger.info("Pushed message to MiniMax synthesizer queue")
        self.internal_queue.put_nowait(copy.deepcopy(message))
