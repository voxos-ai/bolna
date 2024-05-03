import asyncio
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
import audioop
from bolnaTestVersion.helpers.logger_config import configure_logger
from bolnaTestVersion.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()


class PollySynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, audio_format="pcm", sampling_rate="8000", stream=False, engine="neural",
                 buffer_size=400, **kwargs):
        super().__init__(stream, buffer_size)
        self.engine = engine
        self.format = self.get_format(audio_format.lower())
        self.voice = voice
        self.language = language
        self.sample_rate = str(sampling_rate)
        self.client = None
        self.first_chunk_generated = False

    def get_format(self, audio_format):
        if audio_format == "pcm":
            return "pcm"
        else:
            return "mp3"

    @staticmethod
    async def create_client(service: str, session: AioSession, exit_stack: AsyncExitStack):
        # creates AWS session from system environment credentials & config
        return await exit_stack.enter_async_context(session.create_client(service))

    async def __generate_http(self, text):
        session = AioSession()
        async with AsyncExitStack() as exit_stack:
            polly = await self.create_client("polly", session, exit_stack)
            logger.info(f"Generating TTS response for text: {text}, SampleRate {self.sample_rate} format {self.format}")
            try:
                response = await polly.synthesize_speech(
                    Engine=self.engine,
                    Text=text,
                    OutputFormat=self.format,
                    VoiceId=self.voice,
                    LanguageCode=self.language,
                    SampleRate=self.sample_rate
                )
            except (BotoCoreError, ClientError) as error:
                logger.error(error)
            else:
                return await response["AudioStream"].read()

    async def open_connection(self):
        pass

    async def synthesize(self, text):
        # This is used for one off synthesis mainly for use cases like voice lab and IVR
        audio = await self.__generate_http(text)

        return audio

    async def generate(self):
        while True:
            logger.info("Generating TTS response")
            message = await self.internal_queue.get()
            logger.info(f"Generating TTS response for message: {message}")
            meta_info, text = message.get("meta_info"), message.get("data")
            message = await self.__generate_http(text)
            if self.format == "mp3":
                message = convert_audio_to_wav(message, source_format="mp3")
            if not self.first_chunk_generated:
                meta_info["is_first_chunk"] = True
                self.first_chunk_generated = True
            else:
                meta_info["is_first_chunk"] = False
            if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"]:
                meta_info["end_of_synthesizer_stream"] = True
                self.first_chunk_generated = False
            meta_info['text'] = text
            yield create_ws_data_packet(message, meta_info)

    async def push(self, message):
        logger.info("Pushed message to internal queue")
        self.internal_queue.put_nowait(message)
