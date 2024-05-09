from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
from bolnaTestVersion.helpers.logger_config import configure_logger
from  .base import base_synthesizer

logger = configure_logger(__name__)
load_dotenv()


class PollySynthesizer(base_synthesizer):
    def __init__(self, voice, language, audio_format="pcm", sampling_rate="8000", stream=False, engine="neural",
                 buffer_size=400, **kwargs):
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
    async def create_client(service: str, session: AioSession, exit_stack: AsyncExitStack, ):
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
        audio = await self.__generate_http(text)
        return audio

    