from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
from .base_synthesizer import BaseSynthesizer

load_dotenv()


class PollySynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, audio_format="mp3", sampling_rate="22000", stream=False, engine="neural",
                 buffer_size=400, log_dir_name=None):
        super().__init__(stream, buffer_size, log_dir_name)
        self.engine = engine
        self.format = audio_format.lower()
        self.voice = voice
        self.language = language
        self.sample_rate = sampling_rate

        # @TODO: initialize client here
        self.client = None

    @staticmethod
    async def create_client(service: str, session: AioSession, exit_stack: AsyncExitStack):
        # creates AWS session from system environment credentials & config
        return await exit_stack.enter_async_context(session.create_client(service))

    async def generate_tts_response(self, text):
        session = AioSession()

        async with AsyncExitStack() as exit_stack:
            polly = await self.create_client("polly", session, exit_stack)
            self.logger.info(f"Generating TTS response for text: {text}, SampleRate {self.sample_rate}")
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
                self.logger.error(error)
            else:
                yield await response["AudioStream"].read()

    async def generate(self, text):
        self.logger.info('received text for audio generation: {}'.format(text))
        try:
            if text != "" and text != "LLM_END":
                async for message in self.generate_tts_response(text):
                    yield message
        except Exception as e:
            self.logger.error(f"Error in polly generate {e}")
