from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
import audioop
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_audio_to_wav, create_ws_data_packet, pcm_to_wav_bytes, resample
from .base_synthesizer import BaseSynthesizer

logger = configure_logger(__name__)
load_dotenv()


class PollySynthesizer(BaseSynthesizer):
    def __init__(self, voice, language, audio_format="pcm", sampling_rate=8000, stream=False, engine="neural",
                 buffer_size=400, speaking_rate = "100%", volume = "0dB", cache= None, **kwargs):
        super().__init__(stream, buffer_size)
        self.engine = engine
        self.format = self.get_format(audio_format.lower())
        self.voice = voice
        self.language = language
        self.sample_rate = str(sampling_rate)
        self.client = None
        self.first_chunk_generated = False
        self.speaking_rate = speaking_rate
        self.volume = volume
        self.synthesized_characters = 0
        self.cache = cache

    def get_synthesized_characters(self):
        return self.synthesized_characters
    
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
        self.synthesized_characters += len(text)
        session = AioSession()
        async with AsyncExitStack() as exit_stack:
            polly = await self.create_client("polly", session, exit_stack)
            logger.info(f"Generating TTS response for text: {text}, SampleRate {self.sample_rate} format {self.format}")
            # input = f"""
            # <speak> 
            #     <amazon:auto-breaths volume= "x-loud" frequency="x-high" duration="x-long"> 
            #         <prosody volume="{self.volume}" rate="{self.speaking_rate}"> {text} 
            #         </prosody> 
            #     </amazon:auto-breaths>
            # </speak>
            # """

            logger.info(f"Sending text {input}")
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
            if self.cache.get(text):
                logger.info(f"Cache hit and hence returning quickly {text}")
                message = self.cache[text]
            else:
                logger.info(f"Not a cache hit {list(self.cache.data_dict)}")
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
