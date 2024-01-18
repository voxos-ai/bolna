import bolna
from bolna.assistant import Assistant
from bolna.synthesizer import ElevenlabsSynthesizer, XTTSSynthesizer, PollySynthesizer
from bolna.transcriber import DeepgramTranscriber
from bolna.models import *
from bolna.input_handlers import DefaultInputHandler
from bolna.output_handlers import DefaultOutputHandler
from bolna.llms import OpenAiLLM
import asyncio 
bolna.setenv({
    "OPENAI_API_KEY" : "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",
    "DEEPGRAM_AUTH_TOKEN": "b7aea2d67f846f69ef3e957c45e2278702b2cff2",
    "TWILIO_AUTH_TOKEN": "AC1f3285e7c353c7d4036544f8dac36b98",
    "TWILIO_ACCOUNT_SID": "4b67a9be82e6ea5a5853da561471fc3c",
    "TWILIO_ACCOUNT_SID": "+16507638870",
    "ELEVENLABS_API_KEY": "5bb83d503e06369aff83abb071ec0f89",
    "CHECK_FOR_COMPLETION_LLM" : "gpt-3.5-turbo-1106",
    "TTS_API_URL": "http://localhost:8000/",
    "TTS_WS_URL": "ws://localhost:8000/"
})

async def play_audio():
    while True:
        try:
            global chunks
            if len(chunks) > 0:
                chunk = chunks.pop(0)
                if connection_type == "e2e":
                    audio = base64.b64decode(chunk)
                else:
                    audio = chunk
                print(f"Adjusted audio length: {len(audio)}")  
                if len(audio) % 2 != 0:
                    print(f"Audio chunk length is odd: {len(audio)}")
                    audio = audio[:-1]
                    
                print(f"Adjusted audio length: {len(audio)}")  
                audio_data = np.frombuffer(audio, dtype=np.int16)
                print("got audio data")
                sd.play(audio_data, 16000)
                sd.wait()
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in playing audio: {e}")

async def emmiter():
    while True:
        user_input = input("Enter text: ")
        # push api
        await asyncio.sleep(1.5)

async def receiver(DefaultOutputHandler):
    async for message in assistant.execute():
        print(message)


async def main():
    polly_config = PollyConfig(voice = "Kajal", engine = "neural", sampling_rate = "16000", language = "en-US")
    synthesier = Synthesizer(provider = "polly", provider_config = polly_config, stream = False)
    transcriber = Transcriber(model = "deepgram", language ="hi", stream = True, )
    llm_agent = LLM(streaming_model = "gpt-3.5-turbo-16k", max_tokens = 100, agent_flow_type = "streaming", use_fallback = False, family = "openai", temperature = 0.1, prompt = "Helpful assistant")
    assistant = Assistant(name = "trial_agent")
    assistant.add_task("conversation", llm_agent = llm_agent, input_handler = DefaultInputHandler, output_handler = DefaultOutputHandler, transcriber = transcriber, synthesizer = synthesier)
    await asyncio.gather(emitter(), receiver())

asyncio.run(main())