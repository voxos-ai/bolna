from .synthesizer import PollySynthesizer, XTTSSynthesizer, ElevenlabsSynthesizer, OPENAISynthesizer, FourieSynthesizer, DeepgramSynthesizer
from .transcriber import DeepgramTranscriber, WhisperTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler, ExotelInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler, ExotelOutputHandler
from .llms import OpenAiLLM, LiteLLM

SUPPORTED_SYNTHESIZER_MODELS = {
    'polly': PollySynthesizer,
    'xtts': XTTSSynthesizer,
    'elevenlabs': ElevenlabsSynthesizer,
    'openai': OPENAISynthesizer,
    'fourie': FourieSynthesizer,
    'deepgram': DeepgramSynthesizer
}
SUPPORTED_TRANSCRIBER_MODELS = {
    'deepgram': DeepgramTranscriber,
    'whisper': WhisperTranscriber #Seperate out a transcriber for https://github.com/bolna-ai/streaming-transcriber-server or build a deepgram compatible proxy
}
SUPPORTED_LLM_MODELS = {
    'openai': OpenAiLLM,
    'cohere': LiteLLM,
    'ollama': LiteLLM,
    'mistral': LiteLLM,
    'llama': LiteLLM,
    'zephyr': LiteLLM,
    'azure-openai': LiteLLM,
    'perplexity': LiteLLM,
    'vllm': OpenAiLLM
}
SUPPORTED_INPUT_HANDLERS = {
    'default': DefaultInputHandler,
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler
}
SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    'default': DefaultOutputHandler,
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler
}
SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler
}
