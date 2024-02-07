from .synthesizer import PollySynthesizer, XTTSSynthesizer, ElevenlabsSynthesizer, OPENAISynthesizer, FourieSynthesizer
from .transcriber import DeepgramTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler
from .llms import OpenAiLLM, LiteLLM

SUPPORTED_SYNTHESIZER_MODELS = {
    'polly': PollySynthesizer,
    'xtts': XTTSSynthesizer,
    "elevenlabs": ElevenlabsSynthesizer,
    "openai": OPENAISynthesizer,
    "fourie": FourieSynthesizer
}
SUPPORTED_TRANSCRIBER_MODELS = {
    'deepgram': DeepgramTranscriber,
    'whisper': DeepgramTranscriber #Seperate out a transcriber for https://github.com/bolna-ai/streaming-transcriber-server or build a deepgram compatible proxy
}
SUPPORTED_LLM_MODELS = {
    'openai': OpenAiLLM,
    'cohere': LiteLLM,
    'ollama': LiteLLM,
    'mistral': LiteLLM,
    'llama': LiteLLM,
    'zephyr': LiteLLM,
    'perplexity': LiteLLM,
    'vllm': OpenAiLLM
}
SUPPORTED_INPUT_HANDLERS = {
    'default': DefaultInputHandler,
    'twilio': TwilioInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    'default': DefaultOutputHandler,
    'twilio': TwilioOutputHandler
}
