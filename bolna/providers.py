from .synthesizer import PollySynthesizer, XTTSSynthesizer, ElevenlabsSynthesizer
from .transcriber import DeepgramTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler
from .llms import OpenAiLLM, LiteLLM

SUPPORTED_SYNTHESIZER_MODELS = {
    'polly': PollySynthesizer,
    'xtts': XTTSSynthesizer,
    "elevenlabs": ElevenlabsSynthesizer
}
SUPPORTED_TRANSCRIBER_MODELS = {
    'deepgram': DeepgramTranscriber
}
SUPPORTED_LLM_MODELS = {
    'openai': OpenAiLLM,
    'cohere': LiteLLM,
    'ollama': LiteLLM,
    'mistral': LiteLLM,
    'llama': LiteLLM,
    'zephyr': LiteLLM
}
SUPPORTED_INPUT_HANDLERS = {
    'default': DefaultInputHandler,
    'twilio': TwilioInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    'default': DefaultOutputHandler,
    'twilio': TwilioOutputHandler
}
