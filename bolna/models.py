from .synthesizer import PollySynthesizer, XTTSSynthesizer
from .transcriber import DeepgramTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler, DatabaseOutputHandler
from .llms import OpenAiLLM

SUPPORTED_SYNTHESIZER_MODELS = {
    'polly': PollySynthesizer,
    'xtts': XTTSSynthesizer
}
SUPPORTED_TRANSCRIBER_MODELS = {
    'deepgram': DeepgramTranscriber
}
SUPPORTED_LLM_MODELS = {
    'openai': OpenAiLLM
}
SUPPORTED_INPUT_HANDLERS = {
    'default': DefaultInputHandler,
    'twilio': TwilioInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    'default': DefaultOutputHandler,
    'twilio': TwilioOutputHandler,
    'database': DatabaseOutputHandler
}
