import json
from typing import Optional, List, Union, Dict
from pydantic import BaseModel, Field, validator, ValidationError, Json
from .providers import *

AGENT_WELCOME_MESSAGE = "This call is being recorded for quality assurance and training. Please speak now."


def validate_attribute(value, allowed_values):
    if value not in allowed_values:
        raise ValidationError(f"Invalid provider. Supported values: {', '.join(allowed_values)}")
    return value


class PollyConfig(BaseModel):
    voice: str
    engine: str
    language: str
    # volume: Optional[str] = '0dB'
    # rate: Optional[str] = '100%'


class XTTSConfig(BaseModel):
    voice: str
    language: str


class ElevenLabsConfig(BaseModel):
    voice: str
    voice_id: str
    model: str
    temperature: Optional[float] = 0.5
    similarity_boost: Optional[float] = 0.5


class OpenAIConfig(BaseModel):
    voice: str
    model: str


class FourieConfig(BaseModel):
    voice_id: str
    gender: str
    voice: str


class DeepgramConfig(BaseModel):
    voice: str


class Transcriber(BaseModel):
    model: str
    language: Optional[str] = None
    stream: bool = False
    sampling_rate: Optional[int] = 16000
    encoding: Optional[str] = "linear16"
    endpointing: Optional[int] = 400
    keywords: Optional[str] = None

    @validator("model")
    def validate_model(cls, value):
        return validate_attribute(value, list(SUPPORTED_TRANSCRIBER_MODELS.keys()))

    @validator("language")
    def validate_language(cls, value):
        return validate_attribute(value, ["en", "hi", "es", "fr", "pt", "ko", "ja", "zh", "de", "it", "pt-BR"])


class Synthesizer(BaseModel):
    provider: str
    provider_config: Union[PollyConfig, XTTSConfig, ElevenLabsConfig, OpenAIConfig, FourieConfig, DeepgramConfig]
    stream: bool = False
    buffer_size: Optional[int] = 40  # 40 characters in a buffer
    audio_format: Optional[str] = "pcm"
    caching: Optional[bool] = True

    @validator("provider")
    def validate_model(cls, value):
        return validate_attribute(value, ["polly", "xtts", "elevenlabs", "openai", "deepgram"])


class IOModel(BaseModel):
    provider: str
    format: str

    @validator("provider")
    def validate_provider(cls, value):
        return validate_attribute(value, ["twilio", "default", "database", "exotel"])


# Can be used to route across multiple prompts as well
class Route(BaseModel):
    route_name: str
    utterances: List[str]
    response: Union[List[
        str], str]  # If length of responses is less than utterances, a random sentence will be used as a response and if it's equal, respective index will be used to use it as FAQs caching
    score_threshold: Optional[float] = 0.85  # this is the required threshold for cosine similarity


# Routes can be used for FAQs caching, prompt routing, guard rails, agent assist function calling
class Routes(BaseModel):
    embedding_model: Optional[str] = "Snowflake/snowflake-arctic-embed-l"
    routes: List[Route]


class LLM(BaseModel):
    model: Optional[str] = "gpt-3.5-turbo-16k"
    max_tokens: Optional[int] = 100
    agent_flow_type: Optional[str] = "streaming"
    family: Optional[str] = "openai"
    temperature: Optional[float] = 0.1
    request_json: Optional[bool] = False
    stop: Optional[List[str]] = None
    top_k: Optional[int] = 0
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.1
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    provider: Optional[str] = "openai"
    base_url: Optional[str] = None
    routes: Optional[Routes] = None
    extraction_details: Optional[str] = None
    summarization_details: Optional[str] = None


class MessagingModel(BaseModel):
    provider: str
    template: str


# Need to redefine it
class CalendarModel(BaseModel):
    provider: str
    title: str
    email: str
    time: str


class ToolModel(BaseModel):
    calendar: Optional[CalendarModel] = None
    whatsapp: Optional[MessagingModel] = None
    sms: Optional[MessagingModel] = None
    email: Optional[MessagingModel] = None
    webhookURL: Optional[str] = None


class ToolsConfig(BaseModel):
    llm_agent: Optional[LLM] = None
    synthesizer: Optional[Synthesizer] = None
    transcriber: Optional[Transcriber] = None
    input: Optional[IOModel] = None
    output: Optional[IOModel] = None
    api_tools: Optional[ToolModel] = None


class ToolsChainModel(BaseModel):
    execution: str = Field(..., pattern="^(parallel|sequential)$")
    pipelines: List[List[str]]


class ConversationConfig(BaseModel):
    optimize_latency: Optional[bool] = True  # This will work on in conversation
    hangup_after_silence: Optional[int] = 10
    incremental_delay: Optional[int] = 100  # use this to incrementally delay to handle long pauses
    number_of_words_for_interruption: Optional[
        int] = 1  # Maybe send half second of empty noise if needed for a while as soon as we get speaking true in nitro, use that to delay
    interruption_backoff_period: Optional[int] = 100
    hangup_after_LLMCall: Optional[bool] = False
    call_cancellation_prompt: Optional[str] = None
    backchanneling: Optional[bool] = False
    backchanneling_message_gap: Optional[int] = 5
    backchanneling_start_delay: Optional[int] = 5


class Task(BaseModel):
    tools_config: ToolsConfig
    toolchain: ToolsChainModel
    task_type: Optional[str] = "conversation"  # extraction, summarization, notification
    task_config: ConversationConfig = dict()


class AgentModel(BaseModel):
    agent_name: str
    agent_type: str = "other"
    tasks: List[Task]
    agent_welcome_message: Optional[str] = AGENT_WELCOME_MESSAGE
