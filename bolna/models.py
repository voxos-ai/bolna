import json
from typing import Optional, List, Union, Dict
from pydantic import BaseModel, Field, validator, ValidationError, Json
from .providers import *


def validate_attribute(value, allowed_values):
    if value not in allowed_values:
        raise ValidationError(f"Invalid provider. Supported values: {', '.join(allowed_values)}")
    return value


class PollyConfig(BaseModel):
    voice: str
    engine: str
    sampling_rate: Optional[str] = "16000"
    language: str


class XTTSConfig(BaseModel):
    voice: str
    language: str
    sampling_rate: Optional[str] ="24000"


class ElevenLabsConfig(BaseModel):
    voice: str
    voice_id: str
    model: str
    sampling_rate: Optional[str] = "16000"


class OpenAIConfig(BaseModel):
    voice: str
    model: str
    sampling_rate: Optional[str] ="24000"


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

    @validator("provider")
    def validate_model(cls, value):
        return validate_attribute(value, ["polly", "xtts", "elevenlabs", "openai", "deepgram"])


class IOModel(BaseModel):
    provider: str
    format: str

    @validator("provider")
    def validate_provider(cls, value):
        return validate_attribute(value, ["twilio", "default", "database", "exotel"])


class LLM(BaseModel):
    streaming_model: Optional[str] = "gpt-3.5-turbo-16k"
    classification_model: Optional[str] = "gpt-4"
    max_tokens: Optional[int] = 100
    agent_flow_type: Optional[str] = "streaming"
    use_fallback: Optional[bool] = False
    family: Optional[str] = "openai"
    temperature: Optional[float] = 0.1
    request_json: Optional[bool] = False
    stop: Optional[List[str]] = None
    top_k: Optional[int] = 0
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.1
    frequency_penalty: Optional[float] = 0.0  
    presence_penalty: Optional[float] = 0.0

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


class Task(BaseModel):
    tools_config: ToolsConfig
    toolchain: ToolsChainModel
    task_type: Optional[str] = "conversation"  # extraction, summarization, notification


class AgentModel(BaseModel):
    agent_name: str
    agent_type: str = "other"
    tasks: List[Task]
 # Usually of the format task_1: { "system_prompt" : "helpful agent" } #For IVR type it should be a basic graph