import json
from typing import Optional, List, Union
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


class ElevenLabsConfig(BaseModel):
    voice: str
    voice_id: str
    model: str


class TranscriberModel(BaseModel):
    model: str
    language: Optional[str] = None
    stream: bool = False
    sampling_rate: Optional[int] = 16000
    encoding: Optional[str] = "linear16"
    endpointing: Optional[int] = 400

    @validator("model")
    def validate_model(cls, value):
        return validate_attribute(value, list(SUPPORTED_TRANSCRIBER_MODELS.keys()))

    @validator("language")
    def validate_language(cls, value):
        return validate_attribute(value, ["en", "hi", "es", "fr", "pt", "ko", "ja", "zh", "de", "it"])


class SynthesizerModel(BaseModel):
    provider: str
    provider_config: Union[PollyConfig, XTTSConfig, ElevenLabsConfig]
    stream: bool = False
    buffer_size: Optional[int] = 40  # 40 characters in a buffer
    audio_format: Optional[str] = "mp3"

    @validator("provider")
    def validate_model(cls, value):
        return validate_attribute(value, ["polly", "xtts", "elevenlabs"])

class IOModel(BaseModel):
    provider: str
    format: str

    @validator("provider")
    def validate_provider(cls, value):
        return validate_attribute(value, ["twilio", "default", "database"])


class LLM_Model(BaseModel):
    streaming_model: Optional[str] = "gpt-3.5-turbo-16k"
    classification_model: Optional[str] = "gpt-4"
    max_tokens: Optional[int] = 100
    agent_flow_type: Optional[str] = "streaming"
    use_fallback: Optional[bool] = False
    family: Optional[str] = "openai"
    temperature: Optional[float] = 0.1
    request_json: Optional[bool] = False
    langchain_agent: Optional[bool] = False
    extraction_details: Optional[str] = None  # This is the english explaination for the same
    extraction_json: Optional[str] = None  # This is the json required for the same


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


class ToolsConfigModel(BaseModel):
    llm_agent: Optional[LLM_Model] = None
    synthesizer: Optional[SynthesizerModel] = None
    transcriber: Optional[TranscriberModel] = None
    input: Optional[IOModel] = None
    output: Optional[IOModel] = None
    api_tools: Optional[ToolModel] = None


class ToolsChainModel(BaseModel):
    execution: str = Field(..., pattern="^(parallel|sequential)$")
    pipelines: List[List[str]]


class TaskConfigModel(BaseModel):
    tools_config: ToolsConfigModel
    toolchain: ToolsChainModel
    task_type: Optional[str] = "conversation"  # extraction, summarization, notification


class AssistantModel(BaseModel):
    assistant_name: str
    assistant_type: str = "other"
    tasks: List[TaskConfigModel]


class AssistantPromptsModel(BaseModel):
    deserialized_prompts: Optional[Json] = None
    serialized_prompts: Optional[Json] = None
    conversation_graph: Optional[Json] = None


class CreateAssistantPayload(BaseModel):
    user_id: str
    assistant_config: AssistantModel
    assistant_prompts: AssistantPromptsModel
