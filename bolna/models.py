from typing import Optional
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List
from .providers import *


def validate_attribute(value, allowed_values):
    if value not in allowed_values:
        raise ValidationError(f"Invalid provider. Supported values: {', '.join(allowed_values)}")
    return value


class TranscriberModel(BaseModel):
    model: str
    language: Optional[str] = None
    stream: bool = False

    @validator("model")
    def validate_model(cls, value):
        return validate_attribute(value, ["deepgram"])

    @validator("language")
    def validate_language(cls, value):
        return validate_attribute(value, ["en", "hi"])


class SynthesizerModel(BaseModel):
    model: str
    language: Optional[str] = None
    voice: str
    stream: bool = False
    buffer_size: Optional[int] = 40
    audio_format: Optional[str] = "mp3"
    sampling_rate: Optional[str] = "24000"

    @validator("model")
    def validate_model(cls, value):
        return validate_attribute(value, list(SUPPORTED_SYNTHESIZER_MODELS.keys()))

    @validator("language")
    def validate_language(cls, value):
        return validate_attribute(value, ["en"])


class IOModel(BaseModel):
    provider: str
    format: str

    @validator("provider")
    def validate_provider(cls, value):
        return validate_attribute(value, list(SUPPORTED_INPUT_HANDLERS.keys()))


class LLM_Model(BaseModel):
    streaming_model: Optional[str] = "gpt-3.5-turbo-16k"
    classification_model: Optional[str] = "gpt-4"
    max_tokens: Optional[int]
    agent_flow_type: Optional[str] = "streaming"
    use_fallback: Optional[bool] = False
    family: Optional[str] = "openai"
    agent_task: Optional[str] = "conversation"
    request_json: Optional[bool] = False
    langchain_agent: Optional[bool] = False


class ToolsConfigModel(BaseModel):
    llm_agent: Optional[LLM_Model] = None
    synthesizer: Optional[SynthesizerModel] = None
    transcriber: Optional[TranscriberModel] = None
    input: Optional[IOModel] = None
    output: Optional[IOModel] = None


class ToolsChainModel(BaseModel):
    execution: str = Field(..., regex="^(parallel|sequential)$")
    pipelines: List[List[str]]


class TaskConfigModel(BaseModel):
    tools_config: ToolsConfigModel
    toolchain: ToolsChainModel
    task_type: Optional[str] = "conversation"  # extraction, summarization, notification


class AssistantModel(BaseModel):
    assistant_name: str
    tasks: List[TaskConfigModel]
