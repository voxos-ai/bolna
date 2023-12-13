from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List
import sys, os
import uuid
import redis.asyncio as redis
from dotenv import load_dotenv
from agents.agent_manager import AssistantManager
import traceback
import json
from agents.helpers.logger_config import configure_logger
import asyncio
from agents.models import *


logger = configure_logger(__name__)

load_dotenv()
redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
active_websockets: List[WebSocket] = []
app = FastAPI()


try:
    logger.info(sys.version)
    logger.info(f"Nogil python state {sys.flags.nogil}")
except Exception as e:
    logger.info("Nogil python not available, hence threads will run with GIL:")


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
    sequences: List[List[str]]


class TaskConfigModel(BaseModel):
    tools_config: ToolsConfigModel
    toolchain: ToolsChainModel


class AssistantModel(BaseModel):
    assistant_name: str
    tasks: List[TaskConfigModel]


def create_agent(agent_name, tasks):
    # TODO Persist agent
    agent_config = {
        "assistant_name": agent_name,
        "tasks": tasks
    }
    return agent_config


@app.on_event("startup")
async def startup_event():
    pass


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.aclose()
    pass


@app.post("/create_agent")
async def create_agent(agent_data: AssistantModel):
    agent_uuid = '{}'.format(str(uuid.uuid4()))
    redis_task = asyncio.create_task(redis_client.set(agent_uuid, agent_data.json()))
    await asyncio.gather(redis_task)

    return {"agent_id": "{}".format(agent_uuid), "state": "created"}


@app.websocket("/chat/v1/{user_id}/{agent_id}")
async def websocket_endpoint(agent_id: str, user_id: str, websocket: WebSocket):
    logger.info("Connected to ws")
    await websocket.accept()
    active_websockets.append(websocket)

    agent_config, context_data = None, None
    try:
        retrieved_agent_config, retrieved_context_data = await redis_client.mget([agent_id, user_id])
        agent_config, context_data = json.loads(retrieved_agent_config), json.loads(retrieved_context_data)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_manager = AssistantManager(agent_config, websocket, context_data, user_id, agent_id)

    try:
        await agent_manager.run()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")
