from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Optional
from pydantic import BaseModel, Field, validator, ValidationError, Json
from typing import List
import sys, os
import uuid
import redis.asyncio as redis
import requests
from dotenv import load_dotenv
from agents.agent_manager import AssistantManager
import traceback
import json
from agents.helpers.logger_config import configure_logger
from agents.database.dynamodb import DynamoDB
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from agents.helpers.utils import get_md5_hash, execute_tasks_in_chunks, has_placeholders
from agents.models import *
from agents.prompts import *

logger = configure_logger(__name__, True)

load_dotenv()
redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
active_websockets: List[WebSocket] = []
app = FastAPI()
s3_client = boto3.client('s3') #Using blocking client but it's okay as we're only using it in a specific use case
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://app.bolna.dev"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
dynamodb = DynamoDB(os.getenv('TABLE_NAME'))

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
        return validate_attribute(value, ["en", "hi", "es", "fr", "pt", "ko", "ja", "zh", "de", "it"])


class SynthesizerModel(BaseModel):
    model: str
    language: Optional[str] = None
    voice: str
    stream: bool = False
    buffer_size: Optional[int] = 40  # 40 characters in a buffer
    audio_format: Optional[str] = "mp3"
    sampling_rate: Optional[str] = "24000"
    
    # vocoder: Optional[str]
    # model_path: Optional[str]
    @validator("model")
    def validate_model(cls, value):
        return validate_attribute(value, ["polly", "xtts"])

    @validator("language")
    def validate_language(cls, value):
        return validate_attribute(value, ["en", "hi", "es", "fr"])


class IOModel(BaseModel):
    provider: str
    format: str

    @validator("provider")
    def validate_provider(cls, value):
        return validate_attribute(value, ["twilio", "default", "database"])


class LLM_Model(BaseModel):
    streaming_model: Optional[str] = "gpt-3.5-turbo-16k"
    classification_model: Optional[str] = "gpt-4"
    max_tokens: Optional[int]
    agent_flow_type: Optional[str] = "streaming"
    use_fallback: Optional[bool] = False
    family: Optional[str] = "openai"
    request_json: Optional[bool] = False
    langchain_agent: Optional[bool] = False
    extraction_details: Optional[str] = None #This is the english explaination for the same 
    extraction_json: Optional[str]= None #This is the json required for the same 

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
    tools_config: Optional[ToolModel]


class ToolsChainModel(BaseModel):
    execution: str = Field(..., regex="^(parallel|sequential)$")
    pipelines: List[List[str]]


class TaskConfigModel(BaseModel):
    tools_config: ToolsConfigModel
    toolchain: ToolsChainModel
    task_type: Optional[str] = "conversation" #extraction, summarization, notification



class AssistantModel(BaseModel):
    assistant_name: str
    assistant_type: str = "other"
    tasks: List[TaskConfigModel]

class AssistantPromptsModel(BaseModel):
    deserialized_prompts: Optional[Json]
    serialized_prompts: Optional[Json]
    conversation_graph: Optional[Json]

class CreateAssistantPayload(BaseModel):
    user_id: str
    assistant_config: AssistantModel
    assistant_prompts: AssistantPromptsModel


@app.on_event("startup")
async def startup_event():
    pass


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.aclose()
    pass


def get_s3_file(bucket_name, file_key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        return json.loads(file_content)
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/prompts")
async def get_files(user_id: str, assistant_id: str):
    try:
        existing_data = await dynamodb.get_agent_data(user_id, f"agent#{assistant_id}")
        if not existing_data:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        logger.info(f"Found existing assistant {existing_data} ")
        object_key = f'{user_id}/{assistant_id}/deserialized_prompts.json'
        deserialized_json = get_s3_file(BUCKET_NAME, object_key)
        return {
            "data" : deserialized_json
        }

        
    except HTTPException as e:
        return e.detail


async def generate_audio_from_text(model_name, text):
    #A simple polly or eleven labs call for now
    synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get("polly")
    synth_instance = synthesizer_class('polly', 'mp3', 'Kajal', 'en', '24000')
    async for audio_data in synth_instance.generate_tts_response(text):
        return audio_data

def save_to_s3(object_key, data, content_type):
    try:
        if content_type == "json":
            s3_client.put_object(Bucket=BUCKET_NAME, Key=object_key, Body=json.dumps(data))
        elif content_type == "mp3":
            s3_client.put_object(Bucket=BUCKET_NAME, Key=object_key, Body= data)

        logger.info(f"Object saved to S3 with key {object_key} {content_type}")
    except Exception as e:
        logger.error(f"Error while storing audio to s3: {e}")


async def process_and_store_audio(conversation_graph, model_name, user_id, assistant_id):
    logger.info(f"Generating and storing data in S3 for the conversation graph {conversation_graph}")
    for node_key, node_value in conversation_graph['task_1'].items():
        for content in node_value['content']:
            if 'text' in content:
                text = content['text']
                file_name = get_md5_hash(content['text'])
                content['audio'] = file_name
                audio_data = await generate_audio_from_text(model_name, text)
                save_to_s3(f"{user_id}/{assistant_id}/audio/{file_name}.mp3", audio_data, "mp3")
    save_to_s3(f"{user_id}/{assistant_id}/conversation_details.json", conversation_graph, "json")

def get_follow_up_prompts(prompt_json, tasks):
    for ind, task in enumerate(tasks):
        task_id = ind
        task_type = task.get('task_type')

        if task_type == 'extraction':
            extraction_json = task.get("tools_config").get('llm_agent').get('extraction_json')
            prompt = EXTRACTION_PROMPT.format(extraction_json)
            prompt_json['serialized_prompts'][f'task_{task_id + 2}'] = {"system_prompt": prompt}
        elif task_type == 'summarization':
            prompt_json['serialized_prompts'][f'task_{task_id + 2}'] = {"system_prompt": SUMMARIZATION_PROMPT}
    logger.info(f"returning {prompt_json}")
    return prompt_json
    
def create_prompts_for_followup_tasks(tasks, prompt_json):
    if tasks is not None and len(tasks)>0:
        logger.info(f"Now creating prompts for follow up tasks")
        #make changes here
        prompt_json = get_follow_up_prompts(prompt_json, tasks[1:])
        return prompt_json
async def background_process_and_store(conversation_type, prompt_json, assistant_id, user_id, model_name = "polly", tasks = None):
    try:

        # For loop and add multiple prompts into the prompt file from here.
        if conversation_type == "preprocessed":
            logger.info(f"Prompt json {prompt_json}")
            conversation_graph = prompt_json['conversation_graph']
            logger.info(f"Preprocessed conversation. Storing required files to s3")
            prompt_json = create_prompts_for_followup_tasks(tasks, prompt_json)
            await process_and_store_audio(prompt_json['serialized_prompts'], model_name, user_id, assistant_id)
            logger.info("Now storing nodes and graph")

            save_to_s3(f'{user_id}/{assistant_id}/deserialized_prompts.json', prompt_json["deserialized_prompts"], "json") 

        else:        
            #TODO Write this as an experienced developer who will do both concurrently in a non blocking format
            object_key = f'{user_id}/{assistant_id}/conversation_details.json'
            logger.info(f"tasks {tasks}")
            prompt_json = create_prompts_for_followup_tasks(tasks, prompt_json)
            s3_client.put_object(Bucket=BUCKET_NAME, Key=object_key, Body=json.dumps(prompt_json['serialized_prompts']))
            logger.info(f"Now dumping deserialised prompts")
            object_key = f'{user_id}/{assistant_id}/deserialized_prompts.json'
            s3_client.put_object(Bucket=BUCKET_NAME, Key=object_key, Body=json.dumps(prompt_json['deserialized_prompts']))
    
        await dynamodb.update_agent_status(user_id, f"agent#{assistant_id}", "ready")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error while storing conversation details to s3: {e}")
        await dynamodb.update_agent_status(user_id, f"agent#{assistant_id}", "error")

@app.post("/assistant")
async def create_agent(agent_data: CreateAssistantPayload):
    agent_uuid = str(uuid.uuid4())
    user_id = agent_data.user_id

    data_for_db = agent_data.assistant_config.dict()
    data_for_db["assistant_status"] = "seeding"
    assistant_prompts = agent_data.assistant_prompts
    logger.info(f'Data for DB {data_for_db}')
    redis_task = asyncio.create_task(redis_client.set(agent_uuid, json.dumps(data_for_db)))
    dynamodb_task = asyncio.create_task(dynamodb.store_agent(user_id, f"agent#{agent_uuid}", data_for_db))
    background_process_and_store_task = asyncio.create_task(background_process_and_store(data_for_db['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'], assistant_prompts.dict(), agent_uuid, user_id, tasks =data_for_db['tasks']))
    await asyncio.gather(redis_task, dynamodb_task)


    return {"agent_id": agent_uuid, "state": "created"}


@app.put("/assistant/{agent_uuid}")
async def edit_agent(agent_uuid: str, agent_data: CreateAssistantPayload):
    user_id = agent_data.user_id
    existing_data = await dynamodb.get_agent_data(user_id, f"agent#{agent_uuid}")
    if not existing_data:
        raise HTTPException(status_code=404, detail="Agent not found")

    updated_data = agent_data.assistant_config.dict()
    updated_data["assistant_status"] = "updating"
    assistant_prompts = agent_data.assistant_prompts

    redis_task = asyncio.create_task(redis_client.set(agent_uuid, json.dumps(updated_data)))
    dynamodb_task = asyncio.create_task(dynamodb.store_agent(user_id, f"agent#{agent_uuid}", updated_data))
    background_process_and_store_task = asyncio.create_task(background_process_and_store(updated_data['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'], assistant_prompts.dict(), agent_uuid, user_id, tasks =updated_data['tasks']))
    
    await asyncio.gather(redis_task, dynamodb_task)

    return {"agent_id": agent_uuid, "state": "updated"}


@app.get("/assistants")
async def get_all_agents(user_id: str = Query(..., description="The user ID")):
    try:
        agents = await dynamodb.get_all_agents_for_user(user_id)
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_agent")
async def create_agent(agent_data: AssistantModel):
    agent_uuid = '{}'.format(str(uuid.uuid4()))
    user_id = get_user_id()
    redis_task = asyncio.create_task(redis_client.set(agent_uuid, agent_data.json()))
    dynamodb_task = asyncio.create_task(dynamodb.store_agent(user_id, f"agent#{agent_uuid}",json.loads(agent_data.json())))
    await asyncio.gather(redis_task, dynamodb_task)
    # response = requests.get('http://twilio-app:8001/make_call?agent_uuid={}'.format(agent_uuid))

    return {"agent_id": "{}".format(agent_uuid), "state": "created"}


@app.websocket("/chat/v1/{user_id}/{agent_id}")
async def websocket_endpoint(agent_id: str, user_id: str, websocket: WebSocket, user_agent: str = Query(None)):
    logger.info("Connected to ws")
    await websocket.accept()
    active_websockets.append(websocket)
    connected_through_dashboard = True if user_agent == "dashboard" else False

    agent_config, context_data = None, None
    try:
        retrieved_agent_config, retrieved_context_data = await redis_client.mget([agent_id, user_id])
        logger.info(f"Retrieved agent config: {retrieved_agent_config}, retrieved_context_data {retrieved_context_data}")
        agent_config, context_data = json.loads(retrieved_agent_config), json.loads(retrieved_context_data) if retrieved_context_data is not None else None
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_manager = AssistantManager(agent_config, websocket, context_data, user_id, agent_id, connected_through_dashboard = connected_through_dashboard)

    try:
        await agent_manager.run()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")