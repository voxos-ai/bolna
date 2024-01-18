import os
import asyncio
import uuid
import traceback
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import redis.asyncio as redis
from dotenv import load_dotenv
from bolna.helpers.analytics_helpers import calculate_total_cost_of_llm_from_transcript, update_high_level_assistant_analytics_data
from bolna.helpers.utils import get_md5_hash, get_s3_file, store_file, format_messages, load_file, create_ws_data_packet
from bolna.providers import *
from bolna.prompts import *
from bolna.helpers.logger_config import configure_logger
from bolna.models import *
#from bolna.memory.cache import InmemoryScalarCache
from bolna.llms import LiteLLM
from litellm import token_counter
from datetime import datetime, timezone
from models import VoiceRequestModel, CreateUserModel, AddVoiceModel, VoiceRequestModel, DEFAULT_VOICES, DEFAULT_LLM_MODELS
from bolna.agent_manager.assistant_manager import AssistantManager
import base64

load_dotenv()
logger = configure_logger(__name__)

redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
#cache = InmemoryScalarCache(ttl = 3600)
active_websockets: List[WebSocket] = []

app = FastAPI()
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.on_event("startup")
async def startup_event():
    pass


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.aclose()
    pass


#TODO Get agent data from redis 
@app.get("/assistant/prompts")
async def get_files(user_id: str, assistant_id: str):
    try:
        existing_data = await redis_client.mget(assistant_id)
        if not existing_data:
            raise HTTPException(status_code=404, detail="Agent not found")

        logger.info(f"Found existing assistant {existing_data} ")
        object_key = f'{user_id}/{assistant_id}/deserialized_prompts.json'
        file_content = await get_s3_file(BUCKET_NAME, object_key)
        deserialized_json = json.loads(file_content)
        return {
            "data": deserialized_json
        }
    except HTTPException as e:
        return e.detail


async def generate_audio_from_text(text, synthesizer_config):
    synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(synthesizer_config['provider'])
    logger.info(f'Provider config {synthesizer_config["provider_config"]} synthesier class {synthesizer_class}')
    synth_instance = synthesizer_class(stream = False, **synthesizer_config["provider_config"])
    audio_data = await synth_instance.synthesize(text)
    return audio_data


async def process_and_store_audio(conversation_graph, user_id, assistant_id, synthesizer_config):
    logger.info(f"Generating and storing data in S3 for the conversation graph {conversation_graph}")
    for node_key, node_value in conversation_graph['task_1'].items():
        for content in node_value['content']:
            if 'text' in content:
                text = content['text']
                file_name = get_md5_hash(content['text'])
                content['audio'] = file_name
                audio_data = await generate_audio_from_text(text, synthesizer_config)
                await store_file(BUCKET_NAME, f"{user_id}/{assistant_id}/audio/{file_name}.{synthesizer_config['audio_format']}", audio_data, f"{synthesizer_config['audio_format']}")
    await store_file(BUCKET_NAME, f"{user_id}/{assistant_id}/conversation_details.json", conversation_graph, "json", local = True)


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
    if tasks is not None and len(tasks) > 0:
        logger.info(f"Now creating prompts for follow up tasks")
        # make changes here
        prompt_json = get_follow_up_prompts(prompt_json, tasks[1:])
        return prompt_json


async def background_process_and_store(conversation_type, assistant_id, assistant_prompts ,synthesizer_config = None,):
    try:
        # For loop and add multiple prompts into the prompt file from here.
        if conversation_type == "preprocessed":
            conversation_graph = prompt_json['conversation_graph']
            logger.info(f"Preprocessed conversation. Storing required files to s3. {synthesizer_config}")
            prompt_json = create_prompts_for_followup_tasks(tasks, prompt_json)
            await process_and_store_audio(prompt_json, user_id, assistant_id, synthesizer_config)
            logger.info("Now storing nodes and graph")
        else:
            object_key = f'{assistant_id}/conversation_details.json'
            await store_file(bucket_name = None, file_key = object_key, file_data = assistant_prompts, content_type = 'json', local = True)            
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error while storing conversation details to s3: {e}")

@app.post("/assistant")
async def create_agent(agent_data: CreateAssistantPayload):
    agent_uuid = str(uuid.uuid4())
    data_for_db = agent_data.assistant_config.dict()
    data_for_db["assistant_status"] = "seeding"
    assistant_prompts = agent_data.assistant_prompts
    logger.info(f'Data for DB {data_for_db}')

    if len(data_for_db['tasks']) > 0:
        logger.info("Setting up follow up tasks")
        for index, task in enumerate(data_for_db['tasks']):
            if task['task_type'] == "extraction":
                extraction_prompt_llm =  os.getenv("EXTRACTION_PROMPT_GENERATION_MODEL")
                extraction_prompt_generation_llm = LiteLLM(streaming_model = extraction_prompt_llm, max_tokens = 2000)
                extraction_prompt = await extraction_prompt_generation_llm.generate(messages = [{'role':'system', 'content': EXTRACTION_PROMPT_GENERATION_PROMPT}, {'role': 'user', 'content': data_for_db["tasks"][index]['tools_config']["llm_agent"]['extraction_details']}])
                data_for_db["tasks"][index]["tools_config"]["llm_agent"]['extraction_json'] = extraction_prompt

    redis_task = asyncio.create_task(redis_client.set(agent_uuid, json.dumps(data_for_db)))
    background_process_and_store_task = asyncio.create_task(
        background_process_and_store(data_for_db['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'], 
                                agent_uuid, agent_data.assistant_prompts, synthesizer_config = data_for_db['tasks'][0]['tools_config']['synthesizer']))

    return {"agent_id": agent_uuid, "state": "created"}


@app.put("/assistant/{agent_uuid}")
async def edit_agent(agent_uuid: str, agent_data: CreateAssistantPayload):
    user_id = agent_data.user_id
    existing_data = await redis_client.mget(agent_uuid)
    if not existing_data:
        raise HTTPException(status_code=404, detail="Agent not found")

    updated_data = agent_data.assistant_config.dict()
    updated_data["assistant_status"] = "updating"
    assistant_prompts = agent_data.assistant_prompts

    if len(updated_data['tasks']) > 0:
        logger.info("Setting up follow up tasks")
        for index, task in enumerate(updated_data['tasks']):
            if task['task_type'] == "extraction":
                extraction_prompt_llm =  os.getenv("EXTRACTION_PROMPT_GENERATION_MODEL")
                extraction_prompt_generation_llm = LiteLLM(streaming_model = extraction_prompt_llm, max_tokens = 2000)
                extraction_prompt = await extraction_prompt_generation_llm.generate(messages = [{'role':'system', 'content': EXTRACTION_PROMPT_GENERATION_PROMPT}, {'role': 'user', 'content': updated_data["tasks"][index]['tools_config']["llm_agent"]['extraction_details']}])
                updated_data["tasks"][index]["tools_config"]["llm_agent"]['extraction_json'] = extraction_prompt

                logger.info(f"Extraction task. Hencegot the extraction prompt {extraction_prompt}")
    background_process_and_store_task = asyncio.create_task(
        background_process_and_store(updated_data['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'],
                                     assistant_prompts.dict(), agent_uuid, user_id, synthesizer_config = updated_data['tasks'][0]['tools_config']['synthesizer'], tasks=updated_data['tasks']))
    redis_task = await redis_client.set(agent_uuid, json.dumps(updated_data))

    return {"agent_id": agent_uuid, "state": "updated"}

############################################################################################# 
# Websocket 
#############################################################################################
@app.websocket("/chat/v1/{agent_id}")
async def websocket_endpoint(agent_id: str, websocket: WebSocket, user_agent: str = Query(None)):
    logger.info("Connected to ws")
    await websocket.accept()
    active_websockets.append(websocket)
    agent_config, context_data = None, None
    try:
        retrieved_agent_config = await redis_client.get(agent_id)
        logger.info(
            f"Retrieved agent config: {retrieved_agent_config}")
        agent_config = json.loads(retrieved_agent_config)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail="Agent not found")

    assistant_manager = AssistantManager(agent_config, websocket, agent_id)
  
    try:
        async for index, task_output in assistant_manager.run(local = True):
            logger.info(task_output)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")