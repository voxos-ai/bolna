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
from bolna.helpers.utils import get_md5_hash, get_s3_file, put_s3_file, format_messages, load_file
from bolna.providers import *
from bolna.prompts import *
from bolna.helpers.logger_config import configure_logger
from bolna.models import *
from bolna.llms import LiteLLM
from litellm import token_counter
from datetime import datetime, timezone
from models import VoiceRequestModel, CreateUserModel, AddVoiceModel, VoiceRequestModel, DEFAULT_VOICES, DEFAULT_LLM_MODELS
from bolna.agent_manager.assistant_manager import AssistantManager
from database.dynamodb import DynamoDB
import base64

load_dotenv()
logger = configure_logger(__name__)

redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
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
dynamodb = DynamoDB(os.getenv('TABLE_NAME'))


@app.on_event("startup")
async def startup_event():
    pass


@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.aclose()
    pass


@app.get("/assistant/prompts")
async def get_files(user_id: str, assistant_id: str):
    try:
        existing_data = await dynamodb.get_assistant_data(user_id, f"agent#{assistant_id}")
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
    async for audio_data in synth_instance.generate(text):
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
                await put_s3_file(BUCKET_NAME, f"{user_id}/{assistant_id}/audio/{file_name}.{synthesizer_config['audio_format']}", audio_data, f"{synthesizer_config['audio_format']}")
    await put_s3_file(BUCKET_NAME, f"{user_id}/{assistant_id}/conversation_details.json", conversation_graph, "json")


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


async def background_process_and_store(conversation_type, prompt_json, assistant_id, user_id, synthesizer_config = None,
                                       tasks=None):
    try:

        # For loop and add multiple prompts into the prompt file from here.
        if conversation_type == "preprocessed":
            logger.info(f"Prompt json {prompt_json}")
            conversation_graph = prompt_json['conversation_graph']
            logger.info(f"Preprocessed conversation. Storing required files to s3. {synthesizer_config}")
            prompt_json = create_prompts_for_followup_tasks(tasks, prompt_json)
            await process_and_store_audio(prompt_json['serialized_prompts'], user_id, assistant_id, synthesizer_config)
            logger.info("Now storing nodes and graph")

            await put_s3_file(BUCKET_NAME, f'{user_id}/{assistant_id}/deserialized_prompts.json',
                              prompt_json["deserialized_prompts"], "json")

        else:
            # TODO Write this as an experienced developer who will do both concurrently in a non blocking format
            object_key = f'{user_id}/{assistant_id}/conversation_details.json'
            logger.info(f"tasks {tasks}")
            prompt_json = create_prompts_for_followup_tasks(tasks, prompt_json)
            await put_s3_file(BUCKET_NAME, object_key, prompt_json['serialized_prompts'], 'json')
            logger.info(f"Now dumping deserialised prompts")
            object_key = f'{user_id}/{assistant_id}/deserialized_prompts.json'
            await put_s3_file(BUCKET_NAME, object_key, prompt_json['deserialized_prompts'], 'json')

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

    if len(data_for_db['tasks']) > 0:
        logger.info("Setting up follow up tasks")
        for index, task in enumerate(data_for_db['tasks']):
            if task['task_type'] == "extraction":
                extraction_prompt_llm =  os.getenv("EXTRACTION_PROMPT_GENERATION_MODEL")
                extraction_prompt_generation_llm = LiteLLM(streaming_model = extraction_prompt_llm, max_tokens = 2000)
                extraction_prompt = await extraction_prompt_generation_llm.generate(messages = [{'role':'system', 'content': EXTRACTION_PROMPT_GENERATION_PROMPT}, {'role': 'user', 'content': data_for_db["tasks"][index]['tools_config']["llm_agent"]['extraction_details']}])
                data_for_db["tasks"][index]["tools_config"]["llm_agent"]['extraction_json'] = extraction_prompt

    redis_task = asyncio.create_task(redis_client.set(agent_uuid, json.dumps(data_for_db)))
    dynamodb_task = asyncio.create_task(dynamodb.store_agent_data(user_id, f"agent#{agent_uuid}", data_for_db))
    background_process_and_store_task = asyncio.create_task(
        background_process_and_store(data_for_db['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'],
                                     assistant_prompts.dict(), agent_uuid, user_id, synthesizer_config = data_for_db['tasks'][0]['tools_config']['synthesizer'] , tasks=data_for_db['tasks']))
    await asyncio.gather(redis_task, dynamodb_task)

    return {"agent_id": agent_uuid, "state": "created"}


@app.put("/assistant/{agent_uuid}")
async def edit_agent(agent_uuid: str, agent_data: CreateAssistantPayload):
    user_id = agent_data.user_id
    existing_data = await dynamodb.get_assistant_data(user_id, f"agent#{agent_uuid}")
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
    redis_task = asyncio.create_task(redis_client.set(agent_uuid, json.dumps(updated_data)))
    dynamodb_task = asyncio.create_task(dynamodb.store_agent_data(user_id, f"agent#{agent_uuid}", updated_data))
    background_process_and_store_task = asyncio.create_task(
        background_process_and_store(updated_data['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'],
                                     assistant_prompts.dict(), agent_uuid, user_id, synthesizer_config = updated_data['tasks'][0]['tools_config']['synthesizer'], tasks=updated_data['tasks']))

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
    dynamodb_task = asyncio.create_task(
        dynamodb.store_agent_data(user_id, f"agent#{agent_uuid}", json.loads(agent_data.json())))
    await asyncio.gather(redis_task, dynamodb_task)
    # response = requests.get('http://twilio-app:8001/make_call?agent_uuid={}'.format(agent_uuid))

    return {"agent_id": "{}".format(agent_uuid), "state": "created"}

@app.get("/assistant/analytics")
async def get_files(user_id: str, assistant_id: str):
    try:
        agent_analytics = await dynamodb.get_assistant_data(user_id, assistant_id, True)
        logger.info(f"Found high level assistant analytics {agent_analytics} ")
        return {
            "data": agent_analytics
        }
    except HTTPException as e:
        return e.detail


@app.get("/assistant/executions")
async def get_runs(user_id: str, assistant_id: str):
    try:
        logger.info(f"user id {user_id} assistant_id {assistant_id}")
        agent_analytics = await dynamodb.get_all_executions_for_assistant(user_id, assistant_id)
        logger.info(f"Found high level assistant analytics {agent_analytics} ")
        return {
            "data": agent_analytics
        }
    except HTTPException as e:
        return e.detail


##### User Level endpoints
@app.post("/tts")
async def get_voice_demo(tts_request: VoiceRequestModel):
    try:
        tts_request = tts_request.dict()
        logger.info(f"Generating audio for tts request {tts_request}")
        synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(tts_request['provider'])
        text = tts_request.pop('text')
        logger.info(f'Provider config {tts_request["provider_config"]} synthesier class {synthesizer_class}')
        synthesizer = synthesizer_class(stream = False, **tts_request["provider_config"])
        audio = None
        async for message in synthesizer.generate(text):
                audio = message
                base64_audio_frame = base64.b64encode(audio).decode('utf-8')
        return {
            "data": base64_audio_frame
        }
    except Exception as e:
        logger.error(f"Error generating audio {e}")
        traceback.print_exc()

@app.get("/user/models")
async def get_voices(user_id: str):
    try:
        user_details = await dynamodb.get_user(user_id)
        logger.info(f"user voices {user_details['voices']} ")
        return {
            "voices": user_details['voices'],
            "llmModels": user_details['llm_models']
        }
    except HTTPException as e:
        return e.detail

@app.post("/user/voice")
async def add_voices(user_voice: AddVoiceModel):
    try:
        logger.info(f"user id {user_voice.user_id} voice {user_voice.dict()}")
        user_details = await dynamodb.add_voice(user_voice.user_id, user_voice.voice.dict())
        return {
            "message": "added"
        }
    except HTTPException as e:
        return e.detail

@app.post("/user")
async def store_user(user: CreateUserModel):
    try:
        
        user_details = {
            "wallet": user.user_id,
            "voices": DEFAULT_VOICES,
            "llm_models": DEFAULT_LLM_MODELS
        }
        logger.info(f"saving {user_details}")
        agent_analytics = await dynamodb.store_user(user.user_id, user_details)
        return {
            "data": agent_analytics
        }
    except HTTPException as e:
        return e.detail


############################################################################################# 
# Websocket
#############################################################################################
@app.websocket("/chat/v1/{user_id}/{agent_id}")
async def websocket_endpoint(agent_id: str, user_id: str, websocket: WebSocket, user_agent: str = Query(None)):
    logger.info("Connected to ws")
    await websocket.accept()
    active_websockets.append(websocket)
    connected_through_dashboard = True if user_agent == "dashboard" else False

    agent_config, context_data = None, None
    try:
        retrieved_agent_config, retrieved_context_data = await redis_client.mget([agent_id, user_id])
        logger.info(
            f"Retrieved agent config: {retrieved_agent_config}, retrieved_context_data {retrieved_context_data}")
        if retrieved_agent_config is None:
            logger.info(
                "Could not retreive config from redis. So, simply retreiving it from DB and dumping it to redis")
            retrieved_agent_config = await dynamodb.get_assistant_data(user_id, f"agent#{agent_id}")
            redis_task = asyncio.create_task(redis_client.set(agent_id, json.dumps(retrieved_agent_config)))

        agent_config, context_data = json.loads(retrieved_agent_config), json.loads(
            retrieved_context_data) if retrieved_context_data is not None else None
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_manager = AssistantManager(agent_config, websocket, context_data, user_id, agent_id,
                                     connected_through_dashboard=connected_through_dashboard)

    llm_model = agent_config["tasks"][0]["tools_config"]["llm_agent"]["streaming_model"]
    pricing = load_file("model_pricing.json", is_json = True)
    conversation_data = None

    cost_breakdown = {
        "transcriber": 0,
        "llm": 0,
        "synthesizer": 0,
        "network": 0
    }
    usage_breakdown = {
            "transcriberModel": agent_config["tasks"][0]["tools_config"]["transcriber"]["model"],
            "transcriberDuration": 0,
            "llmModel": {
                llm_model: { 
                    "input": 0,
                    "output": 0
                }
            },
            "llmTokens": 0,
            "synthesizerModel": agent_config["tasks"][0]["tools_config"]["synthesizer"]["provider"],
            "synthesizerCharacters": 0
    }

    assistant_analytics_input = None
    try:
        async for index, task_output in agent_manager.run():
            if index == 0:
                logger.info(f"conversation task output {task_output}")
                usage_breakdown["synthesizerCharacters"] = task_output["synthesizer_characters"]
                usage_breakdown["transcriberDuration"] = task_output["transcriber_duration"]
                messages = task_output.get("messages")
                
                check_for_completion = False
                if agent_config['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'] != "preprocessed":
                    check_for_completion = True
                    completion_model = os.getenv('CHECK_FOR_COMPLETION_LLM')
                
                cost_breakdown["llm"], usage_breakdown["llmModel"] = calculate_total_cost_of_llm_from_transcript(messages =messages, cost_per_input_token = pricing["llm"][llm_model]["input"], 
                        cost_per_output_token = pricing["llm"][llm_model]["output"], model =llm_model, check_for_completion = check_for_completion, 
                        ended_by_assistant = task_output["ended_by_assistant"], completion_input_token_cost = pricing["llm"][completion_model]["input"], 
                        completion_output_token_cost = pricing["llm"][completion_model]["output"])
                cost_breakdown["synthesizer"] = round(task_output["synthesizer_characters"] * pricing["tts"]["polly-neural"], 5)
                cost_breakdown["transcriber"] = round(task_output["transcriber_duration"]/60 * pricing["asr"]["nova-2"], 5)
                total_cost = round(cost_breakdown["llm"], 5) + cost_breakdown["synthesizer"] + cost_breakdown["transcriber"] 
                messages = format_messages(messages)
                
                agent_run_details = {
                    "transcript": messages,
                    "conversation_time": round(task_output["conversation_time"], 3),
                    "usage_breakdown": usage_breakdown,
                    "cost_breakdown": cost_breakdown,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "total_cost": total_cost
                }
                assistant_analytics_input = agent_run_details
                conversation_data = messages
                logger.info(f"storing conversation task output {agent_run_details}")
                await dynamodb.store_run(user_id, agent_id, task_output["run_id"], agent_run_details)
            else:
                # Calculating cost for extraction
                if task_output["task_type"] == "extraction":
                    model  = agent_config["tasks"][index]["tools_config"]["llm_agent"]["streaming_model"]
                    input_tokens = token_counter(model=model, text=conversation_data)
                    output_tokens = token_counter(model=model, text=json.dumps(task_output["extracted_data"]))
                    task_cost = round(input_tokens * pricing["llm"][model]["input"] + output_tokens * pricing["llm"][model]["output"], 5)
                    cost_breakdown["llm"] += task_cost
                    assistant_analytics_input["total_cost"] += task_cost
                    if model not in usage_breakdown["llmModel"]:
                        usage_breakdown["llmModel"][model] = {"input": 0, "output": 0}
                    
                    total_cost = round(cost_breakdown["llm"] )
                    usage_breakdown["llmModel"][model]["input"] += input_tokens
                    usage_breakdown["llmModel"][model]["output"] += output_tokens

                    agent_run_details = {
                        "extracted_data": task_output["extracted_data"], 
                        "cost_breakdown": cost_breakdown,
                        "usage_breakdown": usage_breakdown,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "total_cost": assistant_analytics_input["total_cost"]
                    }
                    assistant_analytics_input = {**assistant_analytics_input, **agent_run_details}
                    await dynamodb.update_run(user_id, agent_id, task_output["run_id"], agent_run_details)

                elif task_output["task_type"] == "summarization":
                    model  = agent_config["tasks"][index]["tools_config"]["llm_agent"]["streaming_model"]
                    input_tokens = token_counter(model=model, text=conversation_data)
                    output_tokens = token_counter(model=model, text=task_output["summary"])
                    task_cost = round(input_tokens * pricing["llm"][model]["input"] + output_tokens * pricing["llm"][model]["output"], 5)
                    cost_breakdown["llm"] += task_cost
                    assistant_analytics_input["total_cost"] += task_cost

                    if model not in usage_breakdown["llmModel"]:
                        usage_breakdown["llmModel"][model] = {"input": 0, "output": 0}
                    
                    usage_breakdown["llmModel"][model]["input"] += input_tokens
                    usage_breakdown["llmModel"][model]["output"] += output_tokens

                    agent_run_details = {
                        "summary": task_output["summary"], 
                        "cost_breakdown": cost_breakdown,
                        "usage_breakdown": usage_breakdown,
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                        "total_cost": assistant_analytics_input["total_cost"]
                    }
                    assistant_analytics_input = {**assistant_analytics_input, **agent_run_details}
                    await dynamodb.update_run(user_id, agent_id, task_output["run_id"], {"summary": task_output["summary"], "cost_breakdown": cost_breakdown})
        logger.info(f"assitant_run_details {assistant_analytics_input}")

        agent_analytics = await dynamodb.get_assistant_data(user_id, agent_id, True)
        logger.info(f"agent_analytics {agent_analytics}")
        high_level_analytics_object = update_high_level_assistant_analytics_data(agent_analytics, assistant_analytics_input)
        logger.info(f"high_level_analytics_object {high_level_analytics_object}")
        await dynamodb.store_agent_data(user_id, f"analytics#{agent_id}", high_level_analytics_object)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")