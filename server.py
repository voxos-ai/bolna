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
from bolna.helpers.analytics_helpers import calculate_total_cost_of_llm_from_transcript
from bolna.helpers.utils import get_md5_hash, get_s3_file, put_s3_file, format_messages, load_file
from bolna.providers import *
from bolna.prompts import *
from bolna.helpers.logger_config import configure_logger
from bolna.models import *
from litellm import token_counter

from bolna.agent_manager.assistant_manager import AssistantManager
from database.dynamodb import DynamoDB

load_dotenv()
logger = configure_logger(__name__, True)
redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
active_websockets: List[WebSocket] = []
app = FastAPI()
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
        existing_data = await dynamodb.get_agent_data(user_id, f"agent#{assistant_id}")
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


async def generate_audio_from_text(text):
    # A simple polly or eleven labs call for now
    synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get("polly")
    synth_instance = synthesizer_class('polly', 'mp3', 'Kajal', 'en', '24000')
    async for audio_data in synth_instance.generate_tts_response(text):
        return audio_data


async def process_and_store_audio(conversation_graph, user_id, assistant_id):
    logger.info(f"Generating and storing data in S3 for the conversation graph {conversation_graph}")
    for node_key, node_value in conversation_graph['task_1'].items():
        for content in node_value['content']:
            if 'text' in content:
                text = content['text']
                file_name = get_md5_hash(content['text'])
                content['audio'] = file_name
                audio_data = await generate_audio_from_text(text)
                await put_s3_file(BUCKET_NAME, f"{user_id}/{assistant_id}/audio/{file_name}.mp3", audio_data, "mp3")
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


async def background_process_and_store(conversation_type, prompt_json, assistant_id, user_id, model_name="polly",
                                       tasks=None):
    try:

        # For loop and add multiple prompts into the prompt file from here.
        if conversation_type == "preprocessed":
            logger.info(f"Prompt json {prompt_json}")
            conversation_graph = prompt_json['conversation_graph']
            logger.info(f"Preprocessed conversation. Storing required files to s3")
            prompt_json = create_prompts_for_followup_tasks(tasks, prompt_json)
            await process_and_store_audio(prompt_json['serialized_prompts'], user_id, assistant_id)
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
    redis_task = asyncio.create_task(redis_client.set(agent_uuid, json.dumps(data_for_db)))
    dynamodb_task = asyncio.create_task(dynamodb.store_agent(user_id, f"agent#{agent_uuid}", data_for_db))
    background_process_and_store_task = asyncio.create_task(
        background_process_and_store(data_for_db['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'],
                                     assistant_prompts.dict(), agent_uuid, user_id, tasks=data_for_db['tasks']))
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
    background_process_and_store_task = asyncio.create_task(
        background_process_and_store(updated_data['tasks'][0]['tools_config']['llm_agent']['agent_flow_type'],
                                     assistant_prompts.dict(), agent_uuid, user_id, tasks=updated_data['tasks']))

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
        dynamodb.store_agent(user_id, f"agent#{agent_uuid}", json.loads(agent_data.json())))
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
        logger.info(
            f"Retrieved agent config: {retrieved_agent_config}, retrieved_context_data {retrieved_context_data}")
        if retrieved_agent_config is None:
            logger.info(
                "Could not retreive config from redis. So, simply retreiving it from DB and dumping it to redis")
            retrieved_agent_config = await dynamodb.get_agent_data(user_id, f"agent#{agent_id}")
            redis_task = asyncio.create_task(redis_client.set(agent_id, json.dumps(retrieved_agent_config)))

        agent_config, context_data = json.loads(retrieved_agent_config), json.loads(
            retrieved_context_data) if retrieved_context_data is not None else None
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_manager = AssistantManager(agent_config, websocket, context_data, user_id, agent_id,
                                     connected_through_dashboard=connected_through_dashboard)

    cost_breakdown = {
        "transcriber": 0,
        "llm": 0,
        "synth": 0,
        "network": 0
    }
    usage_breakdown = {
            "transcriberModel": "nova2",
            "transcriberDuration": 0,
            "llmModel": "gpt-3.5-turbo",
            "llmTokens": 0,
            "synthesizerModel": "openai-tts",
            "synthesizerCharacters": 0
    }

    llm_model = agent_config["tasks"][0]["tools_config"]["llm_agent"]["streaming_model"]
    pricing = load_file("model_pricing.json", is_json = True)
    conversation_data = None
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
                cost_breakdown["llm"] = calculate_total_cost_of_llm_from_transcript(messages =messages, cost_per_input_token = pricing["llm"][llm_model]["input"], 
                        cost_per_output_token = pricing["llm"][llm_model]["output"], model =llm_model, check_for_completion = check_for_completion, 
                        ended_by_assistant = task_output["ended_by_assistant"], completion_input_token_cost = pricing["llm"][completion_model]["input"], 
                        completion_output_token_cost = pricing["llm"][completion_model]["output"])
                cost_breakdown["synth"] = round(task_output["synthesizer_characters"] * pricing["tts"]["polly-neural"], 5)
                cost_breakdown["transcriber"] = round(task_output["transcriber_duration"]/60 * pricing["asr"]["nova-2"], 5)
                messages = format_messages(messages)
                agent_run_details = {
                    "transcript": messages,
                    "conversation_time": task_output["conversation_time"],
                    "usage_breakdown": usage_breakdown,
                    "cost_breakdown": cost_breakdown
                }
                conversation_data = messages
                logger.info(f"storing conversation task output {agent_run_details}")
                await dynamodb.store_run(user_id, agent_id, task_output["run_id"], agent_run_details)
            else:
                # Calculating cost for extraction
                if task_output["task_type"] == "extraction":
                    model  = agent_config["tasks"][index]["tools_config"]["llm_agent"]["streaming_model"]
                    input_tokens = token_counter(model=model, text=conversation_data)
                    output_tokens = token_counter(model=model, text=json.dumps(task_output["extracted_data"]))
                    cost_breakdown["llm"] = round(input_tokens * pricing["llm"][model]["input"] + output_tokens * pricing["llm"][model]["output"], 5)
                    await dynamodb.update_run(user_id, agent_id, task_output["run_id"], {"extracted_data": task_output["extracted_data"], "cost_breakdown": cost_breakdown})

                elif task_output["task_type"] == "summarization":
                    model  = agent_config["tasks"][index]["tools_config"]["llm_agent"]["streaming_model"]
                    input_tokens = token_counter(model=model, text=conversation_data)
                    output_tokens = token_counter(model=model, text=task_output["summary"])
                    cost_breakdown["llm"] = round(input_tokens * pricing["llm"][model]["input"] + output_tokens * pricing["llm"][model]["output"], 5)
                    await dynamodb.update_run(user_id, agent_id, task_output["run_id"], {"summary": task_output["summary"], "cost_breakdown": cost_breakdown})

                logger.info(f"storing followup task in database {task_output}")

    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")
