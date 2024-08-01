import os
import asyncio
import uuid
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from dotenv import load_dotenv
from bolna.helpers.utils import store_file
from bolna.prompts import *
from bolna.helpers.logger_config import configure_logger
from bolna.models import *
from bolna.llms import LiteLLM
from bolna.agent_manager.assistant_manager import AssistantManager
from bolna.helpers.data_ingestion_pipe import create_table, ingestion_task, ingestion_tasks, IngestionPipeline, IngestionTask, TaskStatus
import tempfile
import threading
from typing import Dict
import os 
load_dotenv()
logger = configure_logger(__name__)

redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
active_websockets: List[WebSocket] = []

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

logger.info(f"PROCESS ID {os.getpid()}")

class CreateAgentPayload(BaseModel):
    agent_config: AgentModel
    agent_prompts: Optional[Dict[str, Dict[str, str]]]

ingestion_pipeline = IngestionPipeline()
@app.post("/ingestion-pipeline")
async def start_ingestion(table_id:str,file:UploadFile):
    if file.content_type in ["application/pdf","application/x-pdf"]:
        task_id = str(uuid.uuid4())

        logger.info(f"added new task: {task_id}")
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(await file.read())
        prev = temp_file.name
        file_name = f"/tmp/{task_id}.pdf"
        os.rename(prev,file_name)


        task = IngestionTask(file_name)
        await ingestion_pipeline.task_queue.put(task)
        while task._status != TaskStatus.ERROR:
            await asyncio.sleep(0.3)
            logger.info(f"TASK: {task_id} status: {task._status}")
            if task._status == TaskStatus.SUCCESS:
                break
        

        
        # if table_id != "None":
        #     table = table_id
        # else:
        #     table = str(uuid.uuid4())
        
        # thread = threading.Thread(target=create_table,args=(table,file_name))
        # thread.start()
        # while thread.is_alive():
        #     await asyncio.sleep(0.3)


        os.rename(file_name,prev)


        return {"status":task._status,"message":task._message,"table_id":task._table_id}
    
    return {"status":TaskStatus.ERROR,"message":"only accept the pdf types for know","task_id":None}

@app.post("/agent")
async def create_agent(agent_data: CreateAgentPayload):
    agent_uuid = str(uuid.uuid4())
    data_for_db = agent_data.agent_config.model_dump()
    data_for_db["assistant_status"] = "seeding"
    agent_prompts = agent_data.agent_prompts
    logger.info(f'Data for DB {data_for_db}')

    if len(data_for_db['tasks']) > 0:
        logger.info("Setting up follow up tasks")
        for index, task in enumerate(data_for_db['tasks']):
            if task['task_type'] == "extraction":
                extraction_prompt_llm = os.getenv("EXTRACTION_PROMPT_GENERATION_MODEL")
                extraction_prompt_generation_llm = LiteLLM(model=extraction_prompt_llm, max_tokens=2000)
                extraction_prompt = await extraction_prompt_generation_llm.generate(
                    messages=[
                        {'role': 'system', 'content': EXTRACTION_PROMPT_GENERATION_PROMPT},
                        {'role': 'user', 'content': data_for_db["tasks"][index]['tools_config']["llm_agent"]['extraction_details']}
                    ])
                data_for_db["tasks"][index]["tools_config"]["llm_agent"]['extraction_json'] = extraction_prompt

    stored_prompt_file_path = f"{agent_uuid}/conversation_details.json"
    await asyncio.gather(
        redis_client.set(agent_uuid, json.dumps(data_for_db)),
        store_file(file_key=stored_prompt_file_path, file_data=agent_prompts, local=True)
    )

    return {"agent_id": agent_uuid, "state": "created"}


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
        async for index, task_output in assistant_manager.run(local=True):
            logger.info(task_output)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")