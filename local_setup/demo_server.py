import os
import asyncio
import json
import uuid
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from typing import List
import redis.asyncio as redis
from dotenv import load_dotenv
from bolna.agent_manager import AssistantManager
from bolna.helpers.logger_config import configure_logger
from bolna.models import AssistantModel

logger = configure_logger(__name__)
load_dotenv()
redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)
active_websockets: List[WebSocket] = []
app = FastAPI()


@app.post("/create_agent")
async def create_agent(agent_data: AssistantModel):
    agent_uuid = '{}'.format(str(uuid.uuid4()))
    redis_task = asyncio.create_task(redis_client.set(agent_uuid, agent_data.json()))
    await asyncio.gather(redis_task)

    return {"agent_id": "{}".format(agent_uuid), "state": "created"}


@app.websocket("/chat/v1/{user_id}/{agent_id}")
async def websocket_endpoint(agent_id: str, user_id: str, websocket: WebSocket):
    logger.info('ws connected with user_id: {} and agent_id: {}'.format(user_id, agent_id))

    await websocket.accept()
    active_websockets.append(websocket)

    agent_config, context_data = None, None
    try:
        retrieved_agent_config, retrieved_context_data = await redis_client.mget([agent_id, user_id])
        agent_config, context_data = json.loads(retrieved_agent_config), json.loads(retrieved_context_data)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Agent not found")

    is_local = True
    agent_manager = AssistantManager(agent_config, websocket, context_data, user_id, agent_id)

    try:
        async for res in agent_manager.run(is_local):
            print(res)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"error in executing {e}")
