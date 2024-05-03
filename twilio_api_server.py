import os
import json
import requests
import uuid
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from dotenv import load_dotenv
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
import uvicorn

app = FastAPI()
load_dotenv()
port = 8001

twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)

# Initialize Redis client
redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)


def populate_ngrok_tunnels():
    response = requests.get("http://localhost:4040/api/tunnels")  # ngrok interface
    # response = requests.get("http://ngrok:4040/api/tunnels") 
    app_callback_url, websocket_url = None, None

    if response.status_code == 200:
        data = response.json()

        for tunnel in data['tunnels']:
            if tunnel['name'] == 'twilio-app':
                app_callback_url = tunnel['public_url']
            elif tunnel['name'] == 'bolna-app':
                websocket_url = tunnel['public_url'].replace('https:', 'wss:')

        return app_callback_url, websocket_url
    else:
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")


async def close_redis_connection():
    await redis_client.aclose()


@app.post('/call')
async def make_call(request: Request):
    try:
        call_details = await request.json()
        agent_id = call_details.get('agent_id', None)

        if not agent_id:
            raise HTTPException(status_code=404, detail="Agent not provided")
        
        if not call_details or "recipient_phone_number" not in call_details:
            raise HTTPException(status_code=404, detail="Recipient phone number not provided")
        user_id = str(uuid.uuid4())
        app_callback_url = os.getenv('APP_CALLBACK_URL')
        websocket_url = os.getenv('WEBSOCKET_URL')

        if os.getenv('ENVIRONMENT') == 'local':
            app_callback_url, websocket_url = populate_ngrok_tunnels()

        print(f'app_callback_url: {app_callback_url}')
        print(f'websocket_url: {websocket_url}')

        call = twilio_client.calls.create(
            to=call_details.get('recipient_phone_number'),
            from_=twilio_phone_number,
            url=f"{app_callback_url}/twilio_callback?ws_url={websocket_url}&agent_id={agent_id}&user_id={user_id}",
            method="POST",
            record=False
        )

        # persisting user details
        await redis_client.set(user_id, json.dumps(call_details))

        await close_redis_connection()
        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        print(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post('/twilio_callback')
async def twilio_callback(ws_url: str = Query(...), agent_id: str = Query(...), user_id: str = Query(...)):
    try:
        response = VoiceResponse()

        connect = Connect()
        print("connected")
        response.say('Please speak now')

        websocket_twilio_route = f'{ws_url}/chat/v1/{agent_id}'
        connect.stream(url=websocket_twilio_route)
        print(f"websocket connection done to {websocket_twilio_route}")
        response.append(connect)

        return PlainTextResponse(str(response), status_code=200, media_type='text/xml')

    except Exception as e:
        print(f"Exception occurred in twilio_callback: {e}")
if __name__ == "__main__":
    uvicorn.run('twilio_api_server:app',port=8001,host='0.0.0.0',reload=True)