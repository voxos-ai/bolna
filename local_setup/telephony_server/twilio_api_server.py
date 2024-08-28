import os
import json
import requests
import uuid
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from dotenv import load_dotenv
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.responses import PlainTextResponse

app = FastAPI()
load_dotenv(override=True)
port = 8001

twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)


def populate_ngrok_tunnels():
    response = requests.get("http://ngrok:4040/api/tunnels")  # ngrok interface
    app_callback_url, websocket_url = None, None
    telephony_url, bolna_url = None, None

    if response.status_code == 200:
        data = response.json()

        for tunnel in data['tunnels']:
            if tunnel['name'] == 'twilio-app':
                telephony_url = tunnel['public_url']
            elif tunnel['name'] == 'bolna-app':
                bolna_url = tunnel['public_url'].replace('https:', 'wss:')

        return telephony_url, bolna_url
    else:
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")


@app.post('/call')
async def make_call(request: Request):
    try:
        call_details = await request.json()
        agent_id = call_details.get('agent_id', None)

        if not agent_id:
            raise HTTPException(status_code=404, detail="Agent not provided")
        
        if not call_details or "recipient_phone_number" not in call_details:
            raise HTTPException(status_code=404, detail="Recipient phone number not provided")

        telephony_host, bolna_host = populate_ngrok_tunnels()

        print(f'telephony_host: {telephony_host}')
        print(f'bolna_host: {bolna_host}')

        try:
            call = twilio_client.calls.create(
                to=call_details.get('recipient_phone_number'),
                from_=twilio_phone_number,
                url=f"{telephony_host}/twilio_connect?bolna_host={bolna_host}&agent_id={agent_id}",
                method="POST",
                record=True
            )
        except Exception as e:
            print(f'make_call exception: {str(e)}')

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        print(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# # Use this Endpoint to call the playground endpoint directly!
# @app.post('/call')
# async def make_call_v2(request: Request, authorization: str = Header(...)):
#     try:
#         call_details = await request.json()
#         agent_id = call_details.get('agent_id')
#         print(f"Agent ID : {agent_id}")
#         recipient_phone_number = call_details.get('recipient_phone_number')
#         user_data = call_details.get('user_data', {})

#         if not agent_id:
#             raise HTTPException(status_code=400, detail="Agent ID not provided")
        
#         if not recipient_phone_number:
#             raise HTTPException(status_code=400, detail="Recipient phone number not provided")

#         # Prepare the payload for the new endpoint
#         payload = {
#             "agent_id": agent_id,
#             "recipient_phone_number": recipient_phone_number,
#         }

#         headers = {
#             'Authorization': authorization,
#             'Content-Type': 'application/json'
#         }

#         # Send the request to the new endpoint
#         response = requests.post("https://api.bolna.dev/call", headers=headers, json=payload)
#         print(f"Response status code: {response.status_code}")
#         print(f"Response content: {response.content}")

#         if response.status_code != 200:
#             raise HTTPException(status_code=response.status_code, detail=f"Failed to make call: {response.text}")

#         return PlainTextResponse("Call initiated successfully", status_code=200)

#     except Exception as e:
#         print(f"Exception occurred in make_call_v2: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post('/twilio_connect')
async def twilio_connect(bolna_host: str = Query(...), agent_id: str = Query(...)):
    try:
        response = VoiceResponse()

        connect = Connect()
        bolna_websocket_url = f'{bolna_host}/chat/v1/{agent_id}'
        connect.stream(url=bolna_websocket_url)
        print(f"websocket connection done to {bolna_websocket_url}")
        response.append(connect)

        return PlainTextResponse(str(response), status_code=200, media_type='text/xml')

    except Exception as e:
        print(f"Exception occurred in twilio_callback: {e}")
