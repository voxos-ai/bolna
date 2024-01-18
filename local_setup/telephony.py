import asyncio

from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client

from pyngrok import ngrok
from flask import Flask, request
import os
from dotenv import load_dotenv
import redis.asyncio as redis
from pydantic import BaseModel

load_dotenv()

app = Flask(__name__)

port = 8001
websocket_url = os.getenv('WEBSOCKET_URL')
print(f"websocket_url {websocket_url}")

twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)
redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)


async def close_redis_connection():
    await redis_client.aclose()


@app.route('/make_call', methods=['POST'])
async def make_call():
    call_details = request.get_json()
    public_url = os.getenv('APP_CALLBACK_URL')
    if os.getenv('ENVIRONMENT') == 'local':
        public_url = ngrok.connect(port, bind_tls=True).public_url

    call = twilio_client.calls.create(
            to=call_details.get('recipient_phone_number'),
            from_=twilio_phone_number,
            url="{}/twilio_callback".format(public_url),
            method="POST",
            record=True
        )

    print(public_url)
    await redis_client.set(call.sid, str(call_details))
    print('call sid -> {}'.format(call.sid))

    await close_redis_connection()
    return "done", 200


@app.route('/twilio_callback', methods=['POST'])
async def twilio_callback():
    try:
        print("reached here")
        response = VoiceResponse()

        connect = Connect()
        print("connected")
        response.say('Please speak now')
        connect.stream(url=websocket_url)
        print("websocket connection done to {}".format(websocket_url))
        response.append(connect)
        print("Appending connect")
        
        print("Pause done")
        print(f'Incoming call from {request.form["From"]}')
        return str(response), 200, {'Content-Type': 'text/xml'}
    except Exception as e:
        print("here")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(asyncio.gather(
            loop.create_task(app.run(debug=True, port=port, host='0.0.0.0')),
        ))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
