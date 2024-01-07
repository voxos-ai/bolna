import base64
import json
import os
import audioop
import uuid
from twilio.rest import Client
from dotenv import load_dotenv
import redis.asyncio as redis
from .default import DefaultOutputHandler
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()

twilio_client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))
redis_pool = redis.ConnectionPool.from_url(os.getenv('REDIS_URL'), decode_responses=True)
redis_client = redis.Redis.from_pool(redis_pool)


class TwilioOutputHandler(DefaultOutputHandler):
    def __init__(self, websocket=None, mark_set=None, log_dir_name=None):
        super().__init__(websocket, log_dir_name)
        self.mark_set = mark_set

        self.stream_sid = None
        self.current_request_id = None
        self.rejected_request_ids = set()

    async def handle_interruption(self):
        logger.info("interrupting because user spoke in between")
        if len(self.mark_set) > 0:
            message_clear = {
                "event": "clear",
                "streamSid": self.stream_sid,
            }
            await self.websocket.send_text(json.dumps(message_clear))
            self.mark_set = set()

    async def send_sms(self, message_text, call_number):
        message = twilio_client.messages.create(
            to='{}'.format(call_number),
            from_='{}'.format(os.getenv('TWILIO_PHONE_NUMBER')),
            body=message_text)
        logger.info(f'Sent whatsapp message: {message_text}')
        return message.sid

    async def send_whatsapp(self, message_text, call_number):
        message = twilio_client.messages.create(
            to='whatsapp:{}'.format(call_number),
            from_='whatsapp:{}'.format(os.getenv('TWILIO_PHONE_NUMBER')),
            body=message_text)
        logger.info(f'Sent whatsapp message: {message_text}')
        return message.sid

    async def handle(self, ws_data_packet):
        try:
            audio_chunk = ws_data_packet.get('data')
            meta_info = ws_data_packet.get('meta_info')
            self.stream_sid = meta_info.get('stream_sid', None)

            try:
                if self.current_request_id == meta_info['request_id']:
                    if len(audio_chunk) == 1:
                        audio_chunk += b'\x00'

                if audio_chunk and self.stream_sid and len(audio_chunk) != 1:

                    audio = audioop.lin2ulaw(audio_chunk, 2)
                    base64_audio = base64.b64encode(audio).decode("utf-8")
                    message = {
                        'event': 'media',
                        'streamSid': self.stream_sid,
                        'media': {
                            'payload': base64_audio
                        }
                    }

                    await self.websocket.send_text(json.dumps(message))

                    mark_id = str(uuid.uuid4())
                    self.mark_set.add(mark_id)
                    mark_message = {
                        "event": "mark",
                        "streamSid": self.stream_sid,
                        "mark": {
                            "name": mark_id
                        }
                    }
                    await self.websocket.send_text(json.dumps(mark_message))
            except Exception as e:
                logger.error(f'something went wrong while sending message to twilio {e}')

        except Exception as e:
            logger.error(f'something went wrong while handling twilio {e}')
