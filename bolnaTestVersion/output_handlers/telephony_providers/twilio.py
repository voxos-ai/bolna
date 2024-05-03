import base64
import json
import os
import audioop
from twilio.rest import Client
from dotenv import load_dotenv
from bolnaTestVersion.helpers.logger_config import configure_logger
from bolnaTestVersion.output_handlers.telephony import TelephonyOutputHandler

logger = configure_logger(__name__)
load_dotenv()


class TwilioOutputHandler(TelephonyOutputHandler):
    def __init__(self, websocket=None, mark_set=None, log_dir_name=None):
        super().__init__(websocket, mark_set, log_dir_name)
        self.io_provider = 'twilio'
        self.client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))

    async def handle_interruption(self):
        logger.info("interrupting because user spoke in between")
        message_clear = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        await self.websocket.send_text(json.dumps(message_clear))
        self.mark_set = set()

    async def form_media_message(self, audio_data, audio_format="wav"):
        if audio_format != "mulaw":
            audio_data = audioop.lin2ulaw(audio_data, 2)
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        message = {
            'event': 'media',
            'streamSid': self.stream_sid,
            'media': {
                'payload': base64_audio
            }
        }

        return message

    async def form_mark_message(self, mark_id):
        mark_message = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": mark_id
            }
        }

        return mark_message

    async def send_sms(self, message_text, call_number):
        message = self.client.messages.create(
            to='{}'.format(call_number),
            from_='{}'.format(os.getenv('TWILIO_PHONE_NUMBER')),
            body=message_text)
        logger.info(f'Sent whatsapp message: {message_text}')
        return message.sid

    async def send_whatsapp(self, message_text, call_number):
        message = self.client.messages.create(
            to='whatsapp:{}'.format(call_number),
            from_='whatsapp:{}'.format(os.getenv('TWILIO_PHONE_NUMBER')),
            body=message_text)
        logger.info(f'Sent whatsapp message: {message_text}')
        return message.sid
