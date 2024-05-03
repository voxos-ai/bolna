import base64
import json
from dotenv import load_dotenv
from bolnaTestVersion.output_handlers.telephony import TelephonyOutputHandler
from bolnaTestVersion.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class ExotelOutputHandler(TelephonyOutputHandler):
    def __init__(self, websocket=None, mark_set=None, log_dir_name=None):
        super().__init__(websocket, mark_set, log_dir_name)
        self.io_provider = 'exotel'

    async def handle_interruption(self):
        logger.info("interrupting because user spoke in between")
        message_clear = {
            "event": "clear",
            "stream_sid": self.stream_sid,
        }
        await self.websocket.send_text(json.dumps(message_clear))
        self.mark_set = set()

    async def form_media_message(self, audio_data, audio_format):
        base64_audio = base64.b64encode(audio_data).decode("ascii")
        message = {
            'event': 'media',
            'stream_sid': self.stream_sid,
            'media': {
                'payload': base64_audio
            }
        }

        return message

    async def form_mark_message(self, mark_id):
        mark_message = {
            "event": "mark",
            "stream_sid": self.stream_sid,
            "mark": {
                "name": mark_id
            }
        }

        return mark_message
