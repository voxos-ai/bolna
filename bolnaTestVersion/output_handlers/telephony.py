import base64
import json
import os
import audioop
import uuid
import traceback
from dotenv import load_dotenv
from .default import DefaultOutputHandler
from bolnaTestVersion.helpers.logger_config import configure_logger
logger = configure_logger(__name__)
load_dotenv()


class TelephonyOutputHandler(DefaultOutputHandler):
    def __init__(self, websocket=None, mark_set=None, log_dir_name=None):
        super().__init__(websocket, log_dir_name)
        self.mark_set = mark_set

        self.stream_sid = None
        self.current_request_id = None
        self.rejected_request_ids = set()

    async def handle_interruption(self):
        pass

    async def form_media_message(self, audio_data, audio_format):
        pass

    async def form_mark_message(self, mark_id):
        pass

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
                    audio_format = meta_info.get("format", "wav")
                    media_message = await self.form_media_message(audio_chunk, audio_format)
                    await self.websocket.send_text(json.dumps(media_message))

                    mark_id = str(uuid.uuid4())
                    self.mark_set.add(mark_id)

                    mark_message = await self.form_mark_message(mark_id)
                    await self.websocket.send_text(json.dumps(mark_message))
            except Exception as e:
                traceback.print_exc()
                logger.error(f'something went wrong while sending message to twilio {e}')

        except Exception as e:
            logger.error(f'something went wrong while handling twilio {e}')
