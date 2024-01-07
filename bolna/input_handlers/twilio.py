from .default import DefaultInputHandler
import asyncio
import base64
import json
from twilio.rest import Client
from dotenv import load_dotenv
import os
from bolna.helpers.utils import create_ws_data_packet
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()

twilio_client = Client(os.getenv('TWILIO_ACCOUNT_SID'), os.getenv('TWILIO_AUTH_TOKEN'))


class TwilioInputHandler(DefaultInputHandler):
    def __init__(self, queues, websocket=None, input_types=None, mark_set=None, connected_through_dashboard=False):
        super().__init__(queues, websocket, input_types, connected_through_dashboard)
        self.stream_sid = None
        self.call_sid = None
        self.buffer = []
        self.message_count = 0
        self.mark_set = mark_set
        self.last_media_received = 0

    async def call_start(self, packet):
        start = packet['start']
        self.call_sid = start['callSid']
        self.stream_sid = start['streamSid']

    async def process_mark_message(self, packet):
        if packet["mark"]["name"] in self.mark_set:
            self.mark_set.remove(packet["mark"]["name"])

    async def stop_handler(self):
        logger.info("stopping handler")
        self.running = False
        logger.info("sleeping for 5 seconds so that whatever needs to pass is passed")
        await asyncio.sleep(5)
        try:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

    async def ingest_audio(self, audio_data, meta_info):
        ws_data_packet = create_ws_data_packet(data=audio_data, meta_info=meta_info)
        self.queues['transcriber'].put_nowait(ws_data_packet)

    async def _listen(self):
        while True:
            try:
                message = await self.websocket.receive_text()

                packet = json.loads(message)
                if packet['event'] == 'start':
                    await self.call_start(packet)
                elif packet['event'] == 'media':
                    media_data = packet['media']
                    media_audio = base64.b64decode(media_data['payload'])
                    media_ts = int(media_data["timestamp"])

                    if packet['media']['track'] == 'inbound':
                        meta_info = {
                            'io': 'twilio',
                            'call_sid': self.call_sid,
                            'stream_sid': self.stream_sid,
                            'sequence': self.input_types['audio']
                        }

                        if self.last_media_received + 20 < media_ts:
                            bytes_to_fill = 8 * (media_ts - (self.last_media_received + 20))
                            logger.info(f"Filling {bytes_to_fill} bytes of silence")
                            await self.ingest_audio(b"\xff" * bytes_to_fill, meta_info)

                        self.last_media_received = media_ts
                        await self.ingest_audio(media_audio, meta_info)
                    else:
                        logger.info("Getting media elements but not inbound media")

                elif packet['event'] == 'mark':
                    await self.process_mark_message(packet)

                elif packet['event'] == 'stop':
                    logger.info('call stopping')
                    ws_data_packet = create_ws_data_packet(data=None, meta_info={'io': 'default', 'eos': True})
                    self.queues['transcriber'].put_nowait(ws_data_packet)
                    break

            except Exception as e:
                ws_data_packet = create_ws_data_packet(
                    data=None,
                    meta_info={
                        'io': 'default',
                        'eos': True
                    })
                self.queues['transcriber'].put_nowait(ws_data_packet)
                logger.error('Exception in twilio_receiver reading events: {}'.format(e))
                break

    async def handle(self):
        self.websocket_listen_task = asyncio.create_task(self._listen())
