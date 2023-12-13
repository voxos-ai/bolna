import base64
from dotenv import load_dotenv
from agents.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv()


class DefaultOutputHandler:
    def __init__(self, websocket=None):
        self.websocket = websocket

    # @TODO Figure out the best way to handle this
    async def handle_interruption(self):
        message_clear = {
            "event": "clear"
        }

    async def handle(self, packet):
        try:
            if packet["meta_info"]['type'] == 'audio':
                base64_audio = base64.b64encode(packet['data']).decode("utf-8")
                response = {"data": base64_audio, "message": "audio"}
                await self.websocket.send_json(response)
            else:
                logger.error("Other modalities are not implemented yet")
        except Exception as e:
            logger.error(f"something went wrong in speaking {e}")
