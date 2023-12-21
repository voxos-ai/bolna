import base64
from dotenv import load_dotenv
from agents.helpers.logger_config import configure_logger

logger = configure_logger(__name__)
load_dotenv() 


class DefaultOutputHandler:
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.is_interruption_task_on = False

    # @TODO Figure out the best way to handle this
    async def handle_interruption(self):
        message_clear = {
            "event": "clear"
        }

    async def handle(self, packet):
        try:
            logger.info(f"Packet received:")
            if packet["meta_info"]['type'] == 'audio':
                logger.info(f"Sending audio")
                base64_audio = base64.b64encode(packet['data']).decode("utf-8")
                response = {"data": base64_audio, "type": "audio"}
                await self.websocket.send_json(response)
            elif packet["meta_info"]['type'] == 'text':
                logger.info(f"Sending text response {packet['data']}")
                response = {"data": packet['data'], "type": "text"}
                await self.websocket.send_json(response)

            else:
                logger.error("Other modalities are not implemented yet")
        except Exception as e:
            logger.error(f"something went wrong in speaking {e}")
