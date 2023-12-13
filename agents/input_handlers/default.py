import asyncio
import base64
from dotenv import load_dotenv
from agents.helpers.logger_config import configure_logger
from agents.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class DefaultInputHandler:
    def __init__(self, queues=None, websocket=None, input_types=None, mark_set=None):
        self.queues = queues
        self.websocket = websocket
        self.input_types = input_types
        self.websocket_listen_task = None
        self.running = True

    async def stop_handler(self):
        self.running = False
        try:
            await self.websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

    async def _listen(self):
        try:
            while self.running:
                request = await self.websocket.receive_json()
                if request['type'] not in self.input_types.keys():
                    return {"message": "invalid input type"}

                if request['type'] == 'audio':
                    data = base64.b64decode(request['data'])
                    ws_data_packet = create_ws_data_packet(
                        data=data,
                        meta_info={
                            'io': 'default',
                            'type': request['type'],
                            'sequence': self.input_types['audio']
                        })
                    self.queues['transcriber'].put_nowait(ws_data_packet)

                elif request["type"] == "text":
                    data = request['data']
                    ws_data_packet = create_ws_data_packet(
                        data=data,
                        meta_info={
                            'io': 'default',
                            'type': request['type']

                        })
                    self.queues['llm'].put_nowait(ws_data_packet)
                else:
                    return {"message": "Other modalities not implemented yet"}
        except Exception as e:
            ws_data_packet = create_ws_data_packet(
                data=None,
                meta_info={
                    'io': 'default',
                    'eos': True
                })
            self.queues['transcriber'].put_nowait(ws_data_packet)
            logger.info(f"Error while handling websocket message: {e}")
            return

    async def handle(self):
        self.websocket_listen_task = asyncio.create_task(self._listen())
