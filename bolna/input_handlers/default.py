import asyncio
import base64
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class DefaultInputHandler:
    def __init__(self, queues=None, websocket=None, input_types=None, mark_set=None, connected_through_dashboard=False):
        self.queues = queues
        self.websocket = websocket
        self.input_types = input_types
        self.websocket_listen_task = None
        self.running = True
        self.connected_through_dashboard = connected_through_dashboard

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

                if request['type'] not in self.input_types.keys() and not self.connected_through_dashboard:
                    logger.info(f"straight away returning")
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
                    logger.info(f"Received text: {request['data']}")
                    data = request['data']
                    logger.info(f"Sequences {self.input_types}")
                    ws_data_packet = create_ws_data_packet(
                        data=data,
                        meta_info={
                            'io': 'default',
                            'type': request['type'],
                            'sequence': self.input_types['audio']

                        })

                    if self.connected_through_dashboard:
                        ws_data_packet["meta_info"]["bypass_synth"] = True

                    self.queues['llm'].put_nowait(ws_data_packet)
                    logger.info(f"Put into llm queue")
                else:
                    return {"message": "Other modalities not implemented yet"}
        except Exception as e:
            # Send EOS message to transcriber to shut the connection
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
