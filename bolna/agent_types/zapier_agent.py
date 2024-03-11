import aiohttp
from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ZapierAgent(BaseAgent):
    def __init__(self, zap_url, payload=None):
        super().__init__()
        self.zap_url = zap_url
        self.payload = payload or {}

    async def __send_payload(self, payload):
        logger.info(f"Sending a zapier post request {payload}")
        async with aiohttp.ClientSession() as session:
            if payload is not None:
                async with session.post(self.zap_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
            else:
                logger.info("Payload was null")

    async def execute(self, payload):
        response = await self.__send_payload(payload)
        logger.info(f"Response {response}")
        return response['status']
