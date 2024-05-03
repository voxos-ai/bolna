import aiohttp
from .base_agent import BaseAgent
from bolnaTestVersion.helpers.logger_config import configure_logger

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
                        # need to check if the returned response is json or not
                        #data = await response.json()
                        return True
                    else:
                        logger.error(f"Error: {response.status} - {await response.text()}")
                        return None
            else:
                logger.info("Payload was null")
        return None

    async def execute(self, payload):
        if not self.zap_url:
            return None
        response = await self.__send_payload(payload)
        logger.info(f"Response {response}")
        return response
