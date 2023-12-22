import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from .default import DefaultOutputHandler

from aiohttp import ClientSession

from aiodynamo.client import Client
from aiodynamo.credentials import Credentials
from aiodynamo.expressions import F, UpdateExpression
from aiodynamo.expressions import Condition
from aiodynamo.expressions import HashKey, RangeKey
from aiodynamo.http.aiohttp import AIOHTTP


logger = configure_logger(__name__)
load_dotenv()


class DatabaseOutputHandler(DefaultOutputHandler):
    def __init__(self, user_id, assistant_id, run_id):
        super().__init__()
        self.table_name = os.getenv('TABLE_NAME')
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.run_id = run_id
        self.client = None
        self.table = None
        self.session = None

    async def get_table(self):
        if self.table is None:
            self.session = ClientSession()
            self.client = Client(AIOHTTP(self.session), Credentials.auto(), "us-east-1")
            self.table = self.client.table(self.table_name)

        return self.table

    async def store_run(self, user_id, agent_id, run_id, run_parameters):
        try:
            table = await self.get_table()
            response = await table.put_item(
                {
                    'user_id': user_id,
                    'range': run_id,
                    **run_parameters
                }
            )
            return response
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

    async def handle(self, packet):
        await self.store_run(self.user_id, self.assistant_id, self.run_id, packet)
