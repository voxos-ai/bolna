from botocore.exceptions import ClientError
from .base_database import BaseDatabase
from bolna.helpers.logger_config import configure_logger
import uuid

from aiohttp import ClientSession

from aiodynamo.client import Client
from aiodynamo.credentials import Credentials
from aiodynamo.expressions import F, UpdateExpression, SetAction
from aiodynamo.expressions import Condition
from aiodynamo.expressions import HashKey, RangeKey
from aiodynamo.errors import ItemNotFound
from aiodynamo.http.aiohttp import AIOHTTP
from aiodynamo.models import Throughput, KeySchema, KeySpec, KeyType

logger = configure_logger(__name__)


# Initialize a DynamoDB client
# dynamodb = boto3.resource('dynamodb')


class DynamoDB(BaseDatabase):
    def __init__(self, table_name):
        super().__init__(table_name)
        self.table_name = table_name
        self.client = None
        self.table = None

    async def get_table(self):
        if self.table is None:
            self.session = ClientSession()
            self.client = Client(AIOHTTP(self.session), Credentials.auto(), "us-east-1")
            self.table = self.client.table(self.table_name)

        return self.table

    async def store_agent_data(self, user_id, sort_key, agent_data):
        table = await self.get_table()

        logger.info(f"agent_data {type(agent_data)}  {agent_data}")
        try:
            response = await table.put_item(
                {
                    'user_id': str(user_id),
                    'range': str(sort_key),
                    **agent_data
                }
            )
            return response
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

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

    async def update_run(self, user_id, agent_id, run_id, update_parameters):
        try:
            table = await self.get_table()
            key = {
                'user_id': user_id,
                'range': run_id
            }

            update_expression = UpdateExpression()

            ue = None
            for field_name, value in update_parameters.items():
                if ue is None:
                    ue = F(field_name).set(value)
                else:
                    ue = ue & F(field_name).set(value)

            response = await table.update_item(
                key,
                update_expression = ue
            )
            return response
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

    async def update_agent_status(self, user_id, sort_key, new_status):
        table = await self.get_table()
        try:
            key = {
                'user_id': user_id,
                'range': sort_key
            }
            response = await table.update_item(
                key,
                F("assistant_status").set("processed")
            )
            return response
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

    async def get_all_agents_for_user(self, user_id):
        table = await self.get_table()
        try:
            agents = []
            async for item in table.query(
                    key_condition=HashKey('user_id', user_id) & RangeKey('range').begins_with("agent#")):
                agents.append(item)
            return agents
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None
    
    async def get_all_executions_for_assistant(self, user_id, agent_id):
        table = await self.get_table()
        try:
            executions = []
            async for item in table.query(
                    key_condition=HashKey('user_id', user_id) & RangeKey('range').begins_with(f"{agent_id}#")):
                executions.append(item)
            return executions
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

    async def get_assistant_data(self, user_id, agent_id, analytics = False):
        table = await self.get_table()
        try:
            key = {'user_id': user_id, 'range': f"analytics#{agent_id}" if analytics else agent_id}
            agent_data = await table.get_item(key=key)

            return agent_data if agent_data is not None else None

        except ClientError as e:
            print(f"An error occurred: {e}")
            return None
        except ItemNotFound as e:
            logger.error(f"Could not find item with {user_id}, {agent_id} analytics {analytics}")
            return None
        except Exception as e:
            logger.error(f"Error executing query {user_id}, {agent_id} analytics {analytics}")
            return None