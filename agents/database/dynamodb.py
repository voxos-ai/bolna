import boto3
from botocore.exceptions import ClientError
from agents.database.base_database import BaseDatabase
from agents.helpers.logger_config import configure_logger
import uuid

from aiohttp import ClientSession

from aiodynamo.client import Client
from aiodynamo.credentials import Credentials
from aiodynamo.expressions import F
from aiodynamo.http.aiohttp import AIOHTTP
from aiodynamo.models import Throughput, KeySchema, KeySpec, KeyType


logger = configure_logger(__name__)

# Initialize a DynamoDB client
#dynamodb = boto3.resource('dynamodb')


class DynamoDB(BaseDatabase):
    def __init__(self, table_name):
        self.table_name = table_name
        self.client = None
        self.table = None

    async def get_table(self):
        if self.table is None:
            self.session = ClientSession()
            self.client = Client(AIOHTTP(self.session), Credentials.auto(), "us-east-1")
            self.table = self.client.table(self.table_name)

        return self.table

    async def store_agent(self, user_id, sort_key, agent_data):
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
    
#     def update_agent_run(self, user_id, agent_id, run_id, updated_parameters):
#         try:
#             update_expression = "SET "
#             expression_attribute_values = {}
#             for key, value in updated_parameters.items():
#                 update_expression += f"{key} = :{key}, "
#                 expression_attribute_values[f":{key}"] = value

#             update_expression = update_expression.rstrip(", ")

#             response = self.table.update_item(
#                 Key={
#                     'user_id': user_id,
#                     'range': f"{agent_id}#{run_id}"
#                 },
#                 UpdateExpression=update_expression,
#                 ExpressionAttributeValues=expression_attribute_values
#             )
#             return response
#         except ClientError as e:
#             print(f"An error occurred: {e}")
#             return None

# def update_agent(self, user_id, sort_key, updated_parameters):
#         try:
#             update_expression = "SET "
#             expression_attribute_values = {}
#             for key, value in updated_parameters.items():
#                 update_expression += f"{key} = :{key}, "
#                 expression_attribute_values[f":{key}"] = value

#             update_expression = update_expression.rstrip(", ")

#             response = self.table.update_item(
#                 Key={
#                     'user_id': user_id,
#                     'range': sort_key
#                 },
#                 UpdateExpression=update_expression,
#                 ExpressionAttributeValues=expression_attribute_values
#             )
#             return response
#         except ClientError as e:
#             print(f"An error occurred: {e}")
#             return None

# def get_all_agent_runs(self, user_id):
#         try:
#             response = self.table.query(
#                 KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id) & Key('range').begins_with(agent_id)
#             )
#             return response.get('Items', [])
#         except ClientError as e:
#             print(f"An error occurred: {e}")
#             return []

# def get_all_agents(self, user_id):
#         try:
#             response = self.table.query(
#                 KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id) & Key('range').begins_with("agent")

#             )
#             return response.get('Items', [])
#         except ClientError as e:
#             print(f"An error occurred: {e}")
#             return []
