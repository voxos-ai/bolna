import os
from agents.database.dynamodb import DynamoDB
from dotenv import load_dotenv
from agents.helpers.logger_config import configure_logger
from .default import DefaultOutputHandler

logger = configure_logger(__name__)
load_dotenv()


class DatabaseOutputHandler(DefaultOutputHandler):
    def __init__(self, user_id, assistant_id, run_id):
        super().__init__()
        self.dynamodb = DynamoDB(os.getenv('TABLE_NAME'))
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.run_id = run_id

    async def handle(self, packet):
        await self.dynamodb.store_run(self.user_id, self.assistant_id, self.run_id, packet)
