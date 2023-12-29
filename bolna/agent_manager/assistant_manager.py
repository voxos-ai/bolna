import asyncio
import uvloop
import time
import os
from twilio.rest import Client
import requests
import tiktoken
from .base_manager import BaseManager
from .task_manager import TaskManager

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)
enc = tiktoken.get_encoding("cl100k_base")


class AssistantManager(BaseManager):
    def __init__(self, agent_config, ws, context_data=None, user_id=None, assistant_id=None,
                 connected_through_dashboard=None, log_dir_name=None):
        super().__init__(log_dir_name=log_dir_name)
        self.log_dir_name = log_dir_name
        self.tools = {}
        self.websocket = ws
        self.agent_config = agent_config
        self.context_data = context_data
        self.tasks = agent_config.get('tasks', [])
        self.task_states = [False] * len(self.tasks)
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.run_id = f"{self.assistant_id}#{str(int(time.time() * 1000))}"  # multiply by 1000 to get timestamp in nano seconds to reduce probability of collisions in simultaneously triggered runs.
        self.connected_through_dashboard = connected_through_dashboard

    async def run(self, is_local=False):
        '''
        Run will start all tasks in sequential format
        '''
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        input_parameters = None
        for task_id, task in enumerate(self.tasks):
            task_manager = TaskManager(self.agent_config["assistant_name"], task_id, task, self.websocket,
                                       context_data=self.context_data, input_parameters=input_parameters,
                                       user_id=self.user_id, assistant_id=self.assistant_id, run_id=self.run_id,
                                       connected_through_dashboard=self.connected_through_dashboard,
                                       log_dir_name=self.log_dir_name)
            await task_manager.load_prompt(self.agent_config["assistant_name"], task_id, is_local=is_local)
            task_output = await task_manager.run()
            task_output['run_id'] = self.run_id
            yield task_id, task_output
            self.task_states[task_id] = True
            input_parameters = task_output
        self.logger.info("Done with execution of the agent")
