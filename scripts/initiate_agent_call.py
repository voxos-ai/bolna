import os
import sys
import argparse
import asyncio
import aiohttp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bolna.helpers.utils import load_file, execute_tasks_in_chunks
from bolna.constants import PREPROCESS_DIR


async def fetch(session, url, payload):
    async with session.post(url, json=payload) as response:
        return await response.text()


async def main(agent_name, agent_id):
    users_file_path = f"{PREPROCESS_DIR}/{agent_name}/users.json"
    users_list = load_file(users_file_path, True)

    tasks = []
    async with aiohttp.ClientSession() as session:
        for user_id, user_data in users_list.items():
            url = 'http://localhost:8001/make_call'
            payload = {
                'agent_id': agent_id,
                'call_details': {"recipient_phone_number": user_id, "recipient_data": user_data}
            }

            tasks.append(fetch(session, url, payload))

        await execute_tasks_in_chunks(tasks, 10)
        # modify code here to check request responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initiate your agent calls')
    parser.add_argument('agent_name', type=str, help='Name of the agent')
    parser.add_argument('agent_uuid', type=str, help='Agent unique ID')

    args = parser.parse_args()
    agent_name, agent_id = args.agent_name, args.agent_uuid

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(agent_name, agent_id))
    loop.close()
