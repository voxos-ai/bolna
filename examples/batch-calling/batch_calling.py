import asyncio
import os
from dotenv import load_dotenv
import aiohttp

# Load environment variables from .env file
load_dotenv()

# Load from .env
host = "https://api.bolna.dev"
api_key = os.getenv("api_key", None)
agent_id = 'ee153a6c-19f8-3a61-989a-9146a31c7834' #agent_id in which we want to create the batch
file_path = '/path/of/csv/file'
schedule_time = '2024-06-01T04:10:00+05:30'


async def schedule_batch(api_key, agent_id, batch_id=None, scheduled_at=None):
    print("now scheduling batch for batch id : {}".format(batch_id))
    url = f"{host}/batches/schedule"
    headers = {'Authorization': f'Bearer {api_key}'}
    data = {
        'agent_id': agent_id,
        'batch_id': batch_id,
        'scheduled_at': scheduled_at
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                response_data = await response.json()
                if response.status == 200:
                    return response_data
                else:
                    raise Exception(f"Error scheduling batch: {response_data}")
    except aiohttp.ClientError as e:
        print(f"HTTP Client Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


async def get_batch_status(api_key, agent_id, batch_id=None):
    print("now getting batch status for batch id : {}".format(batch_id))
    url = f"{host}/batches/{agent_id}/{batch_id}"
    headers = {'Authorization': f'Bearer {api_key}'}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response_data = await response.json()
                if response.status == 200:
                    return response_data
                else:
                    raise Exception(f"Error getting batch status: {response_data}")
    except aiohttp.ClientError as e:
        print(f"HTTP Client Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


async def get_batch_executions(api_key, agent_id, batch_id=None):
    print("now getting batch executions for batch id : {}".format(batch_id))
    url = f"{host}/batches/{agent_id}/{batch_id}/executions"
    headers = {'Authorization': f'Bearer {api_key}'}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response_data = await response.json()
                if response.status == 200:
                    return response_data
                else:
                    raise Exception(f"Error getting batch executions: {response_data}")
    except aiohttp.ClientError as e:
        print(f"HTTP Client Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


async def create_batch():
    url = f"{host}/batches"
    headers = {'Authorization': f'Bearer {api_key}'}

    with open(file_path, 'rb') as f:
        form_data = aiohttp.FormData()
        form_data.add_field('agent_id', agent_id)
        form_data.add_field('file', f, filename=os.path.basename(file_path), content_type='application/octet-stream')

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form_data) as response:
                response_data = await response.json()
                if response_data.get('state') == 'created':
                    batch_id = response_data.get('batch_id')
                    res = await schedule_batch(api_key, agent_id, batch_id, scheduled_at=schedule_time)
                    if res.get('state') == 'scheduled':
                        check = True
                        while check:
                            # Checking the current status every 1 minute
                            await asyncio.sleep(60)
                            res = await get_batch_status(api_key, agent_id, batch_id)
                            if res.get('status') == 'completed':
                                check = False
                                break
                    if not check:
                        res = await get_batch_executions(api_key, agent_id, batch_id)
                        print(res)
                        return res


if __name__ == "__main__":
    asyncio.run(create_batch())
