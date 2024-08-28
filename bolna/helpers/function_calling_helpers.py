import asyncio
import json

import aiohttp
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import convert_to_request_log

logger = configure_logger(__name__)

async def trigger_api(url, method, param, api_token, meta_info, run_id, **kwargs):
    try:
        code = compile(param % kwargs, "<string>", "exec")
        exec(code, globals(), kwargs)
        req = param % kwargs
        logger.info(f"Params {param % kwargs} \n {type(req)} \n {param} \n {kwargs} \n\n {req}")
        
        headers = {'Content-Type': 'application/json'}
        if api_token:
            headers['Authorization'] = api_token

        convert_to_request_log(req, meta_info , None, "function_call", direction = "request", is_cached= False, run_id = run_id)

        logger.info(f"Sleeping for 700 ms to make sure that we do not send the same message multiple times")
        await asyncio.sleep(0.7)

        async with aiohttp.ClientSession() as session:
            if method.lower() == "get":
                logger.info(f"Sending request {req}, {url}, {headers}")
                async with session.get(url, params=json.loads(req), headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")
            elif method.lower() == "post":
                logger.info(f"Sending request {json.loads(req)}, {url}, {headers}")
                async with session.post(url, json=json.loads(req), headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")
            
            return response_text
    except Exception as e:
        message = f"ERROR CALLING API: There was an error calling the API: {e}"
        logger.error(message)
        return message