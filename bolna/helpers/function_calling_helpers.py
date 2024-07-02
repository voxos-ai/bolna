import json

import aiohttp
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

async def trigger_api(url, method, param, api_token, **kwargs):
    try:
        code = compile(param % kwargs, "<string>", "exec")
        exec(code, globals(), kwargs)
        req = param % kwargs
        logger.info(f"Params {param % kwargs} \n {type(req)} \n {param} \n {kwargs} \n\n {req}")

        headers = {'Content-Type': 'application/json'}
        if api_token:
            headers['Authorization'] = api_token

        async with aiohttp.ClientSession() as session:
            if method.lower() == "get":
                logger.info(f"Sending request {req}, {url}, {headers}")
                async with session.get(url, params=json.loads(req), headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")
                    return response_text
            elif method.lower() == "post":
                logger.info(f"Sending request {json.loads(req)}, {url}, {headers}")
                async with session.post(url, json=json.loads(req), headers=headers) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server: {response_text}")
                    return response_text
    except Exception as e:
        message = f"We sent {method} request to {url} & it returned us this error: {e}"
        logger.error(message)
        return message
