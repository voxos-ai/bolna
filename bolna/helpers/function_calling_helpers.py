import json
import requests
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


async def trigger_api(url, method, param, api_token,  **kwargs):
        try:
            code = compile(param % kwargs, "<string>", "exec")
            exec(code, globals(), kwargs)
            req = param % kwargs
            logger.info(f"Params {param % kwargs} \n {type(req)} \n {param} \n {kwargs} \n\n {req}")

            headers = {'Content-Type': 'application/json'}
            if api_token:
                headers = {'Content-Type': 'application/json', 'Authorization': api_token}
            if method == "get":
                logger.info(f"Sending request {req}, {url}, {headers}")
                response = requests.get(url, params=json.loads(req), headers=headers)
                logger.info(f"Response from The servers {response.text}")
                return response.text
            elif method == "post":
                logger.info(f"Sending request {json.loads(req)}, {url}, {headers}")
                response = requests.post(url, json=json.loads(req), headers=headers)
                logger.info(f"Response from The server {response.text}")
                return response.text
        except Exception as e:
            message = str(f"We send {method} request to {url} & it returned us this error:", e)
            logger.error(message)
            return message