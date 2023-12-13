import json
import asyncio
import re
import numpy as np
import copy
import hashlib
from .logger_config import configure_logger
from agents.constants import PREPROCESS_DIR

logger = configure_logger(__name__)


def load_file(file_path, is_json=False):
    data = None
    with open(file_path, "r") as f:
        if is_json:
            data = json.load(f)
        else:
            data = f.read()

    return data


def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def create_ws_data_packet(data, meta_info=None, is_md5_hash=False, llm_generated=False):
    metadata = copy.deepcopy(meta_info)
    metadata["is_md5_hash"] = is_md5_hash
    metadata["llm_generated"] = llm_generated
    return {
        'data': data,
        'meta_info': metadata
    }


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def float2int(sound):
    sound = np.int16(sound * 32767)
    return sound


def mu_law_encode(audio, quantization_channels=256):
    mu = quantization_channels - 1
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def raw_to_mulaw(raw_bytes):
    # Convert bytes to numpy array of int16 values
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    samples = samples.astype(np.float32) / (2 ** 15)
    mulaw_encoded = mu_law_encode(samples)
    return mulaw_encoded


def get_raw_audio_bytes_from_base64(agent_name, b64_string, audio_format='mp3'):
    # we are already storing pcm formatted audio in the filler config. No need to encode/decode them further
    audio_data = None
    logger.info(f"getting audio from base64 string {b64_string}")
    file_name = f"{PREPROCESS_DIR}/{agent_name}/{audio_format}/{b64_string}.{audio_format}"
    with open(file_name, 'rb') as file:
        # Read the entire file content into a variable
        audio_data = file.read()
    return audio_data


def get_md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def is_valid_md5(hash_string):
    return bool(re.fullmatch(r"[0-9a-f]{32}", hash_string))


def split_payload(payload, max_size=500 * 1024):
    if len(payload) <= max_size:
        return payload
    return [payload[i:i + max_size] for i in range(0, len(payload), max_size)]


def get_required_input_types(task):
    input_types = dict()
    for i, chain in enumerate(task['toolchain']['sequences']):
        first_model = chain[0]
        if chain[0] == "transcriber":
            input_types["audio"] = i
        elif chain[0] == "synthesizer" or chain[0] == "llm":
            input_types["text"] = i
    return input_types


def format_messages(messages):
    formatted_string = ""
    for message in messages:
        role = message['role']
        content = message['content']

        if role == 'assistant':
            formatted_string += "assistant: " + content + "\n"
        elif role == 'user':
            formatted_string += "user: " + content + "\n"

    return formatted_string


def update_prompt_with_context(prompt, context_data):
    return prompt.format(**context_data.get('recipient_data', {}))


def get_prompt_responses(agent_name):
    filepath = f"{PREPROCESS_DIR}/{agent_name}/conversation_details.json"
    data = load_file(filepath, True)
    return data


async def execute_tasks_in_chunks(tasks, chunk_size=10):
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    for chunk in task_chunks:
        await asyncio.gather(*chunk)


def has_placeholders(s):
    return bool(re.search(r'\{[^{}\s]*\}', s))
