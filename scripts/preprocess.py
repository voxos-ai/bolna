import os
import sys
import asyncio
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.helpers.utils import get_md5_hash, load_file, write_json_file, execute_tasks_in_chunks, has_placeholders
from agents.models import SUPPORTED_SYNTHESIZER_MODELS
from agents.constants import PREPROCESS_DIR, USERS_KEY_ORDER


async def update_users_json(agent_name):
    users_file_path = f"{PREPROCESS_DIR}/{agent_name}/users.json"
    data = load_file(users_file_path, True)

    for key, value in data.items():
        value['audio'] = get_md5_hash(' '.join([value.get(k, '') for k in USERS_KEY_ORDER]))

    write_json_file(users_file_path, data)
    return data


async def update_prompt_responses_json(agent_name):
    agent_filepath = f"{PREPROCESS_DIR}/{agent_name}/conversation_details.json"
    data = load_file(agent_filepath, True)

    # Function to recursively update the dictionary
    def update_content_array(content):
        for item in content:
            if 'text' in item and not has_placeholders(item['text']):
                item['audio'] = get_md5_hash(item['text'])

    # Iterate through the main dictionary
    for task_key, task_value in data.items():
        for item_key, item_value in task_value.items():
            if 'content' in item_value:
                update_content_array(item_value['content'])

    # Write the updated data back to the file
    write_json_file(agent_filepath, data)
    return data


def get_unique_audio_hash(json_data, audio_values, users_data):
    for task, task_data in json_data.items():
        if isinstance(task_data, dict):
            content = task_data.get("content", [])
            is_root = task_data.get("is_root", False)
            for content_row in content:
                text_value = content_row.get("text")
                audio_value = content_row.get("audio", None)

                if text_value and audio_value:
                    if is_root:
                        for user_id, user_data in users_data.items():
                            user_formatted_text_value = text_value.format(' '.join([user_data.get(k, '') for k in USERS_KEY_ORDER]))

                            # checking if substitution is there in the conversation response file
                            if user_formatted_text_value != text_value:
                                audio_values.append(
                                    {"text": user_formatted_text_value, "audio": user_data.get('audio')})
                    else:
                        audio_values.append({"text": text_value, "audio": audio_value})


async def process_audio_data(agent_name, audio_data, synth):
    synth_instance = None
    if synth is not None and synth in SUPPORTED_SYNTHESIZER_MODELS.keys():
        synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(synth)
        synth_instance = synthesizer_class('polly', 'pcm', 'Kajal', 'en', '8000')

    # create directory
    agent_directory = get_agent_directory(agent_name, synth_instance.format)
    create_directory(agent_directory)

    # generate audio data
    tasks = []
    for audio_data_row in audio_data:
        tasks.append(generate_and_save_audio(synth_instance, audio_data_row, agent_directory))

    await execute_tasks_in_chunks(tasks, 20)


async def generate_and_save_audio(synth_instance, audio_data_row, agent_directory):
    audio_chunk = await synth_instance.generate(audio_data_row.get('text'))
    with open('{}/{}.{}'.format(agent_directory, audio_data_row.get('audio'), synth_instance.format), 'wb') as f:
        f.write(audio_chunk)


def get_agent_directory(agent_name, audio_format):
    return '{}/{}/{}'.format(PREPROCESS_DIR, agent_name, audio_format)


def create_directory(agent_directory):
    if not os.path.exists(agent_directory):
        os.makedirs(agent_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate preprocessed audio data')
    parser.add_argument('agent_name', type=str, help='Name of the agent')
    parser.add_argument('synthesizer', type=str, help='Chosen Synthesizer model',
                        choices=SUPPORTED_SYNTHESIZER_MODELS.keys())

    args = parser.parse_args()
    agent_name, synth = args.agent_name, args.synthesizer

    # inject md5sum for the audio texts
    loop = asyncio.get_event_loop()
    prompt_data, users_data = loop.run_until_complete(
        asyncio.gather(
            update_prompt_responses_json(agent_name),
            update_users_json(agent_name)
        )
    )
    loop.close()

    audio_text_data = []
    for key in prompt_data:
        get_unique_audio_hash(prompt_data[key], audio_text_data, users_data)

    asyncio.run(process_audio_data(agent_name, audio_text_data, synth))
