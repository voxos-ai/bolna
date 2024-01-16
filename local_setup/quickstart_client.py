import asyncio
import pyaudio
import websockets
import logging
import base64
import json
import subprocess
import queue
import wave
import time
import argparse
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
# Argument parsing
parser = argparse.ArgumentParser(description="Client for WebSocket communication")
parser.add_argument('--run_mode', type=str, default="tts", choices=["tts", "e2e", "asr"], help="Choose between 'tts', 'asr' and 'e2e'")
args = parser.parse_args()

connection_type = args.run_mode

# Set up logging
logging.basicConfig(level=logging.INFO)

# Audio settings
format = pyaudio.paInt16
channels = 1
rate = 16000
chunk = 8000
audio = pyaudio.PyAudio()
start_time = time.time()
chunks = []
interruption_message = 0

# WebSocket server address based on connection type
if connection_type == "e2e":
    server_url = os.getenv("BOLNA_WS_SERVER_URL")
    assistant_id = os.getenv("ASSISTANT_ID")
    uri = f"{server_url}/chat/v1/{assistant_id}"
    logging.info(f"Connecting to {uri}")
elif connection_type == "tts":
    tts_url = os.getenv("TTS_WS_SERVER_URL")
    uri = f"{tts_url}/generate"
elif connection_type == "asr":
    asr_url = os.getenv("ASR_WS_SERVER_URL")
    uri = f"{asr_url}/transcribe"
# Audio queue to store audio frames
input_queue = asyncio.Queue() 

# Callback function to add audio frames to the queue
def _callback(input_data, frame_count, time_info, status_flags):
    input_queue.put_nowait(input_data)
    logging.debug(f"Audio frame added to the queue")
    return (input_data, pyaudio.paContinue)

# Coroutine to open microphone stream and start sending audio
async def microphone():
    if connection_type == "asr":
        global format
        format= pyaudio.paFloat32 #Because whisper needs that
    print("Starting microphone")
    stream = audio.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
        stream_callback=_callback
    )
        
    stream.start_stream()
    print("Listening")
    while stream.is_active():
        await asyncio.sleep(0.1)
                
    stream.stop_stream()
    stream.close()

async def emitter(ws):
    while True:
        audio_frame = await input_queue.get()
        if connection_type == "tts":
            data = json.dumps({"text": audio_frame, "model": "xtts", "voice": "rohan", "language": "en"})
        elif connection_type == "asr":
            data = audio_frame
        else:
            base64_audio_frame = base64.b64encode(audio_frame).decode('utf-8')
            data = json.dumps({"type": "audio", "data": base64_audio_frame})
        
        global start_time 
        start_time = time.time()
        await ws.send(data)
        logging.debug("Audio frame sent to WebSocket server")

def check_if_wav(audio_data):
    header = audio_data[:4]
    if header == b'RIFF':
        print("This might be a WAV file.")
        return True

def check_if_mp3(data):
    if data[:3] == b'ID3':
        return True
    if len(data) < 2:
        return False
    first_two_bytes = data[0] << 8 | data[1]  # Combine the first two bytes
    return (first_two_bytes & 0xFFE0) == 0xFFE0


async def play_audio():
    while True:
        try:
            global chunks
            if len(chunks) > 0:
                chunk = chunks.pop(0)
                if connection_type == "e2e":
                    audio = base64.b64decode(chunk)
                else:
                    audio = chunk
                print(f"Adjusted audio length: {len(audio)}")  
                if len(audio) % 2 != 0:
                    print(f"Audio chunk length is odd: {len(audio)}")
                    audio = audio[:-1]
                    
                print(f"Adjusted audio length: {len(audio)}")  
                audio_data = np.frombuffer(audio, dtype=np.int16)
                print("got audio data")
                sd.play(audio_data, 16000)
                sd.wait()
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in playing audio: {e}")


async def receiver(ws):
    global chunks
    chunk_num = 0
    prev_chunk_time = None

    while True:
        response = await ws.recv()

        if connection_type == "asr":
            response = json.loads(response)
            print(f"response {response['channel']['alternatives'][0]['transcript']}")
            if response['speech_final']:
                print("Speech final is true")
            continue
        if chunk_num == 0:
                global start_time
                print(f"rcvd chunk {chunk_num} in {time.time() - start_time}")
                prev_chunk_time = time.time()
                chunk_num +=1
        else:
            print(f"rcvd chunk {chunk_num} in {time.time() - prev_chunk_time}")
            prev_chunk_time = time.time()
            chunk_num +=1
        if connection_type == "tts":
            b64_audio = response
            if response == b'\x00':
                print("saving ")
                audio_data = b''.join(chunks)
                try:
                    with wave.open("./output.wav", 'wb') as wf:
                        wf.setnchannels(1)  # mono
                        wf.setsampwidth(2) 
                        wf.setframerate(24000)
                        wf.writeframes(audio_data)
                except Exception as e:
                    print(f"Some thing went wrong in saving{e}")
                continue
        if connection_type == "e2e":
            response = json.loads(response)
            logging.info(f"{response.keys()}")
            if response["type"] != "audio":
                print(response)
            if response["type"] == "clear":
                global interruption_message
                logging.info(f"Got interrupt message and clearing chunks {interruption_message}")
                chunks.clear()
                interruption_message +=1
                continue
            if response["data"] is None:
                continue
            b64_audio = response["data"]
        chunks.append(b64_audio)

async def text_input():
    # while True:
    #     user_input = input("Enter text: ")
    #     input_queue.put_nowait(user_input)
    #     await asyncio.sleep(1.5)

    messages = ["Hello, may I", "Know who am I", "speaking with?"]
    for message in messages:
        input_queue.put_nowait(message)
        await asyncio.sleep(2)

async def interface():
    if connection_type == "tts":
        await text_input()
    else:
        print("asr it is")
        await microphone()

async def main():
    async with websockets.connect(uri, open_timeout=None) as ws:
        print("NOW IN THE GATHER PART")
        tasks = [interface(), emitter(ws), receiver(ws)]
        if connection_type == "tts" or connection_type == "e2e":
            tasks.append(play_audio())
        await asyncio.gather(*tasks)

print("Starting with the loop")
# Run the asyncio event loop
asyncio.run(main())