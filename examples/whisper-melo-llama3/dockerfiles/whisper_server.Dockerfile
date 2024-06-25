FROM python:3.10.13-slim

RUN apt-get update && apt-get install libgomp1 git -y
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg
RUN apt-get -y install build-essential
RUN apt-get -y install portaudio19-dev
RUN git clone https://github.com/bolna-ai/streaming-whisper-server.git
WORKDIR streaming-whisper-server
RUN pip install -e .
RUN pip install git+https://github.com/SYSTRAN/faster-whisper.git
RUN pip install transformers

RUN ct2-transformers-converter --model openai/whisper-small --copy_files preprocessor_config.json --output_dir ./Server/ASR/whisper_small --quantization float16
WORKDIR Server
EXPOSE 9000
CMD ["python3", "Server.py", "-p", "9000"]
