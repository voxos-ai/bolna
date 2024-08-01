FROM python:3.10.13-slim

WORKDIR /app
COPY ./requirements.txt /app
COPY ./quickstart_server.py /app

RUN apt-get update && apt-get install libgomp1 git -y
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg
RUN pip install -r requirements.txt
RUN pip install --force-reinstall git+https://github.com/bolna-ai/bolna@master
RUN pip install scipy==1.11.0
RUN pip install torch==2.0.1
RUN pip install torchaudio==2.0.1
RUN pip install pydub==0.25.1
RUN pip install ffprobe
RUN pip install aiofiles

EXPOSE 5001
CMD ["uvicorn", "quickstart_server:app", "--host", "0.0.0.0", "--port", "5001"]
