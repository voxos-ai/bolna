FROM python:3.10.13-slim

WORKDIR /app
COPY ./requirements.txt /app
COPY ./demo_server.py /app

RUN apt-get update && apt-get install libgomp1 git -y
RUN pip install -r requirements.txt
RUN pip install --force-reinstall git+https://github.com/bolna-ai/bolna@master

EXPOSE 5001
CMD ["uvicorn", "demo_server:app", "--host", "0.0.0.0", "--port", "5001"]
