FROM python:3.10.13-slim

WORKDIR /app
COPY ./agents /app/agents
COPY ./requirements.txt /app
COPY ./server.py /app

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install libgomp1 -y

EXPOSE 5001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5001"]
