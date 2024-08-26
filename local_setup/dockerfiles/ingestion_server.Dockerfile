FROM python:3.10.13-slim

WORKDIR /app
COPY ./requirements.txt /app
COPY ./ingestion_server /app/ingestion_server

RUN apt-get update && apt-get install -y --no-install-recommends gcc
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

WORKDIR /app/ingestion_server

EXPOSE 8000

CMD ["uvicorn", "quickstart_ingestion_server:app", "--host", "0.0.0.0", "--port", "8000"]