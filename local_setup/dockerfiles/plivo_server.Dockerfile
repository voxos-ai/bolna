FROM python:3.10.13-slim

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=telephony_server/requirements.txt,target=/app/requirements.txt \
    pip install --no-cache-dir -r requirements.txt
COPY telephony_server/plivo_api_server.py /app/

EXPOSE 8002

CMD ["uvicorn", "plivo_api_server:app", "--host", "0.0.0.0", "--port", "8002"]
