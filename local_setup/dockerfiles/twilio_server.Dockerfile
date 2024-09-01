FROM python:3.10.13-slim

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=telephony_server/requirements.txt,target=/app/requirements.txt \
    pip install -r requirements.txt

COPY telephony_server/twilio_api_server.py /app/

EXPOSE 8001

CMD ["uvicorn", "twilio_api_server:app", "--host", "0.0.0.0", "--port", "8001"]