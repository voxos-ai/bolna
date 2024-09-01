FROM python:3.10.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./requirements.txt,target=/app/requirements.txt \
    --mount=type=bind,source=./pyproject.toml,target=/app/pyproject.toml \
    --mount=type=bind,source=./local_setup/requirements.txt,target=/app/req.txt \
    --mount=type=bind,source=./bolna,target=/app/bolna \
    pip install . -r req.txt

COPY local_setup/quickstart_server.py /app/

EXPOSE 5001

CMD ["uvicorn", "quickstart_server:app", "--host", "0.0.0.0", "--port", "5001"]