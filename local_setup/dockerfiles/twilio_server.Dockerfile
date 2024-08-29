FROM python:3.10.13-slim

WORKDIR /app
COPY ./requirements.txt /app
COPY ./telephony_server/twilio_api_server.py /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "twilio_api_server:app", "--host", "0.0.0.0", "--port", "8001"]
