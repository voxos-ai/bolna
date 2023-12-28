FROM python:3.10.13-slim

WORKDIR /app
COPY ./telephony_server/twilio_api_server.py /app

RUN pip install --no-cache-dir fastapi==0.95.1 python-dotenv==1.0.0 twilio==8.9.0 uvicorn==0.22.0 redis==5.0.1

EXPOSE 8001

CMD ["uvicorn", "twilio_api_server:app", "--host", "0.0.0.0", "--port", "8001"]
