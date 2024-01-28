FROM python:3.10.13-slim

WORKDIR /app
COPY ./local_setup/requirements.txt /app
COPY ./local_setup/demo_server.py /app
COPY . /app


RUN apt-get update && apt-get install libgomp1 git -y
RUN pip install -r requirements.txt
RUN pip install --force-reinstall /app

EXPOSE 5001
CMD ["uvicorn", "demo_server:app", "--host", "0.0.0.0", "--port", "5001"]
