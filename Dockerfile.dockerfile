FROM python:3.7.5-slim-buster

WORKDIR /app_fastapi

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD fastapi run app.py