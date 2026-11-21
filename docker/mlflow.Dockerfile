FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.14.3 sqlalchemy==2.0.32

EXPOSE 5000
