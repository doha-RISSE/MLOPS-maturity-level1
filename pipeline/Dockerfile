FROM python:3.11-slim

WORKDIR /app
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip install --no-cache-dir -r /app/pipeline/requirements.txt
COPY common /app/common
COPY pipeline /app/pipeline
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "pipeline.run_pipeline"]
