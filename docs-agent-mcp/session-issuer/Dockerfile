FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/kubeflow/docs-agent"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY issuer_core.py main.py .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
