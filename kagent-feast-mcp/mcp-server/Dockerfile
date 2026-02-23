FROM python:3.10-slim

WORKDIR /app

COPY mcp-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp-server/server.py .
COPY feast_repo/ feast_repo/

ENV FEAST_REPO_PATH=/app/feast_repo
ENV EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
ENV PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
