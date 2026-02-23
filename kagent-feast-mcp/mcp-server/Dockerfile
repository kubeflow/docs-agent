FROM python:3.10-slim

WORKDIR /app

COPY mcp-server/requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

COPY mcp-server/server.py .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
