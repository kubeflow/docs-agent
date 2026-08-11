FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embeddings_client.py rag_collections.py server.py smoke_tools.py .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
