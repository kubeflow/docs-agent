FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY citations.py compressor.py embeddings_client.py flo_tracing.py intent_router.py milvus_search.py rag_collections.py reranker_client.py server.py smoke_tools.py .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
