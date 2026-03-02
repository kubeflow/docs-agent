# Kagent-native MCP server for Kubeflow docs-agent.
#
# Build from the repo root:
#   docker build -f agent/Dockerfile -t kubeflow-docs-mcp:latest .
#
# Run locally:
#   docker run -p 8000:8000 \
#     -e MILVUS_HOST=host.docker.internal \
#     kubeflow-docs-mcp:latest

FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/agent/requirements.txt

# Writable caches for HF / Sentence-Transformers model downloads
ENV HF_HOME=/home/appuser/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/sentence_transformers \
    XDG_CACHE_HOME=/home/appuser/.cache \
    PORT=8000

RUN mkdir -p $HF_HOME $SENTENCE_TRANSFORMERS_HOME $XDG_CACHE_HOME && \
    chown -R appuser:appuser /home/appuser

# Pre-download the embedding model so it is baked into the image
# (avoids a cold-start download on first request)
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy application code
COPY shared/ /app/shared/
COPY agent/ /app/agent/

# Switch to non-root
USER appuser

EXPOSE 8000

# Run the MCP server via the package entry point
CMD ["python", "-m", "agent.mcp_server"]
