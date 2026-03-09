FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time → no download at container start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

ENV PORT=8080

USER appuser
COPY app.py /app/

EXPOSE 8080
CMD ["python", "-u", "app.py"]
