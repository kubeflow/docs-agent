FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Writable caches for HF/Transformers/Sentence-Transformers
ENV HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/sentence_transformers \
    XDG_CACHE_HOME=/home/appuser/.cache \
    PORT=8000

# Ensure directories exist and owned by appuser
RUN mkdir -p $HF_HOME $SENTENCE_TRANSFORMERS_HOME $XDG_CACHE_HOME && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root before running the app
USER appuser

# App
COPY shared/ /app/shared/
COPY app.py /app/

EXPOSE 8000
CMD ["python", "-u", "app.py"]
