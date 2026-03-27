FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install server dependencies
COPY server-https/requirements.txt ./requirements-server.txt
RUN pip install --no-cache-dir -r requirements-server.txt

# Install ingestion dependencies
RUN pip install --no-cache-dir requests langchain-text-splitters

# Environment variables
ENV PORT=8000

# Switch to non-root before running the app
USER appuser

# Copy server and ingestion script
COPY server-https/app.py /app/
COPY scripts/local_ingest.py /app/

EXPOSE 8000
CMD ["python", "-u", "app.py"]
