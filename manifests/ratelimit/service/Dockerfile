# Build this image and push to your registry e.g. <YOUR_DOCKERHUB_USERNAME>/ratelimit-sliding-window:latest
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rate_limiter.py .

# Non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

CMD ["uvicorn", "rate_limiter:app", "--host", "0.0.0.0", "--port", "8080"]
