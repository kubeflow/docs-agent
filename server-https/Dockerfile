    FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV PORT=8000

# Switch to non-root before running the app
USER appuser

# App
COPY shared/ /app/shared/
COPY app.py /app/

EXPOSE 8000
CMD ["python", "-u", "app.py"]
