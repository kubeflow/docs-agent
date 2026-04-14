FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy dependency definition and install with uv
COPY pyproject.toml ./
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r pyproject.toml

# Environment variables
ENV PORT=8000

# Switch to non-root before running the app
USER appuser

# App
COPY app.py /app/

EXPOSE 8000
CMD ["python", "-u", "app.py"]
