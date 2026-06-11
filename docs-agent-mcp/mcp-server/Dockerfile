FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
