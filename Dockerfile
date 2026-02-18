# Start from a slim Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app.py .
COPY model.pkl .

# Cloud Run sets PORT env var; default to 8080
ENV PORT=8080

# Run with Gunicorn (production WSGI server)
# - Workers: 1 is fine for Cloud Run (it scales by adding containers, not threads)
# - Timeout: 120s for ML models that may take time
CMD exec gunicorn --bind :$PORT --workers 1 --timeout 120 app:app