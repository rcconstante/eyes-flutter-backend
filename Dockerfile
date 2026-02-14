FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python tensorboard 2>/dev/null || true \
    && find /usr/local/lib/python3.11/site-packages -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11/site-packages -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && rm -rf /usr/local/lib/python3.11/site-packages/torch/test \
    && rm -rf /usr/local/lib/python3.11/site-packages/tensorflow/include \
    && rm -rf /root/.cache

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE ${PORT}

CMD /bin/sh -c "gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --timeout 120"
