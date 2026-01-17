FROM python:3.9-slim-bookworm

WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
  --no-install-recommends \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.lpr.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.lpr.txt

# Copy source code
COPY lpr_api.py .

# Create directories
RUN mkdir -p /app/models /app/data

# Expose port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "uvicorn", "lpr_api:app", "--host", "0.0.0.0", "--port", "8000"]
