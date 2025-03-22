FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV, YOLO, and wget
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget

# Copy requirements
COPY requirements.lpr.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.lpr.txt

# Copy source code
COPY lpr_api.py .

# Create directories
RUN mkdir -p /app/models /app/data

# Download YOLOv8n model
RUN wget -O /app/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# Expose port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "uvicorn", "lpr_api:app", "--host", "0.0.0.0", "--port", "8000"]