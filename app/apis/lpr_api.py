from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import io
from PIL import Image
import time
import os
import base64
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import logging 
# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# Setup tracing
resource = Resource(attributes={SERVICE_NAME: "lpr-api"})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Custom metrics
PLATE_DETECTION_COUNT = Counter(
    "lpr_plate_detection_count", 
    "Count of license plates detected",
    ["success"]  # Label to track successful vs unsuccessful detections
)

DETECTION_TIME = Histogram(
    "lpr_detection_time_seconds",
    "Time spent processing images",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

app = FastAPI(title="License Plate Recognition API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
model_path = os.environ.get('MODEL_PATH', 'yolov8n.pt')
model = YOLO(model_path)
reader = easyocr.Reader(['en'])

# Configure static file directories
DATASET_PATH = os.environ.get("DATASET_PATH", "/app/dataset/images")
STATIC_PATH = "/app/static"

logger.info(f"Dataset path: {DATASET_PATH}")
logger.info(f"Static path: {STATIC_PATH}")

# Ensure static directory exists
if not os.path.exists(STATIC_PATH):
    os.makedirs(STATIC_PATH, exist_ok=True)
    logger.info(f"Created static directory: {STATIC_PATH}")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
    logger.info(f"Successfully mounted static directory: {STATIC_PATH}")
except Exception as e:
    logger.error(f"Failed to mount static directory: {e}")

@app.post("/detect_plate/")
async def detect_plate(file: UploadFile = File(...), request: Request = None):
    with tracer.start_as_current_span("detect_plate") as span:
        start_time = time.time()
        
        try:
            # Read image
            with tracer.start_as_current_span("read_image"):
                contents = await file.read()
                img = Image.open(io.BytesIO(contents))
                img_array = np.array(img)
                original_img = img_array.copy()
                span.set_attribute("image.shape", str(img_array.shape))
            
            # Detect cars
            with tracer.start_as_current_span("detect_cars"):
                results = model(img_array)
                car_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 2)  # Class 2 is car
                span.set_attribute("car.count", car_count)
            
            plates = []
            
            # Process cars and find plates
            with tracer.start_as_current_span("find_plates"):
                for box in results[0].boxes:
                    if int(box.cls[0]) == 2:  # Class 2 is car
                        # Extract car region
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # Draw car box
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        try:
                            car_img = img_array[y1:y2, x1:x2]
                            
                            # Process for OCR
                            if car_img.size > 0:
                                # Convert to grayscale if needed
                                if len(car_img.shape) == 3:
                                    gray = cv2.cvtColor(car_img, cv2.COLOR_RGB2GRAY)
                                else:
                                    gray = car_img
                                
                                gray = cv2.equalizeHist(gray)
                                
                                plate_texts = reader.readtext(gray)
                                
                                # Filter likely plates
                                for (bbox, text, prob) in plate_texts:
                                    if len(text) >= 5 and prob > 0.5:
                                        plates.append({
                                            "text": text,
                                            "confidence": float(prob)
                                        })
                                        # Draw license plate text on image
                                        cv2.putText(original_img, f"Plate: {text}", 
                                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.5, (0, 255, 0), 2)
                                        break  # Take first good plate per car
                        except Exception as e:
                            print(f"Error processing car region: {e}")
            
            # Convert processed image to base64 for response
            _, buffer = cv2.imencode('.jpg', original_img)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            # Record metrics
            detection_time = time.time() - start_time
            DETECTION_TIME.observe(detection_time)
            
            if plates:
                PLATE_DETECTION_COUNT.labels(success="true").inc(len(plates))
                span.set_attribute("plate.count", len(plates))
                span.set_attribute("plate.text", plates[0]["text"] if plates else "")
            else:
                PLATE_DETECTION_COUNT.labels(success="false").inc()
                span.set_attribute("plate.count", 0)
            
            return {
                "plates": plates, 
                "processing_time": detection_time,
                "processed_image": processed_image
            }
            
        except Exception as e:
            # Log the error
            print(f"Error in detect_plate: {e}")
            return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "License Plate Recognition API"}

# Add random images endpoint
import random
from typing import List
import uuid

@app.get("/random_images/", response_model=List[str])
async def get_random_images(count: int = 30):
    """Return paths to random images from the dataset"""
    try:
        # Log the dataset path being used
        logger.info(f"Looking for images in: {DATASET_PATH}")
        
        # Check if directory exists
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset path does not exist: {DATASET_PATH}")
            return {"error": f"Dataset path does not exist: {DATASET_PATH}"}
            
        # List all files in the images directory
        image_files = []
        for root, dirs, files in os.walk(DATASET_PATH):
            logger.info(f"Scanning directory: {root}, found {len(files)} files")
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Get relative path from DATASET_PATH
                    rel_path = os.path.relpath(os.path.join(root, file), DATASET_PATH)
                    image_files.append(rel_path)
        
        logger.info(f"Total eligible images found: {len(image_files)}")
        
        # Select random images
        if len(image_files) == 0:
            logger.warning("No images found in dataset")
            return {"error": "No images found in dataset"}
        
        selected_images = random.sample(image_files, min(count, len(image_files)))
        logger.info(f"Selected {len(selected_images)} random images")
        
        # Copy selected images to static directory for serving
        static_urls = []
        for img_path in selected_images:
            src_path = os.path.join(DATASET_PATH, img_path)
            # Create a unique filename
            unique_name = f"{uuid.uuid4()}{os.path.splitext(img_path)[1]}"
            dst_path = os.path.join(STATIC_PATH, unique_name)
            
            try:
                # Copy the file
                with open(src_path, 'rb') as src_file:
                    with open(dst_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                static_urls.append(f"/static/{unique_name}")
            except Exception as e:
                logger.error(f"Error copying image {src_path}: {e}")
        
        # Return the static URLs
        logger.info(f"Returning {len(static_urls)} image URLs")
        return static_urls
    
    except Exception as e:
        logger.exception(f"Error retrieving random images: {e}")
        return {"error": str(e)}

# Add test images endpoint for fallback
import uuid
from fastapi.responses import Response

@app.get("/test_image/{size}")
async def get_test_image(size: str = "200x150"):
    """Generate a test image for debugging"""
    try:
        # Parse size
        width, height = map(int, size.split('x'))
        
        # Create a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Create a black image
        img = np.zeros((height, width, 3), np.uint8)
        
        # Fill with random color
        img[:] = color
        
        # Add a number
        number = random.randint(100, 999)
        cv2.putText(img, f"Test {number}", (width//4, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to bytes
        _, img_encoded = cv2.imencode('.jpg', img)
        
        return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
    except Exception as e:
        logger.exception(f"Error generating test image: {e}")
        return {"error": str(e)}

@app.get("/test_images/")
async def get_test_images(count: int = 30):
    """Return a list of test image URLs"""
    try:
        image_urls = [f"/test_image/200x150?id={uuid.uuid4()}" for _ in range(count)]
        return image_urls
    except Exception as e:
        logger.exception(f"Error creating test image URLs: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Return information about the API's health and configuration"""
    return {
        "status": "healthy",
        "dataset_path": DATASET_PATH,
        "dataset_exists": os.path.exists(DATASET_PATH),
        "image_count": sum(1 for root, _, files in os.walk(DATASET_PATH) 
                           for file in files 
                           if file.lower().endswith(('.png', '.jpg', '.jpeg')))
    }

# Add this endpoint
@app.get("/dataset-images/{image_name}")
async def get_dataset_image(image_name: str):
    # Define the path to your dataset images
    dataset_path = os.environ.get("DATASET_PATH", "/app/data/images")
    image_path = os.path.join(dataset_path, image_name)
    
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "Image not found"}

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)