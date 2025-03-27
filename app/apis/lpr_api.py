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
import re
from prometheus_client import generate_latest
from fastapi.responses import Response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
from prometheus_client import Counter, Histogram, Summary
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# Image processing metrics
IMAGE_PROCESSING_TIME = Summary(
    'lpr_image_processing_seconds',
    'Time spent processing each image'
)

LICENSE_PLATE_SEARCH = Counter(
    'lpr_license_plate_search_total',
    'Number of license plate searches',
    ['status']  # 'success', 'failure'
)

PLATE_DETECTION = Counter(
    'lpr_plate_detection_total',
    'Number of license plates detected',
    ['result']  # 'detected', 'not_detected'
)

PROCESSING_DURATION = Histogram(
    'lpr_processing_duration_seconds',
    'Time spent processing images',
    buckets=[.1, .5, 1.0, 2.0, 5.0, 10.0]
)

# Setup tracing
resource = Resource(attributes={SERVICE_NAME: "lpr-api"})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint="eog-otel-collector-1:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

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
async def detect_plate(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        # Read uploaded file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info(f"Original image shape: {img.shape} with dtype {img.dtype}")

        # Convert from RGBA to RGB if needed
        if img.shape[2] == 4:
            logger.info("Converting image from RGBA to RGB")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            logger.info(f"After conversion image shape: {img.shape}")
        
        # Keep a copy of the original image for drawing
        marked_img = img.copy()
        
        # Run YOLO detection
        results = model(img, verbose=False)
        logger.info(f"YOLO results obtained: {len(results)} with {len(results[0].boxes)} detected objects")
        
        detected_plates = []
        
        # Process each detected car
        for box in results[0].boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            if (results[0].names[class_id] in ['car', 'truck']) and conf > 0.3:
                # Get car coordinates
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                
                # Extract car region
                car_region = img[y1:y2, x1:x2]
                if car_region.size == 0:
                    continue
                
                # Draw car region
                cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Enhance image for better plate detection
                gray = cv2.cvtColor(car_region, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                # Use EasyOCR to find text in car region
                texts = reader.readtext(gray)
                
                # Filter for likely license plate text
                for (bbox, text, prob) in texts:
                    if len(text) >= 5 and prob > 0.3:
                        # Calculate absolute positions for drawing
                        (top_left, _, bottom_right, _) = bbox
                        tx1, ty1 = map(int, top_left)
                        tx3, ty3 = map(int, bottom_right)
                        
                        # Calculate absolute positions in original image
                        abs_tx1, abs_ty1 = x1 + tx1, y1 + ty1
                        abs_tx3, abs_ty3 = x1 + tx3, y1 + ty3
                        
                        # Draw text region
                        cv2.rectangle(marked_img, (abs_tx1, abs_ty1), (abs_tx3, abs_ty3), (0, 255, 0), 2)
                        cv2.putText(marked_img, text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        detected_plates.append({
                            "text": text,
                            "confidence": float(prob)
                        })
        
        # If no plates found, try processing the entire image
        if not detected_plates:
            logger.info("No plates found in car regions, trying full image")
            # Process the full image
            full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            full_gray = cv2.equalizeHist(full_gray)
            texts = reader.readtext(full_gray)
            
            # Filter for likely license plate text
            for (bbox, text, prob) in texts:
                if len(text) >= 5 and prob > 0.3:
                    # Calculate positions for drawing
                    (top_left, _, bottom_right, _) = bbox
                    tx1, ty1 = map(int, top_left)
                    tx3, ty3 = map(int, bottom_right)
                    
                    # Draw text region
                    cv2.rectangle(marked_img, (tx1, ty1), (tx3, ty3), (0, 0, 255), 2)  # Red box for full-image detection
                    cv2.putText(marked_img, text, (tx1, ty1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    detected_plates.append({
                        "text": text,
                        "confidence": float(prob)
                    })
        
        # Clean and format the detected plate text
        if detected_plates:
            plate = detected_plates[0]
            cleaned_text = re.sub(r'[^A-Z0-9]', '', plate["text"].upper())
            detected_plates = [{"text": cleaned_text, "confidence": plate["confidence"]}]
        else:
            detected_plates = []

        # Convert image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        logger.info(f"Text detection completed in {processing_time:.2f} seconds")
        logger.info(f"Number of text regions detected: {len(detected_plates)}")
        
        # Update metrics
        IMAGE_PROCESSING_TIME.observe(processing_time)  # Manual timing observation
        
        # These stay the same
        if len(results[0].boxes) > 0:
            LICENSE_PLATE_SEARCH.labels(status='success').inc()
        else:
            LICENSE_PLATE_SEARCH.labels(status='failure').inc()

        if detected_plates:
            PLATE_DETECTION.labels(result='detected').inc()
        else:
            PLATE_DETECTION.labels(result='not_detected').inc()

        PROCESSING_DURATION.observe(processing_time)

        return {
            "plates": detected_plates,
            "processing_time": processing_time,
            "processed_image": img_str
        }
        
    except Exception as e:
        LICENSE_PLATE_SEARCH.labels(status='failure').inc()
        logger.error(f"Error in detect_plate: {str(e)}")
        try:
            if 'marked_img' in locals():
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(buffer).decode('utf-8')
                return {
                    "error": str(e),
                    "processed_image": img_str
                }
        except:
            pass
            
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "License Plate Recognition API"}

@app.get("/health")
def health_check():
    """Health check endpoint that also verifies the dataset path"""
    try:
        # Check if dataset path exists
        dataset_exists = os.path.exists(DATASET_PATH)
        
        # Count images in the dataset
        image_count = 0
        if dataset_exists:
            image_files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_count = len(image_files)
        
        # Check if we can load the model
        model_status = "Available" if 'model' in globals() and model is not None else "Not available"
        
        # Check if OCR reader is available
        ocr_status = "Available" if 'reader' in globals() and reader is not None else "Not available"
        
        return {
            "status": "healthy",
            "dataset_path": DATASET_PATH,
            "dataset_exists": dataset_exists,
            "image_count": image_count,
            "model_status": model_status,
            "ocr_status": ocr_status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

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

@app.get("/dataset-images/{image_name}")
async def get_dataset_image(image_name: str):
    # Define the path to your dataset images
    dataset_path = os.environ.get("DATASET_PATH", "/app/data/images")
    image_path = os.path.join(dataset_path, image_name)
    
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "Image not found"}

# First, instrument the app
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# Create the Instrumentator but DON'T call expose yet
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],  # Important to exclude metrics endpoint
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# Instrument the app
instrumentator.instrument(app)

# NOW define the metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )