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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# Setup tracing
resource = Resource(attributes={SERVICE_NAME: "lpr-api"})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint="eog-otel-collector-1:4317", insecure=True)
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
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.info(f"After BGR to RGB conversion image shape: {img.shape}")
        
        # Keep a copy of the original image for drawing
        original_img = img.copy()
        marked_img = img.copy()
            
        # Run YOLO on the image to detect cars
        model = YOLO('yolov8n.pt')
        logger.info(f"Model loaded successfully: {type(model)}")
        
        results = model(img, verbose=False)
        logger.info(f"YOLO results obtained: {len(results)} with {len(results[0].boxes)} detected objects")
        
        detected_cars = []
        car_coords = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                if (model.names[class_id] == 'car' or model.names[class_id] == 'truck') and conf > 0.3:
                    logger.info(f"Detected a {model.names[class_id]} with confidence {conf}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    car_coords.append((x1, y1, x2, y2))
                    detected_cars.append(img[y1:y2, x1:x2])
                    cv2.rectangle(marked_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        logger.info(f"Number of cars detected: {len(detected_cars)}")
        
        detected_plates = []
        
        # Process each detected car region
        for i, (car_img, (x1, y1, x2, y2)) in enumerate(zip(detected_cars, car_coords)):
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(car_img, cv2.COLOR_RGB2GRAY)
                
                # Apply multiple preprocessing techniques
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced1 = clahe.apply(gray)
                enhanced2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                enhanced3 = cv2.GaussianBlur(gray, (3, 3), 0)
                enhanced3 = cv2.addWeighted(gray, 1.5, enhanced3, -0.5, 0)
                
                best_detections = {}  # Dictionary to store best confidence detection for each region
                
                # Try OCR on all enhanced versions
                for enhanced in [gray, enhanced1, enhanced2, enhanced3]:
                    plate_texts = reader.readtext(enhanced, min_size=10, text_threshold=0.3, paragraph=False)
                    
                    for (bbox, text, prob) in plate_texts:
                        # Skip very low confidence detections
                        if prob < 0.1:  # 10% minimum confidence threshold
                            continue
                            
                        # Calculate absolute position of the text in original image
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        tx1, ty1 = map(int, top_left)
                        tx3, ty3 = map(int, bottom_right)
                        
                        # Create a region key based on approximate box location
                        region_key = (tx1//10, ty1//10, tx3//10, ty3//10)
                        
                        # Only keep the highest confidence detection for each region
                        if region_key not in best_detections or prob > best_detections[region_key]["confidence"]:
                            best_detections[region_key] = {
                                "text": text,
                                "confidence": float(prob),
                                "bbox": (tx1, ty1, tx3, ty3)
                            }
                
                # Add best detections to results and draw them
                for detection in best_detections.values():
                    detected_plates.append({
                        "text": detection["text"],
                        "confidence": detection["confidence"]
                    })
                    
                    # Draw on image
                    tx1, ty1, tx3, ty3 = detection["bbox"]
                    # Calculate absolute positions in original image
                    abs_tx1, abs_ty1 = x1 + tx1, y1 + ty1
                    abs_tx3, abs_ty3 = x1 + tx3, y1 + ty3
                    
                    # Draw car region
                    cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw text region
                    cv2.rectangle(marked_img, (abs_tx1, abs_ty1), (abs_tx3, abs_ty3), (255, 255, 0), 2)
                    # Add text label
                    cv2.putText(marked_img, detection["text"], (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
            except Exception as car_error:
                logger.error(f"Error processing car {i+1}: {str(car_error)}")
                continue
        
        # If no text found in car regions, try direct detection on the full image
        if not detected_plates:
            logger.info("No text found in car regions, trying direct detection...")
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
                
                # Apply multiple preprocessing techniques
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced1 = clahe.apply(gray)
                enhanced2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                enhanced3 = cv2.GaussianBlur(gray, (3, 3), 0)
                enhanced3 = cv2.addWeighted(gray, 1.5, enhanced3, -0.5, 0)
                
                best_detections = {}  # Dictionary to store best confidence detection for each region
                
                # Try OCR on all enhanced versions
                for enhanced in [gray, enhanced1, enhanced2, enhanced3]:
                    plate_texts = reader.readtext(enhanced, min_size=10, text_threshold=0.3, paragraph=False)
                    
                    for (bbox, text, prob) in plate_texts:
                        # Skip very low confidence detections
                        if prob < 0.1:  # 10% minimum confidence threshold
                            continue
                            
                        # Extract bounding box coordinates
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        x1, y1 = map(int, top_left)
                        x3, y3 = map(int, bottom_right)
                        
                        # Create a region key based on approximate box location
                        region_key = (x1//10, y1//10, x3//10, y3//10)
                        
                        # Only keep the highest confidence detection for each region
                        if region_key not in best_detections or prob > best_detections[region_key]["confidence"]:
                            best_detections[region_key] = {
                                "text": text,
                                "confidence": float(prob),
                                "bbox": (x1, y1, x3, y3)
                            }
                
                # Add best detections to results and draw them
                for detection in best_detections.values():
                    detected_plates.append({
                        "text": detection["text"],
                        "confidence": detection["confidence"]
                    })
                    
                    # Draw on image
                    x1, y1, x3, y3 = detection["bbox"]
                    cv2.rectangle(marked_img, (x1, y1), (x3, y3), (0, 255, 0), 2)
                    cv2.putText(marked_img, detection["text"], (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
            except Exception as direct_error:
                logger.error(f"Error in direct text detection: {str(direct_error)}")
        
        # Sort by confidence
        #detected_plates = sorted(detected_plates, key=lambda x: x["confidence"], reverse=True)
        if detected_plates:
            plate = detected_plates[0]  
            cleaned_text = re.sub(r'[^A-Z0-9]', '', plate["text"].upper())
            detected_plates = [{"text": cleaned_text, "confidence": plate["confidence"]}]
        else:
            detected_plates = []

        # Convert image to base64 for frontend display
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Text detection completed in {processing_time:.2f} seconds")
        logger.info(f"Number of text regions detected: {len(detected_plates)}")
        
        return {
            "plates": detected_plates,
            "processing_time": processing_time,
            "processed_image": img_str
        }
        
    except Exception as e:
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

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)