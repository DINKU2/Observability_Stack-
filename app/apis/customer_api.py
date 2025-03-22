from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import random
from typing import List, Optional
import os
import time

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Prometheus metrics
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# Setup tracing
resource = Resource(attributes={SERVICE_NAME: "customer-api"})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Custom metrics
CUSTOMER_LOOKUP_COUNT = Counter(
    "customer_lookup_count", 
    "Count of customer lookups",
    ["found"]  # Label to track found vs not found
)

LOOKUP_TIME = Histogram(
    "customer_lookup_time_seconds",
    "Time spent looking up customers",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

app = FastAPI(title="Customer Insights API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Common food items for recommendations
FOOD_ITEMS = [
    "Big Mac", "Quarter Pounder", "Chicken McNuggets", "McChicken",
    "French Fries", "Cheeseburger", "Fish Fillet", "Apple Pie",
    "McFlurry", "Chicken Sandwich", "Salad", "Happy Meal"
]

# Global DB variable
customer_db = None

# Load customer database
@app.on_event("startup")
def startup_db_client():
    global customer_db
    db_path = os.environ.get('CUSTOMER_DB_PATH', './data/customer_database.csv')
    
    if os.path.exists(db_path):
        customer_db = pd.read_csv(db_path)
    else:
        customer_db = pd.DataFrame(columns=['license_plate', 'name', 'last_order', 'last_visit', 'total_visits', 'preferred_time'])

@app.get("/customer/{plate_number}")
def get_customer(plate_number: str, request: Request = None):
    with tracer.start_as_current_span("get_customer") as span:
        span.set_attribute("plate_number", plate_number)
        start_time = time.time()
        
        global customer_db
        
        # Find customer by plate
        with tracer.start_as_current_span("database_lookup"):
            customer = customer_db[customer_db['license_plate'] == plate_number]
            span.set_attribute("database.size", len(customer_db))
        
        if len(customer) == 0:
            CUSTOMER_LOOKUP_COUNT.labels(found="false").inc()
            span.set_attribute("customer.found", False)
            
            lookup_time = time.time() - start_time
            LOOKUP_TIME.observe(lookup_time)
            
            return {"found": False, "message": "Customer not found", "processing_time": lookup_time}
        
        with tracer.start_as_current_span("process_customer_data"):
            customer_data = customer.iloc[0].to_dict()
            
            # Convert last_order from string to list if needed
            if isinstance(customer_data['last_order'], str):
                try:
                    customer_data['last_order'] = eval(customer_data['last_order'])
                except:
                    customer_data['last_order'] = [customer_data['last_order']]
        
        # Generate recommendations
        with tracer.start_as_current_span("generate_recommendations"):
            recommendations = generate_recommendations(customer_data)
            span.set_attribute("recommendations.count", len(recommendations))
        
        CUSTOMER_LOOKUP_COUNT.labels(found="true").inc()
        span.set_attribute("customer.found", True)
        span.set_attribute("customer.name", customer_data["name"])
        
        lookup_time = time.time() - start_time
        LOOKUP_TIME.observe(lookup_time)
        
        return {
            "found": True,
            "customer": customer_data,
            "recommendations": recommendations,
            "processing_time": lookup_time
        }

def generate_recommendations(customer_data):
    # Simple recommendation logic
    last_orders = customer_data['last_order']
    
    # Always recommend their favorite
    recommendations = [last_orders[0]] if last_orders else []
    
    # Add some new items
    potential_new_items = [item for item in FOOD_ITEMS if item not in last_orders]
    if potential_new_items:
        recommendations.extend(random.sample(potential_new_items, min(2, len(potential_new_items))))
    
    return recommendations

@app.get("/")
def read_root():
    return {"message": "Customer Insights API"}

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)