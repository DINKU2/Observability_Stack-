from fastapi import FastAPI
import httpx
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Initialize tracing
resource = Resource.create({"service.name": "api1"})
tracer_provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(tracer_provider)

# Create FastAPI app
app = FastAPI()

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Instrument HTTPX
HTTPXClientInstrumentor().instrument()

@app.get("/")
async def read_root():
    return {"Hello": "from API 1"}

@app.get("/call-api2")
async def call_api2():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://api2:8000/")
        return {"API1 got response from API2": response.json()}
