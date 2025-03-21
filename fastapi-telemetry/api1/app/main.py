from fastapi import FastAPI
import httpx
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

# Initialize resource
resource = Resource.create({"service.name": "api1"})

# Initialize tracing
tracer_provider = TracerProvider(resource=resource)
otlp_trace_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))
trace.set_tracer_provider(tracer_provider)

# Initialize metrics
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://otel-collector:4317", insecure=True)
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)

# Get meter
meter = metrics.get_meter(__name__)

# Create metrics
request_counter = meter.create_counter(
    name="api1_requests",
    description="Number of requests to API1",
    unit="1"
)

api2_call_counter = meter.create_counter(
    name="api1_api2_calls",
    description="Number of calls from API1 to API2",
    unit="1"
)

# Create FastAPI app
app = FastAPI()

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Instrument HTTPX
HTTPXClientInstrumentor().instrument()

@app.get("/")
async def read_root():
    request_counter.add(1, {"endpoint": "root"})
    return {"Hello": "from API 1"}

@app.get("/call-api2")
async def call_api2():
    request_counter.add(1, {"endpoint": "call-api2"})
    async with httpx.AsyncClient() as client:
        api2_call_counter.add(1)
        response = await client.get("http://api2:8000/")
        return {"API1 got response from API2": response.json()}
