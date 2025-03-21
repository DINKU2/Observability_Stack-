from fastapi import FastAPI
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

# Initialize resource
resource = Resource.create({"service.name": "api2"})

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
    name="api2_requests",
    description="Number of requests to API2",
    unit="1"
)

# Create FastAPI app
app = FastAPI()

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

@app.get("/")
async def read_root():
    request_counter.add(1, {"endpoint": "root"})
    return {"Hello": "from API 2"} 