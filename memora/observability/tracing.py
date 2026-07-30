"""OTel wiring: tracer provider, FastAPI auto-instrumentation, optional OTLP export.

Export is opt-in (M2.4). With no endpoint configured, spans are created and dropped —
memora phones nobody home by default, which is what the self-host and air-gapped paths
require. Point `OTEL_EXPORTER_OTLP_ENDPOINT` at a collector and the same spans, including
the per-phase retrieval spans behind the p95 number, start flowing there.
"""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_provider_set = False

# Module-level: `get_tracer` returns a proxy that resolves the global provider lazily, so
# this is safe to bind at import time, before configure_tracing runs.
tracer = trace.get_tracer("memora")


def configure_tracing(app: FastAPI, otlp_endpoint: str | None = None) -> None:
    global _provider_set
    if not _provider_set:  # the global provider can only be set once per process
        provider = TracerProvider(resource=Resource.create({"service.name": "memora"}))
        if otlp_endpoint is not None:
            # batched, so exporting never blocks a request on the collector
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        trace.set_tracer_provider(provider)
        _provider_set = True
    FastAPIInstrumentor.instrument_app(app)
