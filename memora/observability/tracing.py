"""OTel skeleton: tracer provider + FastAPI auto-instrumentation.

Spans are created but not yet exported; exporter wiring and the custom memory-span
helper land with the p95 instrumentation in M2.4.
"""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

_provider_set = False


def configure_tracing(app: FastAPI) -> None:
    global _provider_set
    if not _provider_set:  # the global provider can only be set once per process
        trace.set_tracer_provider(
            TracerProvider(resource=Resource.create({"service.name": "memora"}))
        )
        _provider_set = True
    FastAPIInstrumentor.instrument_app(app)
