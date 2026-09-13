"""Optional tracing hooks used by MCP search. No-op unless a tracer is wired later."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

ATTR_RETRIEVAL_INTENT = "retrieval.intent"
ATTR_RETRIEVAL_MODE = "retrieval.mode"
ATTR_LANGFUSE_META = "langfuse.metadata."
ATTR_LANGFUSE_OUTPUT = "langfuse.observation.output"


@contextmanager
def retrieval_span(name: str, **_kwargs: Any) -> Iterator[None]:
    yield None


@contextmanager
def mcp_tool_span(name: str, **_kwargs: Any) -> Iterator[None]:
    yield None


@contextmanager
def embedding_span(*_args: Any, **_kwargs: Any) -> Iterator[None]:
    yield None


def finish_retrieval_span(span: Any, hits: list[dict], **_kwargs: Any) -> None:
    return None


def finish_tool_span(span: Any, result: Any, **_kwargs: Any) -> None:
    return None


def set_span_attributes(span: Any, attrs: dict[str, Any]) -> None:
    if span is None:
        return
    for key, value in attrs.items():
        span.set_attribute(key, value)


def serve_mcp(mcp: Any, host: str = "0.0.0.0", port: int = 8000) -> None:
    mcp.run(transport="http", host=host, port=port)
