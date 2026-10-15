"""Langfuse tracing for the MCP tools. No-op unless LANGFUSE_PUBLIC_KEY is set.

Each tool call, retrieval and query embedding becomes a Langfuse observation
(TOOL, RETRIEVER, EMBEDDING) with truncated inputs and a small output summary
(hit count, mode, top scores). Tracing errors never break a search.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_TRUNC = 300
_DEFAULT_BASE_URL = "http://langfuse-web.langfuse.svc.cluster.local:3000"


def enabled() -> bool:
    return bool((os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip())


def _truncate(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _TRUNC:
            return value
        return value[:_TRUNC] + "..."
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    return value


def _client():
    os.environ.setdefault("LANGFUSE_BASE_URL", _DEFAULT_BASE_URL)
    from langfuse import get_client

    return get_client()


def top_scores(hits: list[dict], limit: int = 5) -> list[float]:
    scores: list[float] = []
    for hit in hits[:limit]:
        # Milvus puts distance/score on the hit; some callers nest it in entity.
        entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
        for row in (hit, entity):
            value = next((row[k] for k in ("distance", "score") if row.get(k) is not None), None)
            if value is not None:
                try:
                    scores.append(round(float(value), 4))
                except (TypeError, ValueError):
                    pass
                break
    return scores


def retriever_output(
    hits: list[dict],
    *,
    mode: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"hit_count": len(hits), "top_scores": top_scores(hits)}
    if mode is not None:
        out["mode"] = mode
    if intent is not None:
        out["intent"] = intent
    return out


def _set_output(obs: Any, output: dict[str, Any]) -> None:
    if obs is None:
        return
    try:
        if hasattr(obs, "update"):
            obs.update(output=output)
    except Exception:
        pass


@contextmanager
def _observe(as_type: str, name: str, input: dict[str, Any], **extra: Any) -> Iterator[Any]:
    if not enabled():
        yield None
        return
    cm = None
    try:
        langfuse = _client()
        cm = langfuse.start_as_current_observation(
            as_type=as_type,
            name=name,
            input=_truncate(input),
            **extra,
        )
        obs = cm.__enter__()
    except Exception:
        yield None
        return
    try:
        yield obs
    finally:
        if cm is not None:
            try:
                cm.__exit__(*sys.exc_info())
            except Exception:
                pass


@contextmanager
def observe_tool(name: str, input: dict[str, Any]) -> Iterator[Any]:
    with _observe("tool", name, input) as obs:
        yield obs


@contextmanager
def observe_retriever(name: str, input: dict[str, Any]) -> Iterator[Any]:
    with _observe("retriever", name, input) as obs:
        yield obs


@contextmanager
def observe_embedding(name: str, input: dict[str, Any], *, model: str) -> Iterator[Any]:
    with _observe("embedding", name, input, model=model) as obs:
        yield obs


def traced(
    as_type: str,
    *,
    inputs: tuple[str, ...],
    output: Callable[[Any], dict[str, Any]] | None = None,
    **extra: Any,
) -> Callable:
    """Decorator: record a call as a Langfuse observation named after the function.

    `inputs` names the arguments to record; `output` maps the return value to a
    summary. functools.wraps keeps the signature, so FastMCP still builds the
    tool schema from the original function.
    """

    def decorate(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled():
                return fn(*args, **kwargs)
            try:
                bound = sig.bind_partial(*args, **kwargs).arguments
                record = {k: bound[k] for k in inputs if k in bound}
            except TypeError:
                record = {}
            with _observe(as_type, fn.__name__, record, **extra) as obs:
                result = fn(*args, **kwargs)
                if output is not None and obs is not None:
                    try:
                        _set_output(obs, output(result))
                    except Exception:
                        pass
                return result

        return wrapper

    return decorate
