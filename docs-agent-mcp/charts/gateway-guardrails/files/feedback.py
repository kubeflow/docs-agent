"""Thumbs up/down from the chat widget, recorded as Langfuse scores.

POST /api/feedback  {"contextId": "...", "turn": 0, "value": "up"|"down", "comment": "..."}
GET  /healthz

The widget's A2A contextId is the Langfuse session id of the chat. Each answer is
one trace in that session; `turn` (0-based) picks which one. The score is
`user_feedback` (BOOLEAN, 1 = up) with a deterministic id, so changing a vote
overwrites it instead of adding a second one. If the trace has not been ingested
yet, the vote is stored on the session instead of being lost.

The gateway requires a valid session JWT on this path and applies the same rate
limits as chat. Standard library only: the chart runs it from a ConfigMap on a
stock python image, so there is no image to build.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LF = os.getenv("LANGFUSE_HOST", "http://langfuse-web.langfuse.svc.cluster.local:3000").rstrip("/")
LF_PK, LF_SK = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
PORT = int(os.getenv("PORT", "8000"))
SCORE_NAME = "user_feedback"
CONTEXT_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
MAX_COMMENT = 1000
MAX_BODY = 4096


class BadRequest(ValueError):
    pass


def parse(body: bytes) -> dict:
    """Validate a feedback request. Raises BadRequest with a message safe to return."""
    if len(body) > MAX_BODY:
        raise BadRequest("body too large")
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        raise BadRequest("invalid JSON") from None
    if not isinstance(data, dict):
        raise BadRequest("expected a JSON object")
    ctx = data.get("contextId")
    if not isinstance(ctx, str) or not CONTEXT_RE.match(ctx):
        raise BadRequest("contextId missing or invalid")
    value = data.get("value")
    if value not in ("up", "down"):
        raise BadRequest('value must be "up" or "down"')
    turn = data.get("turn")
    if turn is not None and (not isinstance(turn, int) or isinstance(turn, bool) or not 0 <= turn < 1000):
        raise BadRequest("turn must be a non-negative integer")
    comment = data.get("comment") or ""
    if not isinstance(comment, str):
        raise BadRequest("comment must be a string")
    return {"contextId": ctx, "value": value, "turn": turn, "comment": comment.strip()[:MAX_COMMENT]}


def pick_trace(observations: list[dict], turn: int | None) -> str | None:
    """Traces of a session in chat order; `turn` indexes them, None means the latest."""
    first_seen: dict[str, str] = {}
    for o in observations:
        t, start = o.get("traceId"), o.get("startTime") or ""
        if t and (t not in first_seen or start < first_seen[t]):
            first_seen[t] = start
    ordered = [t for t, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]
    if not ordered:
        return None
    if turn is None or turn >= len(ordered):
        return ordered[-1]
    return ordered[turn]


def score_body(fb: dict, trace_id: str | None) -> dict:
    body = {"name": SCORE_NAME, "dataType": "BOOLEAN", "value": 1 if fb["value"] == "up" else 0,
            "comment": fb["comment"] or None, "metadata": {"source": "chat-widget", "turn": fb["turn"]}}
    if trace_id:
        return body | {"id": f"{trace_id}-{SCORE_NAME}", "traceId": trace_id}
    turn = "latest" if fb["turn"] is None else fb["turn"]
    return body | {"id": f"{fb['contextId']}-turn-{turn}-{SCORE_NAME}", "sessionId": fb["contextId"]}


def _lf(method: str, path: str, body=None, **params):
    auth = base64.b64encode(f"{LF_PK}:{LF_SK}".encode()).decode()
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(f"{LF}/api/public{path}{'?' + q if q else ''}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b"{}")


def record(fb: dict) -> str | None:
    trace_id = None
    for attempt in range(3):  # traces arrive a few seconds after the answer
        obs = _lf("GET", "/v2/observations", type="AGENT", sessionId=fb["contextId"], fields="core,basic", limit=100)
        trace_id = pick_trace(obs.get("data", []), fb["turn"])
        if trace_id or attempt == 2:
            break
        time.sleep(2)
    _lf("POST", "/scores", score_body(fb, trace_id))
    return trace_id


class Handler(BaseHTTPRequestHandler):
    server_version = "flo-feedback"

    def _reply(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthz":
            self._reply(200, {"status": "ok"})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/feedback":
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            fb = parse(self.rfile.read(min(length, MAX_BODY + 1)))
        except BadRequest as e:
            self._reply(400, {"error": str(e)})
            return
        try:
            trace_id = record(fb)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"langfuse error: {e}", file=sys.stderr)
            self._reply(502, {"error": "could not record feedback"})
            return
        self._reply(200, {"ok": True, "linked": bool(trace_id)})

    def log_message(self, fmt, *args):  # one line per request, no bodies
        print(f"{self.address_string()} {fmt % args}", file=sys.stderr)


if __name__ == "__main__":
    if not (LF_PK and LF_SK):
        sys.exit("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
