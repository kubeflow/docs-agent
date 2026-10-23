#!/usr/bin/env python3
"""Local demo server for the docs-site chatbot widget.

Serves the frontend/ static files and fakes a KAgent A2A
`message/stream` endpoint at POST /mock-agent, so chatbot.js can be
exercised in a browser without a real Kagent cluster.
"""
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_ROOT = Path(__file__).resolve().parent.parent

CANNED_REPLY = (
    "Hello! This is a **mock** response streamed from `mock_server.py`. "
    "It fakes the Kagent A2A `message/stream` protocol so you can see "
    "chatbot.js render tokens as they arrive, including code:\n\n"
    "```python\nprint('hello from the docs bot')\n```"
)


class Handler(BaseHTTPRequestHandler):
    def _serve_static(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            path = "/demo/index.html"
        file_path = FRONTEND_ROOT / path.lstrip("/")
        if not file_path.is_file():
            self.send_error(404, "Not found")
            return
        content_type = "text/html"
        if file_path.suffix == ".js":
            content_type = "application/javascript"
        elif file_path.suffix == ".css":
            content_type = "text/css"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._serve_static()

    def do_POST(self):
        if self.path != "/mock-agent":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        request_body = json.loads(self.rfile.read(length) or b"{}")
        rpc_id = request_body.get("id", str(uuid.uuid4()))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # Stream the canned reply word-by-word to mimic token streaming.
        words = CANNED_REPLY.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            payload = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "message": {
                        "kind": "message",
                        "role": "assistant",
                        "parts": [{"kind": "text", "text": chunk}],
                    }
                },
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)

        # Final chunk signals end-of-turn.
        final_payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"final": True},
        }
        self.wfile.write(f"data: {json.dumps(final_payload)}\n\n".encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):
        print(f"[mock_server] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    port = 8000
    server = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"Serving {FRONTEND_ROOT} at http://localhost:{port}/demo/index.html")
    print("Mock agent endpoint: POST http://localhost:8000/mock-agent")
    server.serve_forever()
