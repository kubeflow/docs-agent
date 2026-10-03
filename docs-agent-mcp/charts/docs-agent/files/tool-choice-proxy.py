"""Rewrite tool_choice on the way to Qwen. Stdlib only. Streaming passthrough."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import sys
import urllib.error
import urllib.request

# Flo calls {baseUrl}/chat/completions with baseUrl=.../openai/v1, so the
# request path is /openai/v1/chat/completions. Forward that path onto the
# Qwen host root - do not append it to a base that already includes /openai/v1.
UPSTREAM = os.environ.get(
    "UPSTREAM",
    "http://qwen-llm-stable.ml-infra.svc.cluster.local",
).rstrip("/")
PORT = int(os.environ.get("PORT", "8080"))


CHITCHAT = re.compile(
    r"^(hi+|hello|hey|yo|ssup|wassup|whats? ?up|what'?s? ?up|what is up|sup|"
    r"thanks|thank you|thx|ty|bye|goodbye|ok|okay|cool|nice|cheers|howdy|"
    r"namaste|hola|good (morning|evening|night|afternoon))"
    r"(\s+\w+){0,2}[\s!.?]*$",
    re.I,
)
IN_SCOPE = re.compile(
    r"kubeflow|kfp|katib|\bkserve\b|pipeline|notebook|gsoc|install|deploy|yaml|"
    r"error|pytorch|training|istio|pvc|secret|release|version|manifest|"
    r"how (do|to|can)|what is\b(?! up\b)|tell me|search|docs",
    re.I,
)


def last_nonsystem_role(messages):
    role = None
    for message in messages or []:
        current = message.get("role")
        if current and current != "system":
            role = current
    return role


def last_user_text(messages):
    text = ""
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    return text.strip()


def is_chitchat(text: str) -> bool:
    if not text:
        return True
    if IN_SCOPE.search(text):
        return False
    return bool(CHITCHAT.fullmatch(text))


def rewrite_body(raw: bytes) -> bytes:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not data.get("tools"):
        return raw
    role = last_nonsystem_role(data.get("messages"))
    if role == "user" and is_chitchat(last_user_text(data.get("messages"))):
        data["tool_choice"] = "none"
    elif role == "user":
        data["tool_choice"] = "required"
    else:
        data["tool_choice"] = "auto"
    return json.dumps(data).encode()


def hop_headers(headers):
    skip = {"host", "content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in skip}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path in ("/healthz", "/"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._proxy(b"", method="GET")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path.rstrip("/").endswith("chat/completions"):
            raw = rewrite_body(raw)
        self._proxy(raw, method="POST")

    def _proxy(self, body: bytes, method: str):
        url = UPSTREAM + self.path
        headers = hop_headers(self.headers)
        if method == "POST":
            headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
            headers["Content-Length"] = str(len(body))
        req = urllib.request.Request(url, data=body or None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() in ("transfer-encoding", "connection", "content-length"):
                        continue
                    self.send_header(key, value)
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as exc:
            err = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "text/plain"))
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
        except Exception as exc:
            msg = str(exc).encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"tool-choice-proxy upstream={UPSTREAM} port={PORT}", flush=True)
    server.serve_forever()
