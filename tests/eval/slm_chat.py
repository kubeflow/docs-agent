"""Minimal browser chat for the model on the spare A10 (stdlib only).

kubectl -n slm-eval port-forward svc/slm 8000:8000
python tests/eval/slm_chat.py            # then open http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SLM chat</title>
<style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px;background:#0f1115;color:#e6e6e6}
#log{display:flex;flex-direction:column;gap:10px;margin-bottom:12px}
.msg{padding:10px 12px;border-radius:8px;white-space:pre-wrap;line-height:1.45}
.user{background:#1f3a5f;align-self:flex-end;max-width:80%}
.bot{background:#1b1e24;border:1px solid #2a2e36}
details{color:#9aa3ad;font-size:.9em;margin-bottom:6px}
.meta{color:#7d8590;font-size:.8em;margin-top:6px}
form{display:flex;gap:8px;position:sticky;bottom:0;background:#0f1115;padding:8px 0}
textarea{flex:1;resize:vertical;min-height:44px;background:#1b1e24;color:#e6e6e6;border:1px solid #2a2e36;border-radius:6px;padding:8px}
button{background:#2f81f7;color:#fff;border:0;border-radius:6px;padding:0 16px;cursor:pointer}
label{font-size:.9em;color:#9aa3ad}
</style></head><body>
<h3>__MODEL__ <label><input type="checkbox" id="think" checked> thinking</label>
<button type="button" id="clear" style="float:right;padding:4px 10px">clear</button></h3>
<div id="log"></div>
<form id="f"><textarea id="q" placeholder="Ask something... (Enter to send, Shift+Enter for newline)"></textarea>
<button>Send</button></form>
<script>
const log = document.getElementById('log'), q = document.getElementById('q');
let history = [];
function add(cls){const d=document.createElement('div');d.className='msg '+cls;log.appendChild(d);return d;}
document.getElementById('clear').onclick=()=>{history=[];log.innerHTML='';};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit();}});
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const text = q.value.trim(); if(!text) return; q.value='';
  add('user').textContent = text;
  history.push({role:'user', content:text});
  const bot = add('bot');
  const det = document.createElement('details'); det.innerHTML='<summary>thinking</summary><div></div>';
  const thinkEl = det.querySelector('div'); const out = document.createElement('div'); const meta = document.createElement('div');
  meta.className='meta'; bot.append(det, out, meta); det.style.display='none';
  const start = performance.now(); let first = null, answer = '', usage = null;
  const resp = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({messages: history, thinking: document.getElementById('think').checked})});
  if(!resp.ok){ out.textContent = 'error: ' + await resp.text(); history.pop(); return; }
  const reader = resp.body.getReader(), dec = new TextDecoder(); let buf = '';
  while(true){
    const {done, value} = await reader.read(); if(done) break;
    buf += dec.decode(value, {stream:true});
    const lines = buf.split('\\n'); buf = lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: ') || line === 'data: [DONE]') continue;
      const c = JSON.parse(line.slice(6));
      if(c.usage) usage = c.usage;
      const d = (c.choices && c.choices[0] && c.choices[0].delta) || {};
      const r = d.reasoning_content || d.reasoning;
      if((r || d.content) && first === null) first = performance.now();
      if(r){ det.style.display=''; thinkEl.textContent += r; }
      if(d.content){ answer += d.content; out.textContent = answer; }
    }
    window.scrollTo(0, document.body.scrollHeight);
  }
  const end = performance.now();
  history.push({role:'assistant', content:answer});
  if(usage && first){
    const tps = (usage.completion_tokens - 1) / ((end - first) / 1000);
    meta.textContent = `${usage.completion_tokens} tokens | first token ${((first-start)/1000).toFixed(2)}s | ${tps.toFixed(1)} tok/s | total ${((end-start)/1000).toFixed(2)}s`;
  }
};
</script></body></html>"""


def make_handler(upstream: str, model: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            body = PAGE.replace("__MODEL__", model).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            payload = {
                "model": model,
                "messages": req["messages"],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": bool(req.get("thinking"))},
            }
            upstream_req = urllib.request.Request(
                upstream.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                resp = urllib.request.urlopen(upstream_req, timeout=300)
            except (urllib.error.URLError, ConnectionError) as exc:
                msg = f"upstream {upstream} unreachable: {exc}. Is the port-forward running?".encode()
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with resp:
                for line in resp:
                    self.wfile.write(line)
                    self.wfile.flush()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="gemma-4-e4b")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.upstream, args.model))
    print(f"chat UI on http://localhost:{args.port} -> {args.upstream} ({args.model})")
    server.serve_forever()


if __name__ == "__main__":
    main()
