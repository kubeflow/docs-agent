#!/usr/bin/env python3
"""Feedback triage: which layer failed on a thumbs-down, and which working group owns the fix.

Subcommands
  feedback  triage the answers users marked not helpful (Langfuse score user_feedback = 0)
            in the last --since-hours. This is what the nightly CronJob runs.
  triage    triage given trace ids, or the --recent N latest chats.
  ask       send questions to the chat endpoint and print their Langfuse session/trace ids
            (to create test cases).

For each trace: take the question, tool calls, retrieved chunks and answer from Langfuse,
search the index again independently (MCP server: docs top 8, issues top 5), and let the
judge model decide which layer failed:

  ok | generation | retrieval | tool_routing | content_gap_handled |
  content_gap_hallucination | agent_skipped_retrieval

Pipeline layers (generation, retrieval, tool_routing, agent_skipped_retrieval) belong to the
docs-agent maintainers; content gaps belong to the component's working group (suggested repo
from keywords). With --write the result is stored as scores triage_layer and
triage_component on the trace. Nothing is filed automatically: --report prints a Markdown
summary for a human to turn into issues.

The judge is flo-judge (Qwen), a different model family from the answer model (Gemma).
Its agreement with human labels has not been measured yet; treat layers as suggestions.

Standard library only, so it runs in a plain python image.

Env: LANGFUSE_HOST (or LANGFUSE_BASE_URL), LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
     MCP_URL, JUDGE_URL, JUDGE_MODEL, AGENT_BASE + AGENT_PATH (ask only).
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

LF = (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "http://langfuse-web.langfuse.svc.cluster.local:3000").rstrip("/")
LF_PK, LF_SK = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
AGENT = os.getenv("AGENT_BASE", "http://kagent-controller.docs-agent.svc.cluster.local:8083").rstrip("/")
AGENT_PATH = os.getenv("AGENT_PATH", "/api/a2a/docs-agent/kubeflow-docs-agent/")
MCP_URL = os.getenv("MCP_URL", "http://mcp-kubeflow-docs.docs-agent.svc.cluster.local:8000/mcp")
JUDGE_URL = os.getenv("JUDGE_URL", "http://llm-judge.ml-infra.svc.cluster.local/v1/chat/completions")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "flo-judge")

PIPELINE_LAYERS = {"generation", "retrieval", "tool_routing", "agent_skipped_retrieval"}

# component -> (keywords, suggested repo). The repo is a suggestion for a human to confirm.
COMPONENTS = {
    "pipelines": (["pipeline", "kfp", "argo workflow", "components/pipelines"], "kubeflow/pipelines"),
    "notebooks": (["notebook", "jupyter", "workspace", "components/notebooks"], "kubeflow/notebooks"),
    "trainer": (["trainer", "trainjob", "training operator", "pytorchjob", "tfjob", "distributed training"], "kubeflow/trainer"),
    "katib": (["katib", "hyperparameter", "hpo", "tuning"], "kubeflow/katib"),
    "kserve": (["kserve", "inferenceservice", "model serving"], "kserve/kserve"),
    "spark": (["spark"], "kubeflow/spark-operator"),
    "model-registry": (["model registry", "model-registry"], "kubeflow/model-registry"),
    "manifests": (["install", "manifests", "kustomize", "helm", "deploy kubeflow"], "kubeflow/manifests"),
}


def guess_component(question: str, sources: list[str]) -> str:
    q = question.lower()
    src = " ".join(sources).lower()
    best, best_score = "general", 0
    for name, (words, _repo) in COMPONENTS.items():
        score = sum(4 for w in words if w in q) + sum(1 for w in words if w in src)
        if score > best_score:
            best, best_score = name, score
    return best


def repo_for(component: str) -> str:
    return COMPONENTS.get(component, ([], "kubeflow/website"))[1]


# ----- HTTP ----------------------------------------------------------------------------

def http(method: str, url: str, body=None, headers=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return (json.loads(raw) if raw else {}), r.headers


def lf_get(path: str, **params):
    auth = base64.b64encode(f"{LF_PK}:{LF_SK}".encode()).decode()
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return http("GET", f"{LF}/api/public{path}?{q}", headers={"Authorization": f"Basic {auth}"}, timeout=60)[0]


def lf_score(trace_id: str, name: str, value: str, comment: str = ""):
    auth = base64.b64encode(f"{LF_PK}:{LF_SK}".encode()).decode()
    body = {"id": f"{trace_id}-{name}", "traceId": trace_id, "name": name, "value": value,
            "dataType": "CATEGORICAL", "comment": comment[:500]}
    http("POST", f"{LF}/api/public/scores", body, {"Authorization": f"Basic {auth}"}, timeout=60)


# ----- Langfuse reads ------------------------------------------------------------------

def thumbs_down(since: dt.datetime) -> list[dict]:
    """user_feedback = 0 scores newer than `since`: [{trace_id, comment, at}]."""
    out, cursor = [], None
    for _ in range(50):
        page = lf_get("/v3/scores", name="user_feedback", dataType="BOOLEAN", value="false",
                      fields="core,subject,details", limit=100, cursor=cursor)
        for s in page.get("data", []):
            at = s.get("timestamp") or s.get("createdAt") or ""
            if at and at < since.isoformat().replace("+00:00", "Z"):
                continue
            subject = s.get("subject") or {}
            trace = subject.get("id") if subject.get("kind") == "trace" else None
            if subject.get("kind") == "session":  # vote arrived before the trace; use the session's latest
                trace = trace_for_session(subject.get("id", ""), wait_s=0)
            if trace:
                out.append({"trace_id": trace, "comment": s.get("comment") or "", "at": at})
        cursor = (page.get("meta") or {}).get("cursor")
        if not cursor:
            break
    return out


def trace_observations(trace_id: str) -> list[dict]:
    return lf_get("/v2/observations", traceId=trace_id, fields="core,basic,io,metadata", limit=100).get("data", [])


def recent_traces(n: int) -> list[str]:
    obs = lf_get("/v2/observations", type="AGENT", fields="core,basic", limit=100).get("data", [])
    seen, out = set(), []
    for o in sorted(obs, key=lambda x: x["startTime"], reverse=True):
        if o["traceId"] not in seen:
            seen.add(o["traceId"])
            out.append(o["traceId"])
    return out[:n]


def trace_for_session(session_id: str, wait_s: int = 90) -> str | None:
    deadline = time.time() + wait_s
    while True:
        data = lf_get("/v2/observations", type="AGENT", sessionId=session_id, fields="core,basic", limit=100).get("data", [])
        if data:
            return max(data, key=lambda o: o.get("startTime") or "")["traceId"]
        if time.time() >= deadline:
            return None
        time.sleep(5)


def _j(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def extract(trace_id: str) -> dict:
    """Pull question, tool calls, retrieved chunks and the final answer out of a trace."""
    obs = trace_observations(trace_id)
    chats = sorted((o for o in obs if o["name"] == "openai.chat"), key=lambda o: o["startTime"])
    question, answer = "", ""
    if chats:
        msgs = _j(chats[0].get("input")) or []
        users = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        question = users[-1] if users else ""
        for o in reversed(chats):
            out = _j(o.get("output")) or []
            if out and isinstance(out[0], dict) and out[0].get("finish_reason") == "stop":
                answer = out[0].get("content") or ""
                break
    tool_calls, chunks = [], []
    for o in obs:
        if o["type"] == "TOOL" and o.get("input"):
            tool_calls.append({"tool": o["name"], "input": _j(o["input"])})
            out = _j(o.get("output"))
            text = ""
            if isinstance(out, dict):
                text = "".join(p.get("text", "") for p in out.get("content", []) if isinstance(p, dict))
            if text:
                chunks.extend(split_chunks(text))
    return {"trace_id": trace_id, "question": question, "answer": answer, "tool_calls": tool_calls, "chunks": chunks}


def split_chunks(text: str) -> list[str]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "markdown_summary" in data:
            text = data["markdown_summary"]
    except ValueError:
        pass
    parts = re.split(r"(?m)^### Result ", text)
    return [("### Result " + p).strip() for p in parts[1:]]


def sources_of(chunks: list[str]) -> list[str]:
    out = []
    for c in chunks:
        out += re.findall(r"\*\*(?:File|Source|Section):\*\*\s*(.+)", c)
    return out


# ----- MCP probe (streamable HTTP, no client library) -----------------------------------

def _mcp_post(body: dict, session: str | None) -> tuple[dict, str | None]:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        headers["mcp-session-id"] = session
    req = urllib.request.Request(MCP_URL, data=json.dumps(body).encode(), method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode()
        sid = r.headers.get("mcp-session-id") or session
    if not raw.strip():
        return {}, sid
    if raw.lstrip().startswith("{"):
        return json.loads(raw), sid
    datas = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    return (json.loads(datas[-1]) if datas else {}), sid


def probe(question: str) -> dict:
    res: dict = {"docs": [], "issues": []}
    try:
        _, sid = _mcp_post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "flo-triage", "version": "1"}}}, None)
        _mcp_post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return res | {"init_error": str(e)[:120]}
    for i, (key, tool, args) in enumerate((
        ("docs", "search_kubeflow_docs", {"query": question, "top_k": 8}),
        ("issues", "search_github_issues", {"query": question, "top_k": 5}),
    ), start=2):
        try:
            r, _ = _mcp_post({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": tool, "arguments": args}}, sid)
            content = (r.get("result") or {}).get("content") or []
            res[key] = split_chunks(content[0].get("text", "") if content else "")
        except (urllib.error.URLError, TimeoutError, ValueError) as e:  # reported, not fatal
            res[key + "_error"] = str(e)[:120]
    return res


# ----- Judge ---------------------------------------------------------------------------

def judge(prompt: str) -> dict:
    body = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 300, "messages": [{"role": "user", "content": prompt}]}
    text = (http("POST", JUDGE_URL, body, timeout=180)[0]["choices"][0]["message"]["content"]) or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def relevant(question: str, chunks: list[str]) -> list[int]:
    """Numbers of passages that contain the specific steps/facts asked for. Topic mentions do not count."""
    if not chunks:
        return []
    listing = "\n\n".join(f"[{i + 1}] {c[:700]}" for i, c in enumerate(chunks[:8]))
    out = judge(
        "A user asked a Kubeflow question. Decide whether the passages contain the SPECIFIC steps, "
        "settings or facts needed to answer it. A passage that only mentions the product or topic, or "
        "that says something is possible without saying how, does NOT count. Be strict.\n"
        f"Question: {question}\n\nPassages:\n{listing}\n\n"
        'Reply with JSON only: {"sufficient": "yes|partial|no", "numbers": [numbers of passages that supply the answer]}'
    )
    if str(out.get("sufficient", "no")).lower() != "yes":
        return []
    return [n for n in out.get("numbers", []) if isinstance(n, int)]


def answer_check(question: str, answer: str, chunks: list[str]) -> dict:
    listing = "\n\n".join(f"[{i + 1}] {c[:700]}" for i, c in enumerate(chunks[:6]))
    return judge(
        f"Question: {question}\n\nRetrieved passages:\n{listing}\n\nAnswer given:\n{answer[:1200]}\n\n"
        "Rate the answer. Reply with JSON only: "
        '{"grounded": "yes|partial|no", "admits_not_found": true|false, "reason": "one short sentence"}'
    )


def diagnose(trace_id: str) -> dict:
    t = extract(trace_id)
    q, a = t["question"], t["answer"]
    comp = guess_component(q, sources_of(t["chunks"]))
    base = {"trace_id": trace_id, "question": q, "component": comp, "suggested_repo": repo_for(comp)}
    if not q:
        return {**base, "layer": "unknown", "reason": "Trace has no chat input (not ingested yet, or not an agent trace)."}
    if not t["tool_calls"]:
        return {**base, "layer": "agent_skipped_retrieval", "reason": "No retrieval tool call in the trace."}

    p = probe(q)
    agent_rel = relevant(q, t["chunks"])
    docs_rel = relevant(q, p["docs"])
    issues_rel = relevant(q, p["issues"])
    used_issues = any(c["tool"] == "search_github_issues" for c in t["tool_calls"])
    chk = answer_check(q, a, t["chunks"]) if a else {}

    if agent_rel:
        ok = chk.get("grounded") == "yes"
        layer = "ok" if ok else "generation"
        reason = chk.get("reason") or "Relevant chunk was retrieved."
    elif docs_rel:
        layer, reason = "retrieval", "A relevant doc exists in the index but the agent's search did not surface it."
    elif issues_rel and not used_issues:
        layer, reason = "tool_routing", "A relevant GitHub issue exists but the agent never searched issues."
    else:
        handled = bool(chk.get("admits_not_found"))
        layer = "content_gap_handled" if handled else "content_gap_hallucination"
        reason = "No relevant passage in indexed docs or issues. " + ("The agent said it could not find it." if handled else "The agent answered anyway.")
    return {**base, "layer": layer, "reason": reason, "agent_relevant": len(agent_rel), "docs_relevant": len(docs_rel),
            "issues_relevant": len(issues_rel), "probe_errors": [v for k, v in p.items() if k.endswith("_error")]}


def owner(d: dict) -> str:
    if d["layer"] in PIPELINE_LAYERS:
        return "kubeflow/docs-agent"
    if d["layer"].startswith("content_gap"):
        return d["suggested_repo"]
    return "-"


def report(results: list[dict]) -> str:
    """Markdown summary for a human: what to file where. Nothing is filed automatically."""
    lines = [f"# Flo feedback triage ({len(results)} thumbs-down)", ""]
    by_layer = collections.Counter(d["layer"] for d in results)
    lines += ["| Layer | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in by_layer.most_common()] + [""]
    groups = collections.defaultdict(list)
    for d in results:
        groups[owner(d)].append(d)
    for repo, items in sorted(groups.items()):
        lines += [f"## {repo}", ""]
        for d in items:
            note = f" User said: \"{d['comment']}\"" if d.get("comment") else ""
            lines.append(f"- **{d['layer']}** ({d['component']}): {d['question'][:160]!r}. {d['reason']}{note} "
                         f"Trace `{d['trace_id']}`.")
        lines.append("")
    return "\n".join(lines)


# ----- Chat (to create test cases) ------------------------------------------------------

def ask(question: str) -> str:
    ctx = str(uuid.uuid4())
    body = {"jsonrpc": "2.0", "id": "1", "method": "message/send",
            "params": {"message": {"kind": "message", "messageId": str(uuid.uuid4()), "role": "user",
                                    "contextId": ctx, "parts": [{"kind": "text", "text": question}]}}}
    http("POST", f"{AGENT}{AGENT_PATH}", body, timeout=300)
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("questions", nargs="+")
    for name in ("feedback", "triage"):
        p = sub.add_parser(name)
        p.add_argument("--write", action="store_true", help="store triage_layer/triage_component scores")
        p.add_argument("--report", action="store_true", help="print a Markdown summary")
        p.add_argument("--out", default="", help="write results as JSON")
        if name == "feedback":
            p.add_argument("--since-hours", type=float, default=24)
        else:
            p.add_argument("trace_ids", nargs="*")
            p.add_argument("--recent", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "ask":
        for q in args.questions:
            ctx = ask(q)
            print(json.dumps({"question": q, "session": ctx, "trace_id": trace_for_session(ctx)}))
        return 0

    if args.cmd == "feedback":
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.since_hours)
        votes = {v["trace_id"]: v for v in thumbs_down(since)}
        targets = [(t, v["comment"]) for t, v in votes.items()]
        print(f"{len(targets)} thumbs-down since {since.isoformat(timespec='minutes')}", file=sys.stderr)
    else:
        targets = [(t, "") for t in (args.trace_ids or recent_traces(args.recent or 5))]

    results = []
    for trace_id, comment in targets:
        try:
            d = diagnose(trace_id) | {"comment": comment}
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            print(f"warn: {trace_id}: {e}", file=sys.stderr)
            continue
        results.append(d)
        print(json.dumps(d))
        if args.write and d["layer"] != "unknown":
            lf_score(trace_id, "triage_layer", d["layer"], d["reason"])
            lf_score(trace_id, "triage_component", d["component"], d["suggested_repo"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    if args.report:
        print(report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
