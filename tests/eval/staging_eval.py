"""End-to-end eval of a deployed docs-agent environment, recorded in Langfuse.

Sends the single-turn cases of flo_eval_dataset.json to the real agents (through
the kagent controller A2A API), checks tool routing and query fidelity, asks a
judge model whether each answer is grounded in what the tools returned, and
records everything in Langfuse as a dataset run, so two commits can be compared
side by side. Exits non-zero when a gate fails.

Unlike slm_tool_eval.py (model + canned tool results), this exercises the whole
deployed path: agent, MCP server, Milvus, model.

Standard library only, so it runs in a plain python image.

Environment (defaults suit the in-cluster PostSync Job in docs-agent-staging):
  AGENT_BASE          kagent controller URL
  AGENT_NAMESPACE     namespace whose agents are evaluated
  JUDGE_URL           OpenAI-compatible chat endpoint of the judge model
  JUDGE_MODEL         served name of the judge (flo-judge = Qwen, not the answer model)
  LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY   evaluation project
  REVISION            commit SHA being evaluated (Argo sets it on the Job)
  GITHUB_REPO         owner/name, used to find the PR and author of REVISION
  MIN_ROUTING         gate: share of cases with correct tool routing (default 0.85)
  MIN_GROUNDED        gate: share of judged answers rated grounded=yes (default 0.65)
  CONCURRENCY, LIMIT, DATASET_FILE
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
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
from pathlib import Path

AGENT_BASE = os.getenv("AGENT_BASE", "http://kagent-controller.docs-agent.svc.cluster.local:8083")
AGENT_NS = os.getenv("AGENT_NAMESPACE", "docs-agent-staging")
AGENTS = {"docs": "kubeflow-docs-agent", "debug": "kubeflow-debug-agent"}
JUDGE_URL = os.getenv("JUDGE_URL", "http://llm-judge.ml-infra.svc.cluster.local/v1/chat/completions")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "flo-judge")
LF = os.getenv("LANGFUSE_HOST", "http://langfuse-web.langfuse.svc.cluster.local:3000").rstrip("/")
LF_PK, LF_SK = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
REVISION = os.getenv("REVISION", "")
REPO = os.getenv("GITHUB_REPO", "kubeflow/docs-agent")
MIN_ROUTING = float(os.getenv("MIN_ROUTING", "0.85"))
MIN_GROUNDED = float(os.getenv("MIN_GROUNDED", "0.65"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "2"))
LIMIT = int(os.getenv("LIMIT", "0"))
DATASET_FILE = os.getenv("DATASET_FILE", str(Path(__file__).with_name("flo_eval_dataset.json")))
DATASET_NAME = "flo-e2e"


# ----- HTTP ----------------------------------------------------------------------------

def http(method: str, url: str, body=None, headers=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def lf(method: str, path: str, body=None, **params):
    auth = base64.b64encode(f"{LF_PK}:{LF_SK}".encode()).decode()
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return http(method, f"{LF}/api/public{path}{'?' + q if q else ''}", body, {"Authorization": f"Basic {auth}"}, timeout=60)



# ----- Cases ---------------------------------------------------------------------------

def load_cases() -> list[dict]:
    """Single-turn cases only: the multi-turn and canned-tool-result cases need fixtures."""
    cases = json.loads(Path(DATASET_FILE).read_text())["cases"]
    keep = [c for c in cases if "q" in c and "history" not in c and c["expect"]["type"] in ("call", "no_call") and c["agent"] in AGENTS]
    return keep[:LIMIT] if LIMIT else keep


# ----- Agent ---------------------------------------------------------------------------

def ask(case: dict, context_id: str) -> dict:
    url = f"{AGENT_BASE}/api/a2a/{AGENT_NS}/{AGENTS[case['agent']]}/"
    msg = {"kind": "message", "role": "user", "messageId": str(uuid.uuid4()), "contextId": context_id,
           "parts": [{"kind": "text", "text": case["q"]}]}
    t0 = time.time()
    try:
        res = http("POST", url, {"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {"message": msg}}, timeout=300)
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e), "latency": time.time() - t0, "calls": [], "context": "", "answer": ""}
    result = res.get("result") or {}
    calls, context = [], []
    for m in result.get("history", []):
        if m.get("role") != "agent":
            continue
        for p in m.get("parts", []):
            kind = (p.get("metadata") or {}).get("kagent_type", "")
            data = p.get("data") or {}
            if kind == "function_call":
                calls.append({"name": data.get("name"), "args": data.get("args") or {}})
            elif kind == "function_response":
                for c in (data.get("response") or {}).get("content", []) or []:
                    if isinstance(c, dict) and c.get("text"):
                        context.append(c["text"])
    answer = "\n".join(p.get("text", "") for a in result.get("artifacts", []) for p in a.get("parts", []) if p.get("kind") == "text")
    return {"error": res.get("error"), "state": (result.get("status") or {}).get("state"), "latency": time.time() - t0,
            "calls": calls, "context": "\n\n".join(context), "answer": answer.strip()}


# ----- Checks --------------------------------------------------------------------------

def routing_ok(case: dict, out: dict) -> bool:
    names = [c["name"] for c in out["calls"]]
    if case["expect"]["type"] == "no_call":
        return not names
    return any(n in case["expect"].get("tools", []) for n in names)


def fidelity_ok(case: dict, out: dict) -> bool | None:
    keep = case["expect"].get("keep")
    if not keep:
        return None
    queries = " ".join(json.dumps(c["args"]).lower() for c in out["calls"])
    return all(k.lower() in queries for k in keep)


JUDGE_PROMPT = """You grade an assistant's answer about Kubeflow.

Question:
{q}

Text the assistant's search tools returned:
<<<
{context}
>>>

Answer:
<<<
{answer}
>>>

Rate strictly:
- grounded: "yes" if every factual claim in the answer is supported by the returned text,
  "partial" if some claims are unsupported, "no" if the main claim is unsupported.
  Saying the information was not found counts as "yes" when the text indeed lacks it.
- answers: "yes" if the answer addresses the question, "partial", or "no".
Reply with JSON only: {{"grounded": "...", "answers": "...", "reason": "<one sentence>"}}"""


def judge(case: dict, out: dict) -> dict | None:
    if case["expect"]["type"] != "call" or not out["answer"]:
        return None
    prompt = JUDGE_PROMPT.format(q=case["q"], context=out["context"][:12000] or "(nothing)", answer=out["answer"][:4000])
    try:
        r = http("POST", JUDGE_URL, {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 200,
                                     "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        text = r["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        v = json.loads(m.group(0)) if m else {}
        return {k: str(v.get(k, "")).lower() for k in ("grounded", "answers")} | {"reason": v.get("reason", "")}
    except Exception as e:  # judge trouble must not hide agent results
        return {"grounded": "error", "answers": "error", "reason": str(e)[:200]}


# ----- Git context ---------------------------------------------------------------------

def git_context() -> dict:
    ctx = {"sha": REVISION, "pr": None, "author": None, "url": None}
    if not REVISION:
        return ctx
    try:
        pulls = http("GET", f"https://api.github.com/repos/{REPO}/commits/{REVISION}/pulls",
                     headers={"Accept": "application/vnd.github+json"}, timeout=20)
        if pulls:
            ctx |= {"pr": pulls[0]["number"], "author": pulls[0]["user"]["login"], "url": pulls[0]["html_url"]}
        else:
            c = http("GET", f"https://api.github.com/repos/{REPO}/commits/{REVISION}", timeout=20)
            ctx["author"] = (c.get("author") or {}).get("login")
    except Exception as e:
        print(f"warn: no GitHub context for {REVISION[:7]}: {e}", file=sys.stderr)
    return ctx


# ----- Langfuse ------------------------------------------------------------------------

def traces_by_session(since: str, sessions: set[str], wait_s: int = 120) -> dict[str, str]:
    found: dict[str, str] = {}
    deadline = time.time() + wait_s
    while time.time() < deadline and len(found) < len(sessions):
        cursor = None
        for _ in range(20):
            page = lf("GET", "/v2/observations", type="AGENT", fields="core,basic", limit=100,
                      fromStartTime=since, cursor=cursor)
            for o in page.get("data", []):
                if o.get("sessionId") in sessions:
                    found[o["sessionId"]] = o["traceId"]
            cursor = (page.get("meta") or {}).get("cursor")
            if not cursor:
                break
        if len(found) < len(sessions):
            time.sleep(10)
    return found


def record(run_name: str, run_meta: dict, rows: list[dict], since: str) -> None:
    if not (LF_PK and LF_SK):
        print("warn: no Langfuse keys, results not recorded", file=sys.stderr)
        return
    try:
        lf("POST", "/datasets", {"name": DATASET_NAME, "description": "Single-turn end-to-end cases from tests/eval/flo_eval_dataset.json"})
    except urllib.error.HTTPError as e:
        if e.code not in (409,):
            print(f"warn: dataset create: {e}", file=sys.stderr)
    traces = traces_by_session(since, {r["session"] for r in rows})
    for r in rows:
        c, item_id = r["case"], f"{DATASET_NAME}-{r['case']['id']}"
        try:
            lf("POST", "/dataset-items", {"datasetName": DATASET_NAME, "id": item_id,
                                          "input": {"q": c["q"], "agent": c["agent"]}, "expectedOutput": c["expect"],
                                          "metadata": {"group": c["group"]}})
        except urllib.error.HTTPError as e:
            print(f"warn: dataset item {item_id}: {e}", file=sys.stderr)
        trace_id = traces.get(r["session"])
        if not trace_id:
            print(f"warn: no trace for {c['id']} (session {r['session']})", file=sys.stderr)
            continue
        scores = [("routing", int(r["routing"]), "NUMERIC")]
        if r["fidelity"] is not None:
            scores.append(("query_fidelity", int(r["fidelity"]), "NUMERIC"))
        if r["judge"]:
            scores += [("grounded", r["judge"]["grounded"], "CATEGORICAL"), ("answers", r["judge"]["answers"], "CATEGORICAL")]
        for name, value, dtype in scores:
            try:
                lf("POST", "/scores", {"id": f"{trace_id}-{name}", "traceId": trace_id, "name": name, "value": value,
                                       "dataType": dtype, "comment": (r["judge"] or {}).get("reason", "") if dtype == "CATEGORICAL" else None,
                                       "metadata": run_meta})
            except urllib.error.HTTPError as e:
                print(f"warn: score {name} on {trace_id}: {e}", file=sys.stderr)
        try:
            lf("POST", "/dataset-run-items", {"runName": run_name, "runDescription": run_meta.get("description"),
                                              "metadata": run_meta, "datasetItemId": item_id, "traceId": trace_id})
        except urllib.error.HTTPError as e:
            print(f"warn: run item {item_id}: {e}", file=sys.stderr)


# ----- Main ----------------------------------------------------------------------------

def main() -> int:
    cases = load_cases()
    git = git_context()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha7 = (git["sha"] or "local")[:7]
    run_name = f"{sha7} " + (f"pr-{git['pr']} " if git["pr"] else "") + (f"@{git['author']} " if git["author"] else "") + stamp
    since = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"run {run_name}: {len(cases)} cases against {AGENT_NS}, judge {JUDGE_MODEL}")

    def one(case):
        session = f"eval-{sha7}-{stamp}-{case['id']}"
        out = ask(case, session)
        row = {"case": case, "session": session, "out": out, "routing": routing_ok(case, out), "fidelity": fidelity_ok(case, out)}
        row["judge"] = judge(case, out)
        return row

    with cf.ThreadPoolExecutor(CONCURRENCY) as pool:
        rows = list(pool.map(one, cases))

    n = len(rows)
    routing = sum(r["routing"] for r in rows) / n
    fid = [r["fidelity"] for r in rows if r["fidelity"] is not None]
    judged = [r["judge"] for r in rows if r["judge"]]
    grounded = sum(j["grounded"] == "yes" for j in judged) / len(judged) if judged else 0.0
    answers = sum(j["answers"] == "yes" for j in judged) / len(judged) if judged else 0.0
    errors = sum(bool(r["out"].get("error")) for r in rows)
    lat = sorted(r["out"]["latency"] for r in rows)
    summary = {"run": run_name, "sha": git["sha"], "pr": git["pr"], "author": git["author"], "cases": n, "errors": errors,
               "routing": round(routing, 3), "query_fidelity": round(sum(fid) / len(fid), 3) if fid else None,
               "grounded": round(grounded, 3), "answers": round(answers, 3), "judged": len(judged),
               "latency_p50_s": round(lat[n // 2], 2), "latency_p95_s": round(lat[min(n - 1, int(n * 0.95))], 2)}
    gates = {"routing": routing >= MIN_ROUTING, "grounded": grounded >= MIN_GROUNDED, "errors": errors == 0}
    summary["passed"] = all(gates.values())
    summary["gates"] = {"min_routing": MIN_ROUTING, "min_grounded": MIN_GROUNDED, **gates}

    for r in rows:
        if not r["routing"] or (r["judge"] and r["judge"]["grounded"] != "yes") or r["out"].get("error"):
            calls = ",".join(c["name"] for c in r["out"]["calls"]) or "-"
            print(f"  {r['case']['id']:<28} routing={'ok' if r['routing'] else 'FAIL'} calls={calls} "
                  f"grounded={(r['judge'] or {}).get('grounded', '-')} {r['out'].get('error') or ''}")

    record(run_name, {"sha": git["sha"], "pr": git["pr"], "author": git["author"], "pr_url": git["url"],
                      "namespace": AGENT_NS, "judge": JUDGE_MODEL, "description": f"{n} cases, routing {routing:.2f}, grounded {grounded:.2f}"},
           rows, since)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
