"""Flo tool-use eval at the LLM layer (no kagent, no MCP; tool_choice auto).

Sends the live Flo system prompts + MCP tool schemas (plus kagent's ask_user) to an
OpenAI-compatible endpoint and scores the first assistant turn per case. Canned tool
results are rendered with mcp-server/citations.py so the model sees prod formatting.

Active model (llm-runtime chart, stable alias flo-llm):
  kubectl -n ml-infra port-forward svc/llm-stable 8081:80
  python tests/eval/slm_tool_eval.py run --name flo-llm \
      --base-url http://localhost:8081/v1 --model flo-llm --repeats 3

Qwen baseline: the qwen2.5-7b-awq candidate below, or llm-runtime with model=qwen.

Candidate on the spare A10:
  python tests/eval/slm_tool_eval.py manifest granite-4.2-3b | kubectl apply -f -
  kubectl -n slm-eval port-forward svc/slm 8000:8000
  python tests/eval/slm_tool_eval.py run --name granite-4.2-3b --model granite-4.2-3b \
      --thinking off
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "flo_eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "slm_results"

sys.path.insert(0, str(ROOT / "docs-agent-mcp/mcp-server"))
from citations import format_code_hits, format_docs_hits, format_issues_hits  # noqa: E402

URL_RE = re.compile(r"https?://|www\.|\]\(")
LABEL_RE = re.compile(r"\[c\d+\]")
PATH_RE = re.compile(r"github\.com|\b[\w.-]+/[\w.-]+/[\w./-]*\.(?:ya?ml|py|go|json|md)\b")
FACT_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)*\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{3,}\b")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
TOOLCALL_LEAK_RE = re.compile(r"<tool_call>|<function=|<\|tool_call|\"name\":\s*\"(?:search_|ask_user)")
CODE_RE = re.compile(r"```.*?```", re.S)
BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.M)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*\"'(])")
NOTFOUND_RE = re.compile(
    r"not (?:directly |explicitly )?(?:found|covered|documented|mentioned|available|specified|in the indexed)"
    r"|\bno\b[^.]{0,80}\b(?:found|documented|available)\b"
    r"|(?:could not|couldn't|unable to) find"
    r"|does(?:n't| not) (?:have|mention|specify|cover|contain|include)"
    r"|no (?:results|information|evidence|relevant|matching|documentation|direct)"
    r"|don't have|do not have|did(?:n't| not) return|isn't covered",
    re.I,
)

# Optional: pin the eval pod to one GPU node by hostname; otherwise any GPU node.
NODE = os.environ.get("SLM_EVAL_NODE", "")
IMAGE = "docker.io/vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
NO_THINK = '{"enable_thinking": false}'

CANDIDATES = {
    "granite-4.2-3b": {
        "hf": "ibm-granite/granite-4.2-3b",
        "revision": "e459acceac81e5fe67c07d9cfc72329a332e7eb1",
        "args": ["--tool-call-parser", "qwen3_coder", "--default-chat-template-kwargs", NO_THINK],
    },
    "qwen3.5-4b": {
        "hf": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "args": [
            "--tool-call-parser",
            "qwen3_coder",
            "--reasoning-parser",
            "qwen3",
            "--default-chat-template-kwargs",
            NO_THINK,
            "--limit-mm-per-prompt",
            '{"image": 0, "video": 0}',
        ],
    },
    "gemma-4-e4b": {
        "hf": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "args": [
            "--tool-call-parser",
            "gemma4",
            "--reasoning-parser",
            "gemma4",
            "--default-chat-template-kwargs",
            NO_THINK,
            "--limit-mm-per-prompt",
            '{"image": 0, "audio": 0}',
        ],
    },
    "qwen2.5-7b-awq": {
        "hf": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "revision": "b25037543e9394b818fdfca67ab2a00ecc7dd641",
        "args": ["--quantization", "awq", "--tool-call-parser", "hermes"],
    },
}

TOOL_SCHEMAS = {
    "search_kubeflow_docs": {
        "description": "Search Kubeflow documentation. Search mode is chosen here, not by the LLM.",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}},
    },
    "search_github_issues": {
        "description": "Search Kubeflow GitHub issues.",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "repo": {"type": "string", "default": ""},
            "state": {"type": "string", "default": ""},
        },
    },
    "search_kubeflow_code": {
        "description": "Search Kubeflow code and YAML manifests.",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "resource_kind": {"type": "string", "default": ""},
            "repo": {"type": "string", "default": ""},
        },
    },
}
ASK_USER = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": "Ask the user one or more clarifying questions and wait for the answers.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "choices": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["question"],
                    },
                }
            },
            "required": ["questions"],
        },
    },
}


def tool_defs(names: list[str]) -> list[dict]:
    defs = []
    for name in names:
        if name == "ask_user":
            defs.append(ASK_USER)
            continue
        schema = TOOL_SCHEMAS[name]
        defs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema["description"],
                    "parameters": {"type": "object", "properties": schema["properties"], "required": ["query"]},
                },
            }
        )
    return defs


def render_fixture(fixture: dict) -> str:
    if fixture["kind"] == "text":
        return fixture["text"]
    formatter = {"docs": format_docs_hits, "issues": format_issues_hits, "code": format_code_hits}[fixture["kind"]]
    body, _ = formatter(fixture["hits"])
    return body


def build_messages(case: dict, system: str, fixtures: dict) -> tuple[list[dict], list[str]]:
    messages = [{"role": "system", "content": system}]
    prior_queries: list[str] = []
    for n, item in enumerate(case.get("history", [])):
        if "user" in item:
            messages.append({"role": "user", "content": item["user"]})
        elif "assistant" in item:
            messages.append({"role": "assistant", "content": item["assistant"]})
        else:
            call_id = f"call_{n}"
            args = {"query": item["query"], "top_k": 5}
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": item["tool"], "arguments": json.dumps(args)},
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": render_fixture(fixtures[item["fixture"]])}
            )
            prior_queries.append(item["query"])
    if case.get("q"):
        messages.append({"role": "user", "content": case["q"]})
    return messages, prior_queries


def chat(args, body: dict) -> tuple[dict, float]:
    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=raw, headers={"Content-Type": "application/json"}
    )
    for attempt in range(30):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                return json.loads(resp.read()), time.perf_counter() - start
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException) as exc:
            if isinstance(exc, urllib.error.HTTPError) or attempt == 29:
                raise
            time.sleep(3)
    raise RuntimeError("unreachable")


def sentence_count(text: str) -> int:
    text = CODE_RE.sub("", text).strip()
    bullets = BULLET_RE.findall(text)
    if bullets:
        return len(bullets)
    return len([s for s in SENTENCE_SPLIT.split(text) if s.strip()]) if text else 0


def ungrounded_facts(text: str, evidence: str) -> list[str]:
    missing = []
    for fact in FACT_RE.findall(text):
        if fact.lstrip("vV") not in evidence:
            missing.append(fact)
    return sorted(set(missing))


def has_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


def score(case: dict, agent: dict, msg: dict, messages: list[dict], prior_queries: list[str]) -> dict:
    row = _score(case, agent, msg, messages, prior_queries)
    if row.get("parse_error"):
        row["pass"] = False
    return row


def _score(case: dict, agent: dict, msg: dict, messages: list[dict], prior_queries: list[str]) -> dict:
    exp = case["expect"]
    calls = msg.get("tool_calls") or []
    names = [c.get("function", {}).get("name") for c in calls]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    if "</think>" in content:
        reasoning = reasoning or content.rsplit("</think>", 1)[0]
        content = content.rsplit("</think>", 1)[1]
    text = THINK_RE.sub("", content).strip()
    row = {"tools": names, "text": text, "issues": [], "reasoning_chars": len(reasoning)}
    if TOOLCALL_LEAK_RE.search(text):
        row["parse_error"] = True
        row["issues"].append("unparsed tool call in content")
    if "ask_user" in names:
        row["issues"].append("called ask_user")

    if exp["type"] == "call":
        row["decision_ok"] = bool(calls)
        row["tool_ok"] = bool(names) and names[0] in exp["tools"]
        if len(calls) > agent["max_calls"]:
            row["issues"].append(f"{len(calls)} calls")
        args = {}
        if calls:
            try:
                args = json.loads(calls[0]["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                row["issues"].append("bad JSON args")
        query = args.get("query") if isinstance(args, dict) else None
        row["query"] = query
        row["args"] = args if isinstance(args, dict) else None
        q = (query or "").lower()
        missing = [k for k in exp.get("keep", []) if k.lower() not in q]
        if exp.get("keep_any") and not any(k.lower() in q for k in exp["keep_any"]):
            missing.append("any of " + "/".join(exp["keep_any"]))
        added = [k for k in exp.get("forbid_in_query", []) if k.lower() in q]
        if missing:
            row["issues"].append(f"dropped {missing}")
        if added:
            row["issues"].append(f"added {added}")
        if calls and not row["tool_ok"]:
            row["issues"].append(f"wrong tool {names[0]}")
        if not calls:
            row["issues"].append("no tool call")
        if text and calls:
            row["preamble"] = True
        row["pass"] = (
            row["tool_ok"]
            and bool(query and query.strip())
            and not missing
            and not added
            and len(calls) <= agent["max_calls"]
            and "ask_user" not in names
        )
        return row

    if exp["type"] == "no_call":
        row["decision_ok"] = not calls
        leaked = [m for m in exp.get("must_not", []) if m.lower() in text.lower()]
        if leaked:
            row["issues"].append(f"leaked {leaked}")
        on_topic = not exp.get("must_any") or has_any(text, exp["must_any"])
        if text and not on_topic:
            row["issues"].append("reply ignores the question")
        if calls:
            row["issues"].append("unexpected tool call")
        if not text and not calls:
            row["issues"].append("empty reply")
        row["pass"] = not calls and bool(text) and not leaked and on_topic
        return row

    allowed = exp.get("allow_retry", [])
    if calls:
        query = ""
        try:
            query = json.loads(calls[0]["function"].get("arguments") or "{}").get("query", "")
        except (json.JSONDecodeError, AttributeError):
            pass
        fresh = query.strip().lower() not in {p.lower() for p in prior_queries}
        retry_ok = len(calls) == 1 and names[0] in allowed and fresh
        row["decision_ok"] = retry_ok
        row["retried"] = True
        row["query"] = query
        if not retry_ok:
            row["issues"].append("tool call instead of answer" if not allowed else "bad retry")
        row["pass"] = retry_ok
        return row

    row["decision_ok"] = True
    evidence = "\n".join(m["content"] for m in messages if m["role"] in ("tool", "user") and m.get("content"))
    checks = {
        "non_empty": bool(text),
        "no_urls": not URL_RE.search(text),
        "no_labels": not LABEL_RE.search(text),
    }
    if agent.get("no_paths"):
        checks["no_paths"] = not PATH_RE.search(text)
    if agent.get("max_sentences"):
        n = sentence_count(text)
        row["sentences"] = n
        checks["length"] = n <= agent["max_sentences"]
    if exp.get("must_all"):
        checks["must_all"] = all(m.lower() in text.lower() for m in exp["must_all"])
    if exp.get("must_any"):
        checks["must_any"] = has_any(text, exp["must_any"])
    if exp.get("notfound"):
        checks["notfound"] = bool(NOTFOUND_RE.search(text))
    if exp.get("must_not"):
        checks["must_not"] = not has_any(text, exp["must_not"])
    if exp.get("must_not_all"):
        checks["must_not_all"] = not all(m.lower() in text.lower() for m in exp["must_not_all"])
    ungrounded = ungrounded_facts(text, evidence)
    checks["grounded"] = not ungrounded
    if ungrounded:
        row["ungrounded"] = ungrounded
    row["checks"] = checks
    row["issues"] += [f"failed {k}" for k, ok in checks.items() if not ok]
    row["pass"] = all(checks.values())
    return row


def run(args: argparse.Namespace) -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    agents = data["agents"]
    systems = {name: (ROOT / a["prompt_file"]).read_text(encoding="utf-8") for name, a in agents.items()}
    if args.docs_prompt:
        systems["docs"] = Path(args.docs_prompt).read_text(encoding="utf-8")
    tools = {name: tool_defs(a["tools"]) for name, a in agents.items()}
    cases = [c for c in data["cases"] if not args.only or args.only in c["id"] or args.only == c["group"]]
    rows = []
    for rep in range(1, args.repeats + 1):
        for case in cases:
            agent = agents[case["agent"]]
            messages, prior = build_messages(case, systems[case["agent"]], data["fixtures"])
            body = {
                "model": args.model,
                "messages": messages,
                "tools": tools[case["agent"]],
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            if args.thinking != "unset":
                body["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
            try:
                data_resp, latency = chat(args, body)
                row = score(case, agent, data_resp["choices"][0]["message"], messages, prior)
                usage = data_resp.get("usage") or {}
                row.update(
                    latency_s=round(latency, 2),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=data_resp["choices"][0].get("finish_reason"),
                )
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.HTTPException,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as exc:
                detail = (
                    exc.read().decode(errors="replace")[:300] if isinstance(exc, urllib.error.HTTPError) else str(exc)
                )
                row = {"pass": False, "decision_ok": False, "tools": [], "issues": [f"request failed: {detail}"]}
            row.update(id=case["id"], agent=case["agent"], group=case["group"], type=case["expect"]["type"], rep=rep)
            rows.append(row)
            mark = "PASS" if row["pass"] else "FAIL"
            print(
                f"[{rep}] {mark}  {case['id']:<28} tools={row['tools']} {row.get('latency_s', '-')}s  {'; '.join(row['issues'])}"
            )
            if not row["pass"]:
                detail = row.get("query") or row.get("text", "")
                if detail:
                    print(f"        -> {detail[:180]!r}")

    summary = summarize(args, rows, cases)
    print(json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return 0


def stream_once(args, body: dict) -> dict:
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("tool_calls")
            ):
                first = time.perf_counter()
    end = time.perf_counter()
    tokens = usage.get("completion_tokens") or 0
    first = first or end
    return {
        "ttft_s": round(first - start, 3),
        "total_s": round(end - start, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": tokens,
        "decode_tps": round((tokens - 1) / (end - first), 1) if tokens > 1 and end > first else None,
    }


def speed(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    agent = data["agents"]["docs"]
    system = (ROOT / agent["prompt_file"]).read_text(encoding="utf-8")
    docs = [render_fixture(f) for f in data["fixtures"].values() if f["kind"] == "docs"]
    evidence = ""
    while len(evidence) < 12000:
        evidence += "\n\n".join(docs) + "\n\n"
    args_json = json.dumps({"query": "Kubeflow Katib KFP install overview", "top_k": 5})
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Give me an overview of Katib, Pipelines and installing Kubeflow."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "search_kubeflow_docs", "arguments": args_json},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": evidence[:12000]},
        ],
        "tools": tool_defs(agent["tools"]),
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": args.output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.thinking != "unset":
        body["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}

    stream_once(args, body)
    single = [stream_once(args, body) for _ in range(args.runs)]
    result = {
        "name": args.name,
        "model": args.model,
        "prompt_tokens": single[0]["prompt_tokens"],
        "output_tokens": args.output_tokens,
        "single": {
            "ttft_s_median": statistics.median(r["ttft_s"] for r in single),
            "decode_tps_median": statistics.median(r["decode_tps"] for r in single if r["decode_tps"]),
            "runs": single,
        },
    }
    if args.concurrency > 1:
        start = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as pool:
            par = list(pool.map(lambda _: stream_once(args, body), range(args.concurrency * 2)))
        wall = time.perf_counter() - start
        result["concurrent"] = {
            "concurrency": args.concurrency,
            "ttft_s_median": statistics.median(r["ttft_s"] for r in par),
            "per_stream_tps_median": statistics.median(r["decode_tps"] for r in par if r["decode_tps"]),
            "aggregate_tps": round(sum(r["completion_tokens"] for r in par) / wall, 1),
        }
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "single"}
            | {"single": {k: v for k, v in result["single"].items() if k != "runs"}},
            indent=2,
        )
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.name}-speed.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return 0


def rescore(args: argparse.Namespace) -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    systems = {name: (ROOT / a["prompt_file"]).read_text(encoding="utf-8") for name, a in data["agents"].items()}
    path = RESULTS_DIR / f"{args.name}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in data["cases"]}
    changed = 0
    for row in saved["rows"]:
        case = by_id[row["id"]]
        # Rows saved before full-text capture were cut at 400 chars and can't be rescored reliably.
        rescorable_answer = row["type"] == "answer" and not row.get("retried") and "checks" in row
        rescorable_no_call = row["type"] == "no_call" and not row.get("tools") and "text" in row
        if not (rescorable_answer or rescorable_no_call) or len(row["text"]) == 400:
            continue
        messages, prior = build_messages(case, systems[case["agent"]], data["fixtures"])
        fresh = score(case, data["agents"][case["agent"]], {"content": row["text"]}, messages, prior)
        if fresh["pass"] != row["pass"]:
            changed += 1
            print(f"[{row['rep']}] {'PASS' if fresh['pass'] else 'FAIL'}  {row['id']:<28} {'; '.join(fresh['issues'])}")
        for key in ("checks", "issues", "pass", "ungrounded", "sentences", "decision_ok"):
            row.pop(key, None)
        fresh["issues"] = fresh.get("issues", [])
        row["decision_ok"] = fresh.get("decision_ok")
        row.update({k: fresh[k] for k in ("checks", "issues", "pass", "ungrounded", "sentences") if k in fresh})
    ns = argparse.Namespace(**{k: saved["summary"][k] for k in ("name", "model", "tool_choice", "thinking", "repeats")})
    cases = [by_id[i] for i in dict.fromkeys(r["id"] for r in saved["rows"])]
    saved["summary"] = summarize(ns, saved["rows"], cases)
    print(json.dumps(saved["summary"], indent=2))
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"rescored {changed} rows, saved {path}")
    return 0


def summarize(args, rows: list[dict], cases: list[dict]) -> dict:
    def rate(sel):
        return f"{sum(r['pass'] for r in sel)}/{len(sel)}" if sel else "-"

    by_group, by_type, by_agent, per_case = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
        by_type[r["type"]].append(r)
        by_agent[r["agent"]].append(r)
        per_case[r["id"]].append(r["pass"])
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    completion = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    answers = [r for r in rows if r["type"] == "answer" and r.get("checks")]
    return {
        "name": args.name,
        "model": args.model,
        "tool_choice": "auto",
        "thinking": args.thinking,
        "repeats": args.repeats,
        "cases": len(cases),
        "overall": rate(rows),
        "by_agent": {k: rate(v) for k, v in sorted(by_agent.items())},
        "by_type": {k: rate(v) for k, v in sorted(by_type.items())},
        "by_group": {k: rate(v) for k, v in sorted(by_group.items())},
        "decision_accuracy": f"{sum(bool(r.get('decision_ok')) for r in rows)}/{len(rows)}",
        "answer_checks_failed": {
            k: sum(1 for r in answers if not r["checks"].get(k, True))
            for k in (
                "no_urls",
                "no_labels",
                "no_paths",
                "length",
                "grounded",
                "must_all",
                "must_any",
                "notfound",
                "must_not",
                "must_not_all",
            )
        },
        "tool_parse_errors": sum(1 for r in rows if r.get("parse_error") or "bad JSON args" in r.get("issues", [])),
        "preamble_with_call": sum(1 for r in rows if r.get("preamble")),
        "ask_user_calls": sum(1 for r in rows if "ask_user" in r.get("tools", [])),
        "request_errors": sum(1 for r in rows if any(i.startswith("request failed") for i in r["issues"])),
        "flaky_cases": sorted(k for k, v in per_case.items() if 0 < sum(v) < len(v)),
        "always_failing": sorted(k for k, v in per_case.items() if not any(v)),
        "latency_s": {
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "completion_tokens_mean": round(statistics.mean(completion), 1) if completion else None,
    }


def manifest(args: argparse.Namespace) -> int:
    cand = CANDIDATES[args.candidate]
    vllm_args = [
        "--model",
        cand["hf"],
        "--revision",
        cand["revision"],
        "--served-model-name",
        args.candidate,
        "--port",
        "8000",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-auto-tool-choice",
        *cand["args"],
    ]
    labels = {"app": "slm"}
    docs = [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "slm-eval"}},
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": "hf-cache", "namespace": "slm-eval"},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": "oci-bv",
                "resources": {"requests": {"storage": "50Gi"}},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "slm", "namespace": "slm-eval", "labels": labels},
            "spec": {"selector": labels, "ports": [{"name": "http", "port": 8000, "targetPort": 8000}]},
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "slm", "namespace": "slm-eval", "labels": labels},
            "spec": {
                "replicas": 1,
                "strategy": {"type": "Recreate"},
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels, "annotations": {"slm/candidate": args.candidate}},
                    "spec": {
                        "nodeSelector": {"kubernetes.io/hostname": NODE} if NODE else {},
                        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                        "containers": [
                            {
                                "name": "vllm",
                                "image": IMAGE,
                                "imagePullPolicy": "IfNotPresent",
                                "args": vllm_args,
                                "env": [{"name": "HF_HOME", "value": "/cache"}],
                                "ports": [{"containerPort": 8000}],
                                "startupProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "periodSeconds": 10,
                                    "failureThreshold": 180,
                                },
                                "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 10},
                                "resources": {
                                    "requests": {"cpu": "4", "memory": "24Gi", "nvidia.com/gpu": "1"},
                                    "limits": {"memory": "48Gi", "nvidia.com/gpu": "1"},
                                },
                                "volumeMounts": [
                                    {"name": "cache", "mountPath": "/cache"},
                                    {"name": "shm", "mountPath": "/dev/shm"},
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "cache", "persistentVolumeClaim": {"claimName": "hf-cache"}},
                            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
                        ],
                    },
                },
            },
        },
    ]
    for doc in docs:
        print("---")
        print(json.dumps(doc, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--thinking", choices=["unset", "on", "off"], default="unset")
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--timeout", type=int, default=180)
    r.add_argument("--only", default="", help="case id substring or group name")
    r.add_argument("--docs-prompt", default="", help="override the docs agent system prompt file")
    m = sub.add_parser("manifest")
    m.add_argument("candidate", choices=sorted(CANDIDATES))
    s = sub.add_parser("rescore", help="re-apply answer checks to a saved run without calling the model")
    s.add_argument("--name", required=True)
    sp = sub.add_parser("speed", help="streamed TTFT and decode tokens/s on a ~3K-token docs answer")
    sp.add_argument("--name", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--base-url", default="http://localhost:8000/v1")
    sp.add_argument("--thinking", choices=["unset", "on", "off"], default="unset")
    sp.add_argument("--runs", type=int, default=5)
    sp.add_argument("--concurrency", type=int, default=4)
    sp.add_argument("--output-tokens", type=int, default=256)
    sp.add_argument("--timeout", type=int, default=180)
    a = parser.parse_args()
    return {"run": run, "manifest": manifest, "rescore": rescore, "speed": speed}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
