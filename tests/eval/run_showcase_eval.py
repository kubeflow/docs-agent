#!/usr/bin/env python3
"""Run the two-gate Flo showcase evaluation without submitting ingestion.

Gate 1 executes the named MCP tools inside the existing pod and requires every
golden Source URL. Gate 2 is structurally unreachable until all Gate 1 rows
pass, then opens a fresh A2A context for each query and scores tool use,
citations, required strings, and forbidden strings.

This script deliberately has no pipeline-submission code. Ingestion is an
operator-approved prerequisite documented in README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_RE = re.compile(r"^\*\*Source:\*\*\s+(https?://\S+)\s*$", re.MULTILINE)
MARKDOWN_URL_RE = re.compile(r"\[[^\]\n]+\]\((https?://[^\s)]+)\)")
BARE_URL_RE = re.compile(r"https?://[^\s<>\"'\]]+")
DEFAULT_AGENT_URL = "https://agent.santhoshtoorpu.com/a2a/docs-agent/kubeflow-docs-agent"
DEFAULT_SESSION_URL = "https://agent.santhoshtoorpu.com/api/session"
DEFAULT_ORIGIN = "https://kubeflowdemochatbot.netlify.app"


def tls_context() -> ssl.SSLContext:
    """Build a verified TLS context, including common macOS Python CA paths."""
    configured = os.getenv("SSL_CERT_FILE", "").strip()
    candidates = [
        configured,
        ssl.get_default_verify_paths().cafile or "",
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()


TLS_CONTEXT = tls_context()


@dataclass
class RetrievalResult:
    row: dict[str, Any]
    output: str
    source_urls: set[str]
    missing_sources: list[str]
    missing_evidence: list[str]

    @property
    def passed(self) -> bool:
        return not self.missing_sources


@dataclass
class AgentResult:
    row: dict[str, Any]
    answer: str
    tool_fired: bool
    tool_calls: list[str]
    tool_source_urls: set[str]
    cited_urls: set[str]
    invented_urls: list[str]
    missing_answer_strings: list[str]
    present_forbidden_strings: list[str]

    @property
    def passed(self) -> bool:
        return (
            self.tool_fired
            and not self.invented_urls
            and not self.missing_answer_strings
            and not self.present_forbidden_strings
        )


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["queries"]


def run_mcp_search(row: dict[str, Any], namespace: str, deployment: str, top_k: int) -> str:
    tool = row["tool"]
    if tool not in {"search_kubeflow_docs", "search_github_issues", "search_kubeflow_code"}:
        raise ValueError(f"Unsupported eval tool: {tool}")
    # Gate 1 evaluates the focused query the agent should send to the tool, not
    # necessarily the conversational wording the user sent to the agent.
    retrieval_query = row.get("retrieval_query", row["query"])
    snippet = f"import server; print(server.{tool}({retrieval_query!r}, top_k={top_k}))"
    command = [
        "kubectl",
        "exec",
        "-n",
        namespace,
        f"deploy/{deployment}",
        "--",
        "python",
        "-c",
        snippet,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def score_retrieval(row: dict[str, Any], output: str) -> RetrievalResult:
    sources = set(SOURCE_RE.findall(output))
    missing_sources = [url for url in row["expected_source_urls"] if url not in sources]
    # Evidence coverage is diagnostic rather than a hard Gate 1 condition. It
    # distinguishes "right document, incomplete/stale chunks" from a response
    # generation failure when the answer later misses a required fact.
    missing_evidence = [value for value in row["must_appear_in_answer"] if value not in output]
    return RetrievalResult(row, output, sources, missing_sources, missing_evidence)


def request_json(
    url: str,
    *,
    origin: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = 30,
) -> Any:
    data = json.dumps(payload).encode() if payload is not None else b""
    headers = {"Content-Type": "application/json", "Origin": origin}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:
        return json.loads(response.read())


def iter_sse_events(response: Any):
    """Yield complete JSON events; urlopen line iteration preserves SSE frames."""
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                if payload != "[DONE]":
                    yield json.loads(payload)
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            yield json.loads(payload)


def message_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    result = event.get("result") or {}
    return result.get("message") or (result.get("status") or {}).get("message")


def event_names_tool(event: Any, expected_tool: str) -> bool:
    if isinstance(event, dict):
        for key, value in event.items():
            if key in {"name", "toolName", "tool_name"} and value == expected_tool:
                return True
            if event_names_tool(value, expected_tool):
                return True
    elif isinstance(event, list):
        return any(event_names_tool(item, expected_tool) for item in event)
    return False


def strings_in_event(event: Any):
    """Yield string leaves so embedded MCP responses can be inspected safely."""
    if isinstance(event, str):
        yield event
    elif isinstance(event, dict):
        for value in event.values():
            yield from strings_in_event(value)
    elif isinstance(event, list):
        for item in event:
            yield from strings_in_event(item)


def tool_calls_in_event(event: Any, expected_tool: str) -> list[str]:
    """Return stable JSON argument snapshots for the expected named tool."""
    calls: list[str] = []
    if isinstance(event, dict):
        names = [event.get(key) for key in ("name", "toolName", "tool_name")]
        if expected_tool in names:
            for key in ("args", "arguments"):
                if key in event:
                    calls.append(json.dumps(event[key], sort_keys=True, ensure_ascii=False))
        for value in event.values():
            calls.extend(tool_calls_in_event(value, expected_tool))
    elif isinstance(event, list):
        for item in event:
            calls.extend(tool_calls_in_event(item, expected_tool))
    return calls


def extract_answer_urls(answer: str) -> set[str]:
    """Extract Markdown and bare URLs without retaining Markdown's closing parenthesis."""
    markdown_urls = set(MARKDOWN_URL_RE.findall(answer))
    remaining = MARKDOWN_URL_RE.sub("", answer)
    bare_urls = {url.rstrip(".,;:)") for url in BARE_URL_RE.findall(remaining)}
    return markdown_urls | bare_urls


def select_answer(final_text: str, streamed: list[str]) -> str:
    """Prefer Kagent's authoritative aggregate, with partials as fallback."""
    return final_text.strip() or "".join(streamed).strip()


def run_agent(
    row: dict[str, Any],
    allowed_urls: set[str],
    *,
    agent_url: str,
    session_url: str,
    origin: str,
    timeout: int,
) -> AgentResult:
    session = request_json(session_url, origin=origin, timeout=30)
    token = session.get("access_token")
    payload = {
        "jsonrpc": "2.0",
        "method": "message/stream",
        "params": {
            "message": {
                "kind": "message",
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": row["query"]}],
                "contextId": str(uuid.uuid4()),
                "metadata": {"displaySource": "user"},
            },
            "metadata": {},
        },
        "id": str(uuid.uuid4()),
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Origin": origin,
        "Authorization": f"Bearer {token}",
    }
    request = urllib.request.Request(agent_url, data=json.dumps(payload).encode(), headers=headers, method="POST")

    streamed: list[str] = []
    final_text = ""
    tool_fired = False
    tool_calls: list[str] = []
    tool_source_urls: set[str] = set()
    with urllib.request.urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:
        for event in iter_sse_events(response):
            tool_fired = tool_fired or event_names_tool(event, row["tool"])
            for call in tool_calls_in_event(event, row["tool"]):
                if call not in tool_calls:
                    tool_calls.append(call)
            for value in strings_in_event(event):
                tool_source_urls.update(SOURCE_RE.findall(value))
            message = message_from_event(event)
            if not message or message.get("role") == "user":
                continue
            text = "".join(part.get("text", "") for part in message.get("parts", []) if part.get("kind") == "text")
            if not text:
                continue
            is_partial = (message.get("metadata") or {}).get("kagent_adk_partial")
            if is_partial is False:
                final_text = text
            else:
                streamed.append(text)

    # Kagent emits incremental partial text and an authoritative aggregated
    # final message. Prefer the final message; partials are only a fallback for
    # interrupted streams that never deliver aggregation.
    answer = select_answer(final_text, streamed)
    cited_urls = extract_answer_urls(answer)
    invented = sorted(cited_urls - allowed_urls)
    missing = [value for value in row["must_appear_in_answer"] if value not in answer]
    forbidden = [value for value in row["forbidden_in_answer"] if value in answer]
    return AgentResult(
        row,
        answer,
        tool_fired,
        tool_calls,
        tool_source_urls,
        cited_urls,
        invented,
        missing,
        forbidden,
    )


def print_retrieval(result: RetrievalResult) -> None:
    status = "PASS" if result.passed else "BLOCKED"
    print(f"[{status}] retrieval {result.row['id']}")
    if result.missing_sources:
        print(f"  missing Source URLs: {', '.join(result.missing_sources)}")
    if result.missing_evidence:
        print(f"  retrieval evidence missing: {', '.join(result.missing_evidence)}")
    print(f"  returned Source URLs: {len(result.source_urls)}")


def print_agent(result: AgentResult, *, verbose: bool = False) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] agent {result.row['id']}")
    print(f"  named tool fired: {result.tool_fired}")
    if result.tool_source_urls:
        print(f"  Source URLs observed in agent tool events: {len(result.tool_source_urls)}")
    if result.invented_urls:
        print(f"  URLs absent from MCP Source lines: {', '.join(result.invented_urls)}")
    if result.missing_answer_strings:
        print(f"  answer strings missing: {', '.join(result.missing_answer_strings)}")
    if result.present_forbidden_strings:
        print(f"  forbidden strings present: {', '.join(result.present_forbidden_strings)}")
    if verbose:
        if result.tool_calls:
            print("  named tool arguments:")
            for call in result.tool_calls:
                print(f"    {call}")
        print("  final answer:")
        print("    " + result.answer.replace("\n", "\n    "))
        if result.tool_source_urls:
            print("  agent tool Source URLs:")
            for url in sorted(result.tool_source_urls):
                print(f"    {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=Path(__file__).with_name("queries.json"))
    parser.add_argument("--namespace", default="docs-agent")
    parser.add_argument("--deployment", default="mcp-kubeflow-docs")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--agent-url", default=DEFAULT_AGENT_URL)
    parser.add_argument("--session-url", default=DEFAULT_SESSION_URL)
    parser.add_argument("--origin", default=DEFAULT_ORIGIN)
    parser.add_argument("--timeout", type=int, default=330)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--row-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.queries)
    if args.row_id:
        requested = set(args.row_id)
        rows = [row for row in rows if row["id"] in requested]
        missing_ids = sorted(requested - {row["id"] for row in rows})
        if missing_ids:
            raise ValueError(f"Unknown --row-id values: {', '.join(missing_ids)}")

    retrieval_results: list[RetrievalResult] = []
    for row in rows:
        output = run_mcp_search(row, args.namespace, args.deployment, args.top_k)
        result = score_retrieval(row, output)
        retrieval_results.append(result)
        print_retrieval(result)

    blocked = [result for result in retrieval_results if not result.passed]
    if blocked:
        print("\nGate 2 not run: at least one expected Source URL is absent from MCP results.")
        return 2
    if args.retrieval_only:
        print("\nGate 1 passed; --retrieval-only requested, so no session or agent POST was made.")
        return 0

    agent_results = []
    for retrieval in retrieval_results:
        result = run_agent(
            retrieval.row,
            retrieval.source_urls,
            agent_url=args.agent_url,
            session_url=args.session_url,
            origin=args.origin,
            timeout=args.timeout,
        )
        agent_results.append(result)
        print_agent(result, verbose=args.verbose)

    return 0 if all(result.passed for result in agent_results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.CalledProcessError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"eval infrastructure error: {error}", file=sys.stderr)
        raise SystemExit(3) from error
