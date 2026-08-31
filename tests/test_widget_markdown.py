"""Behavioral tests for the widget's dependency-free Markdown formatter."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


CHATBOT_JS = Path(__file__).parent.parent / "frontend" / "docs_scripts" / "chatbot.js"


def run_formatter(text: str, *, streaming: bool = False) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for widget formatter tests")

    script = f"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync({json.dumps(str(CHATBOT_JS))}, 'utf8');
const prelude = source.split("document.addEventListener('DOMContentLoaded'")[0];
const context = {{}};
vm.createContext(context);
vm.runInContext(prelude, context);
process.stdout.write(context.formatChatMarkdown(
  {json.dumps(text)},
  {json.dumps(streaming)}
));
"""
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def run_sse_parser(chunks: list[str]) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for widget SSE parser tests")

    script = f"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync({json.dumps(str(CHATBOT_JS))}, 'utf8');
const prelude = source.split("document.addEventListener('DOMContentLoaded'")[0];
const context = {{}};
vm.createContext(context);
vm.runInContext(prelude, context);
const parser = context.createSSEFrameParser();
const events = [];
for (const chunk of {json.dumps(chunks)}) {{
  events.push(...parser.push(chunk));
}}
events.push(...parser.finish());
process.stdout.write(JSON.stringify({{
  events,
  eventsAfterSecondFinish: parser.finish()
}}));
"""
    completed = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_linkifies_only_http_sources_with_safe_anchor_attributes():
    rendered = run_formatter("[Katib Experiment](https://www.kubeflow.org/docs/components/katib/)")

    assert rendered == (
        '<a href="https://www.kubeflow.org/docs/components/katib/" '
        'target="_blank" rel="noopener noreferrer">Katib Experiment</a>'
    )
    assert run_formatter("[unsafe](javascript:alert(1))") == ("[unsafe](javascript:alert(1))")


def test_escapes_markdown_link_label_and_query_delimiter():
    rendered = run_formatter("[<img src=x>](https://example.test/docs?a=1&b=2)")

    assert "&lt;img src=x&gt;" in rendered
    assert 'href="https://example.test/docs?a=1&amp;b=2"' in rendered
    assert "<img" not in rendered


@pytest.mark.parametrize("streaming", [False, True])
def test_preserves_dollar_sequences_and_links_inside_fenced_yaml(streaming):
    closing_fence = "" if streaming else "```"
    markdown = (
        "```yaml\n"
        "command: ${trialParameters.learningRate}\n"
        "replacement: '$&-$1'\n"
        "source: [literal](https://example.test/not-a-link)\n"
        f"{closing_fence}"
    )

    rendered = run_formatter(markdown, streaming=streaming)

    assert "${trialParameters.learningRate}" in rendered
    assert "$&amp;-$1" in rendered
    assert "[literal](https://example.test/not-a-link)" in rendered
    assert "<a " not in rendered
    assert '<pre><code class="language-yaml">' in rendered


def test_does_not_linkify_markdown_inside_inline_code():
    rendered = run_formatter("Use `[title](https://example.test/literal)` then [open docs](https://example.test/docs).")

    assert "<code>[title](https://example.test/literal)</code>" in rendered
    assert rendered.count("<a ") == 1


def test_sse_parser_preserves_fragmented_tool_and_citation_events():
    tool_event = {
        "result": {
            "message": {
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "name": "search_kubeflow_docs",
                            "args": {"query": "Katib Experiment"},
                        },
                    }
                ]
            }
        }
    }
    citation_event = {
        "result": {
            "message": {
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "name": "search_kubeflow_docs",
                            "structuredContent": {
                                "citations": [
                                    {
                                        "title": "Configure a Katib Experiment",
                                        "url": "https://www.kubeflow.org/docs/components/katib/",
                                    }
                                ]
                            },
                        },
                    }
                ]
            }
        }
    }
    final_event = {
        "result": {
            "message": {
                "role": "agent",
                "parts": [{"kind": "text", "text": "Katib uses Experiments."}],
            }
        }
    }

    pretty_citation = json.dumps(citation_event, indent=2)
    citation_frame = "\r\n".join(f"data: {line}" for line in pretty_citation.splitlines())
    wire = (
        ": heartbeat\r\n\r\n"
        f"data: {json.dumps(tool_event)}\r\n\r\n"
        f"{citation_frame}\r\n\r\n"
        f"data: {json.dumps(final_event)}"
    )

    # Split inside CRLF delimiters, JSON keys, and values to model arbitrary
    # ReadableStream chunk boundaries. The final frame intentionally has no
    # blank-line terminator and must be flushed when the stream closes.
    split_points = [1, 14, 15, 37, 73, wire.index("citations") + 4, len(wire) - 9]
    chunks = []
    start = 0
    for end in split_points:
        chunks.append(wire[start:end])
        start = end
    chunks.append(wire[start:])

    parsed = run_sse_parser(chunks)

    assert len(parsed["events"]) == 3
    decoded = [json.loads(event) for event in parsed["events"]]
    assert decoded[0] == tool_event
    assert decoded[1] == citation_event
    assert (
        decoded[1]["result"]["message"]["parts"][0]["data"]["structuredContent"]["citations"]
        == citation_event["result"]["message"]["parts"][0]["data"]["structuredContent"]["citations"]
    )
    assert decoded[2] == final_event
    assert parsed["eventsAfterSecondFinish"] == []


def test_sse_parser_handles_single_byte_lf_frames_without_duplicates():
    first = json.dumps({"result": {"taskId": "task-1"}})
    second = json.dumps({"result": {"final": True}})
    wire = f"data:{first}\n\ndata: {second}\n\n"

    parsed = run_sse_parser(list(wire))

    assert parsed["events"] == [first, second]
    assert parsed["eventsAfterSecondFinish"] == []
