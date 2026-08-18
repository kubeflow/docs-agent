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
