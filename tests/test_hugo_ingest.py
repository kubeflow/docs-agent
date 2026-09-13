import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../docs-agent-mcp/pipelines/utils'))
from hugo_ingest import clean_hugo_markdown, parse_frontmatter, process_html_table

def test_frontmatter_extraction():
    content = """+++
title = "Kubeflow 26.03"
weight = 89
+++
## Introduction
Some text."""
    meta, body = parse_frontmatter(content)
    assert meta.get("title") == "Kubeflow 26.03"
    assert meta.get("weight") == 89
    assert "Introduction" in body

def test_hf_token_survives():
    content = "+++ \n+++\nHere is the token: <YOUR_HF_TOKEN>"
    meta, body = clean_hugo_markdown(content)
    assert "<YOUR_HF_TOKEN>" in body, "The token should survive HTML cleaning"

def test_rowspan_table():
    html = """<table>
      <tr><td rowspan="2">Group A</td><td>Comp 1</td><td>v1</td></tr>
      <tr><td>Comp 2</td><td>v2</td></tr>
    </table>"""
    processed = process_html_table(html)
    assert "| Group A | Comp 1 | v1 |" in processed
    assert "| Group A | Comp 2 | v2 |" in processed

def test_shortcodes():
    content = "+++ \n+++\n{{% alert title=\"Note\" color=\"warning\" %}}This is a warning{{% /alert %}}"
    meta, body = clean_hugo_markdown(content)
    assert "NOTE: This is a warning" in body
