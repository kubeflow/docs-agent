"""Tests for Hugo shortcode stripping in the canonical (v4) ingest path."""

import sys
from pathlib import Path

import pytest

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR / "utils"))

from canonical_rag_ingest import _strip_hugo_artifacts, parse_canonical_document  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Angle-bracket shortcode followed later by a percent shortcode: the text
        # between them must survive.
        (
            'A {{< figure src="/a.png" >}} keep this {{% kf-latest-version %}} B',
            "A  keep this  B",
        ),
        ("A {{< tab >}} keep {{< /tab >}} B", "A  keep  B"),
        ("A {{% param %}} keep {{% /param %}} B", "A  keep  B"),
        ('A {{< ref "x" >}}\n\nkeep\n\n{{% x %}}', "A \n\nkeep\n\n"),
    ],
)
def test_strip_hugo_artifacts_only_removes_the_shortcodes(text, expected):
    assert _strip_hugo_artifacts(text) == expected


@pytest.mark.unit
def test_section_between_angle_and_percent_shortcodes_is_indexed():
    doc = """---
title: Install
---
Intro text.

{{< figure src="/a.png" >}}

## Prerequisites

You need Kubernetes 1.29 and kubectl.

## Steps

Run the installer for {{% kf-latest-version %}} now.
"""
    parsed = parse_canonical_document(doc)
    paths = [section.section_path for section in parsed.sections]
    text = " ".join(block.content for section in parsed.sections for block in section.blocks)

    assert "Install > Prerequisites" in paths
    assert "You need Kubernetes 1.29 and kubectl." in text
    assert "{{" not in text
