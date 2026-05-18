"""Tests for pipeline utility functions and component logic."""

import sys
from pathlib import Path

# Add pipelines directory to path
PIPELINES_DIR = Path(__file__).parent.parent / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

from utils import clean_content


class TestCleanContent:
    """Tests for the clean_content utility function."""

    def test_removes_yaml_frontmatter(self):
        """Should remove --- delimited YAML frontmatter."""
        content = "---\ntitle: Test\ndate: 2026-01-01\n---\n\nActual content here."
        result = clean_content(content)
        assert "title:" not in result
        assert "Actual content here." in result

    def test_removes_toml_frontmatter(self):
        """Should remove +++ delimited TOML frontmatter."""
        content = "+++\ntitle = 'Test'\n+++\n\nActual content here."
        result = clean_content(content)
        assert "title =" not in result
        assert "Actual content here." in result

    def test_removes_hugo_template_syntax(self):
        """Should remove {{ ... }} Hugo template expressions."""
        content = 'Install with {{ .Get "name" }} command. Then run it.'
        result = clean_content(content)
        assert "{{" not in result
        assert "Install with" in result
        assert "command" in result

    def test_removes_html_comments(self):
        """Should remove <!-- ... --> HTML comments."""
        content = "Before <!-- this is a comment --> After"
        result = clean_content(content)
        assert "this is a comment" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_html_tags(self):
        """Should remove HTML tags but keep their text content."""
        content = "<div class='main'><p>Hello <strong>world</strong></p></div>"
        result = clean_content(content)
        assert "<div" not in result
        assert "<p>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_navigation_artifacts(self):
        """Should remove navigation/menu text artifacts."""
        content = "Get Started with Kubeflow. Home Menu Navigation overview."
        result = clean_content(content)
        assert "Get Started" not in result
        assert "Menu" not in result
        assert "Navigation" not in result
        assert "Kubeflow" in result

    def test_removes_urls(self):
        """Should remove HTTP/HTTPS URLs."""
        content = "Visit https://kubeflow.org/docs for more info."
        result = clean_content(content)
        assert "https://kubeflow.org" not in result
        assert "Visit" in result
        assert "for more info." in result

    def test_converts_markdown_links_to_text(self):
        """Should convert [text](url) markdown links to just text."""
        content = "See [the documentation](https://kubeflow.org/docs) for details."
        result = clean_content(content)
        assert "the documentation" in result
        assert "(https://kubeflow.org/docs)" not in result

    def test_normalizes_whitespace(self):
        """Should collapse multiple spaces into single space."""
        content = "Hello    world     test"
        result = clean_content(content)
        assert "  " not in result
        assert "Hello world test" in result

    def test_strips_leading_trailing_whitespace(self):
        """Should strip leading and trailing whitespace."""
        content = "   Hello world   "
        result = clean_content(content)
        assert result == "Hello world"

    def test_returns_empty_for_frontmatter_only(self):
        """Content that is only frontmatter should return empty string."""
        content = "---\ntitle: Nothing\ndate: 2026-01-01\n---"
        result = clean_content(content)
        assert result == ""

    def test_preserves_meaningful_content(self):
        """Should not remove meaningful documentation content."""
        content = (
            "Kubeflow Pipelines is a platform for building and deploying "
            "portable, scalable ML workflows based on Docker containers."
        )
        result = clean_content(content)
        assert "Kubeflow Pipelines" in result
        assert "ML workflows" in result
        assert "Docker containers" in result

    def test_handles_multiline_html_comment(self):
        """Should remove HTML comments that span multiple lines."""
        content = "Before\n<!--\nThis is a\nmulti-line comment\n-->\nAfter"
        result = clean_content(content)
        assert "multi-line comment" not in result
        assert "Before" in result
        assert "After" in result

    def test_handles_multiline_hugo_template(self):
        """Should remove Hugo templates that span multiple lines."""
        content = "Before {{ if .IsHome }}\nstuff\n{{ end }} After"
        result = clean_content(content)
        assert "{{ if" not in result
        assert "{{ end }}" not in result

    def test_real_world_kubeflow_doc_snippet(self):
        """Test with a realistic Kubeflow documentation snippet."""
        content = """---
title: KServe Overview
description: Overview of KServe
weight: 1
---

<!-- This is auto-generated -->

<div class="section-index">

## What is KServe?

[KServe](https://kserve.github.io/website/) enables serverless inference
on Kubernetes. Visit https://kubeflow.org/docs/external-add-ons/kserve/
for more details.

{{ partial "section-index.html" . }}

</div>
"""
        result = clean_content(content)
        assert "title:" not in result
        assert "auto-generated" not in result
        assert "section-index" not in result
        assert "KServe" in result
        assert "serverless inference" in result
        assert "Kubernetes" in result


class TestCleanContentEdgeCases:
    """Edge cases for content cleaning."""

    def test_empty_string(self):
        """Should handle empty string input."""
        assert clean_content("") == ""

    def test_whitespace_only(self):
        """Should handle whitespace-only input."""
        assert clean_content("   \n\n  \t  ") == ""

    def test_no_cleaning_needed(self):
        """Plain text should pass through with minimal changes."""
        content = "Simple plain text with no special formatting."
        result = clean_content(content)
        assert result == content

    def test_nested_html_tags(self):
        """Should handle deeply nested HTML tags."""
        content = "<div><div><div><span>Deep content</span></div></div></div>"
        result = clean_content(content)
        assert "Deep content" in result
        assert "<" not in result
