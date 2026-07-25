"""Tests for GitHub issues pipeline utilities (docs-agent-mcp/pipelines/issues_utils.py).

Tests the pure-Python metadata parsing and chunking logic extracted
from the chunk_and_embed_issues KFP component for testability.
"""

import sys
from pathlib import Path

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

from issues_utils import (
    MAX_TEI_INPUT_CHARS,
    parse_issue_metadata,
    build_metadata_prefix,
    split_issue_into_chunks,
)


# --- Sample content fixtures ---

SAMPLE_ISSUE_CONTENT = """# KServe model not loading

**Repository:** kubeflow/kubeflow
**Issue:** #42
**URL:** https://github.com/kubeflow/kubeflow/issues/42
**Labels:** kind/bug, area/kserve
**State:** open
**Created:** 2026-01-15
**Updated:** 2026-01-20

The model fails to load when using GPU. I've tried multiple configurations
but keep getting OOM errors.

---
**Comment by @alice** (2026-01-16):
Have you tried setting memory limits in your InferenceService spec?

---
**Comment by @bob** (2026-01-17):
Fixed by upgrading KServe to v0.12. The GPU memory allocation was improved."""

SHORT_ISSUE_CONTENT = """# Typo in docs

**Repository:** kubeflow/website
**Issue:** #99
**URL:** https://github.com/kubeflow/website/issues/99
**Labels:** kind/docs
**State:** closed
**Created:** 2026-02-01
**Updated:** 2026-02-02

There is a typo on the installation page."""

NO_LABELS_CONTENT = """# Feature request

**Repository:** kubeflow/pipelines
**Issue:** #500
**URL:** https://github.com/kubeflow/pipelines/issues/500
**Labels:**
**State:** open
**Created:** 2026-03-01
**Updated:** 2026-03-10

Please add support for caching."""


class TestParseIssueMetadata:
    """Tests for parse_issue_metadata."""

    def test_extracts_title(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        assert meta["title"] == "KServe model not loading"

    def test_extracts_repo_name(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        assert meta["repo_name"] == "kubeflow/kubeflow"

    def test_extracts_issue_number(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        assert meta["issue_number"] == 42

    def test_extracts_state(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        assert meta["issue_state"] == "open"

    def test_extracts_labels(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        assert meta["issue_labels"] == "kind/bug, area/kserve"

    def test_extracts_citation_url(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        assert meta["citation_url"] == "https://github.com/kubeflow/kubeflow/issues/42"

    def test_handles_empty_labels(self):
        meta = parse_issue_metadata(NO_LABELS_CONTENT)
        assert meta["issue_labels"] == ""

    def test_handles_closed_state(self):
        meta = parse_issue_metadata(SHORT_ISSUE_CONTENT)
        assert meta["issue_state"] == "closed"

    def test_returns_zero_for_missing_number(self):
        meta = parse_issue_metadata("No metadata here")
        assert meta["issue_number"] == 0

    def test_returns_empty_for_missing_fields(self):
        meta = parse_issue_metadata("Just plain text")
        assert meta["title"] == ""
        assert meta["repo_name"] == ""
        assert meta["citation_url"] == ""

    def test_warns_when_all_fields_empty(self, capsys):
        parse_issue_metadata("Just plain text")
        assert "WARNING: Failed to parse GitHub issue metadata" in capsys.readouterr().err

    def test_no_warning_when_fields_present(self, capsys):
        parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        assert capsys.readouterr().err == ""


class TestBuildMetadataPrefix:
    """Tests for build_metadata_prefix."""

    def test_includes_issue_number_and_title(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        assert "[Issue #42]" in prefix
        assert "KServe model not loading" in prefix

    def test_includes_repo(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        assert "Repo: kubeflow/kubeflow" in prefix

    def test_includes_state(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        assert "State: open" in prefix

    def test_includes_labels(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        assert "Labels: kind/bug, area/kserve" in prefix

    def test_omits_empty_labels(self):
        meta = parse_issue_metadata(NO_LABELS_CONTENT)
        prefix = build_metadata_prefix(meta)
        assert "Labels:" not in prefix

    def test_ends_with_double_newline(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        assert prefix.endswith("\n\n")


class TestSplitIssueIntoChunks:
    """Tests for split_issue_into_chunks."""

    def test_short_issue_single_chunk(self):
        meta = parse_issue_metadata(SHORT_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        chunks = split_issue_into_chunks(SHORT_ISSUE_CONTENT, prefix, chunk_size=2000)
        assert len(chunks) == 1

    def test_short_issue_has_prefix(self):
        meta = parse_issue_metadata(SHORT_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        chunks = split_issue_into_chunks(SHORT_ISSUE_CONTENT, prefix, chunk_size=2000)
        assert chunks[0].startswith("[Issue #99]")

    def test_long_issue_splits_at_comment_boundaries(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        # Use a small chunk_size to force splitting
        chunks = split_issue_into_chunks(SAMPLE_ISSUE_CONTENT, prefix, chunk_size=300)
        assert len(chunks) > 1

    def test_every_chunk_has_prefix(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        chunks = split_issue_into_chunks(SAMPLE_ISSUE_CONTENT, prefix, chunk_size=300)
        for chunk in chunks:
            assert chunk.startswith("[Issue #42]")

    def test_no_empty_chunks(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        chunks = split_issue_into_chunks(SAMPLE_ISSUE_CONTENT, prefix, chunk_size=300)
        for chunk in chunks:
            # Each chunk should have content beyond just the prefix
            assert len(chunk) > len(prefix)

    def test_issue_with_no_comments(self):
        meta = parse_issue_metadata(SHORT_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        chunks = split_issue_into_chunks(SHORT_ISSUE_CONTENT, prefix, chunk_size=2000)
        assert len(chunks) == 1
        assert "typo" in chunks[0].lower()

    def test_respects_chunk_size(self):
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        chunk_size = 400
        chunks = split_issue_into_chunks(SAMPLE_ISSUE_CONTENT, prefix, chunk_size=chunk_size)
        for chunk in chunks:
            # Allow some tolerance for the text splitter
            assert len(chunk) <= chunk_size + 100

    def test_handles_empty_content(self):
        chunks = split_issue_into_chunks("", "prefix\n\n", chunk_size=1000)
        assert len(chunks) == 1

    def test_clamps_to_max_tei_input_chars_even_with_larger_chunk_size(self):
        """Regression test: a chunk_size above MAX_TEI_INPUT_CHARS must not produce
        chunks longer than what TEI actually embeds, or the tail of the chunk is
        silently invisible to semantic search (see issues-pipeline.py chunk_and_embed_issues).
        """
        meta = parse_issue_metadata(SAMPLE_ISSUE_CONTENT)
        prefix = build_metadata_prefix(meta)
        long_body = SAMPLE_ISSUE_CONTENT + ("\n\nMore detail. " * 200)
        chunks = split_issue_into_chunks(long_body, prefix, chunk_size=1500)
        for chunk in chunks:
            assert len(chunk) <= MAX_TEI_INPUT_CHARS + 100
