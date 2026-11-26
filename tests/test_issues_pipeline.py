"""Tests for GitHub issues pipeline utilities (docs-agent-mcp/pipelines/issues_utils.py).

Tests the pure-Python metadata parsing and chunking logic extracted
from the chunk_and_embed_issues KFP component for testability.
"""

import sys
from pathlib import Path

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))
# Shared helpers moved to pipelines/utils/ (the pipelines add it to sys.path the same way).
sys.path.insert(0, str(PIPELINES_DIR / "utils"))

from issues_utils import (
    build_issue_record,
    build_metadata_prefix,
    fetch_issue_comments,
    format_issue_markdown,
    has_structured_metadata,
    issue_content_segments,
    parse_issue_metadata,
    resolve_issue_metadata,
    split_issue_into_chunks,
    split_issue_record_into_chunks,
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

SAMPLE_STRUCTURED_RECORD = build_issue_record(
    repo_name="kubeflow/kubeflow",
    repo_short_name="kubeflow",
    issue_number=42,
    title="KServe model not loading",
    url="https://github.com/kubeflow/kubeflow/issues/42",
    labels="kind/bug, area/kserve",
    state="open",
    created_at="2026-01-15",
    updated_at="2026-01-20",
    body=("The model fails to load when using GPU. I've tried multiple configurations\nbut keep getting OOM errors."),
    comments=[
        {
            "author": "alice",
            "created_at": "2026-01-16",
            "body": "Have you tried setting memory limits in your InferenceService spec?",
            "url": "https://github.com/kubeflow/kubeflow/issues/42#issuecomment-1001",
        },
        {
            "author": "bob",
            "created_at": "2026-01-17",
            "body": "Fixed by upgrading KServe to v0.12. The GPU memory allocation was improved.",
            "url": "https://github.com/kubeflow/kubeflow/issues/42#issuecomment-1002",
        },
    ],
)


class TestParseIssueMetadata:
    """Tests for parse_issue_metadata (legacy markdown regex path)."""

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


class TestStructuredIssueRecord:
    """Tests for machine-readable download_github_issues JSONL records."""

    def test_build_issue_record_includes_markdown_and_structured_fields(self):
        record = SAMPLE_STRUCTURED_RECORD
        assert record["content"].startswith("# KServe model not loading")
        assert "**Repository:** kubeflow/kubeflow" in record["content"]
        assert record["title"] == "KServe model not loading"
        assert record["repo_name"] == "kubeflow/kubeflow"
        assert record["issue_number"] == 42
        assert record["issue_state"] == "open"
        assert record["issue_labels"] == "kind/bug, area/kserve"
        assert record["url"] == "https://github.com/kubeflow/kubeflow/issues/42"
        assert record["body"].startswith("The model fails")
        assert len(record["comments"]) == 2
        assert record["comments"][0]["author"] == "alice"

    def test_format_issue_markdown_matches_legacy_layout(self):
        md = format_issue_markdown(
            title="Typo in docs",
            repo_name="kubeflow/website",
            issue_number=99,
            url="https://github.com/kubeflow/website/issues/99",
            labels="kind/docs",
            state="closed",
            created_at="2026-02-01",
            updated_at="2026-02-02",
            body="There is a typo on the installation page.",
            comments=[],
        )
        assert md == SHORT_ISSUE_CONTENT

    def test_has_structured_metadata(self):
        assert has_structured_metadata(SAMPLE_STRUCTURED_RECORD) is True
        assert has_structured_metadata({"content": SAMPLE_ISSUE_CONTENT}) is False

    def test_resolve_prefers_structured_fields_over_markdown(self):
        # Intentionally corrupt content; structured fields must win
        record = dict(SAMPLE_STRUCTURED_RECORD)
        record["content"] = "garbage that would fail regex parsing"
        meta = resolve_issue_metadata(record)
        assert meta["title"] == "KServe model not loading"
        assert meta["issue_number"] == 42
        assert meta["citation_url"] == "https://github.com/kubeflow/kubeflow/issues/42"

    def test_resolve_falls_back_to_markdown_regex(self):
        legacy = {"content": SAMPLE_ISSUE_CONTENT, "url": "ignored-when-parsed-from-md"}
        meta = resolve_issue_metadata(legacy)
        assert meta["title"] == "KServe model not loading"
        assert meta["issue_number"] == 42
        assert meta["citation_url"] == "https://github.com/kubeflow/kubeflow/issues/42"

    def test_resolve_warns_when_legacy_parse_empty(self, capsys):
        meta = resolve_issue_metadata({"content": "no metadata here"})
        assert meta["issue_number"] == 0
        captured = capsys.readouterr()
        assert "WARNING: Failed to parse GitHub issue metadata" in captured.out

    def test_issue_content_segments_from_structured_comments(self):
        segments = issue_content_segments(SAMPLE_STRUCTURED_RECORD)
        assert len(segments) == 3  # header+body, comment1, comment2
        assert "KServe model not loading" in segments[0]
        assert "Comment by @alice" in segments[1]
        assert "Comment by @bob" in segments[2]

    def test_split_issue_record_uses_structured_segments(self):
        meta = resolve_issue_metadata(SAMPLE_STRUCTURED_RECORD)
        prefix = build_metadata_prefix(meta)
        chunks = split_issue_record_into_chunks(SAMPLE_STRUCTURED_RECORD, prefix, chunk_size=300)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.startswith("[Issue #42]")


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


def _api_comment(login, created_at, body, html_url):
    """Build one fake GitHub API comment dict."""
    return {
        "user": {"login": login} if login is not None else None,
        "created_at": created_at,
        "body": body,
        "html_url": html_url,
    }


class TestFetchIssueComments:
    """Tests for fetch_issue_comments (mocked request_fn, no network)."""

    def test_maps_author_date_body_url(self):
        def request_fn(url, params):
            assert url == ("https://api.github.com/repos/kubeflow/docs-agent/issues/9/comments")
            assert params == {"per_page": 100, "page": 1}
            return [
                _api_comment(
                    "alice",
                    "2026-01-16T10:00:00Z",
                    "Have you tried setting memory limits?",
                    "https://github.com/kubeflow/docs-agent/issues/9#issuecomment-1001",
                )
            ]

        comments = fetch_issue_comments("kubeflow", "docs-agent", 9, request_fn)
        assert comments == [
            {
                "author": "alice",
                "created_at": "2026-01-16",
                "body": "Have you tried setting memory limits?",
                "url": "https://github.com/kubeflow/docs-agent/issues/9#issuecomment-1001",
            }
        ]

    def test_pagination_two_pages(self):
        def request_fn(url, params):
            if params["page"] == 1:
                return [_api_comment("a", "2026-01-01T00:00:00Z", f"c{i}", "") for i in range(100)]
            return [_api_comment("b", "2026-01-02T00:00:00Z", "last", "")]

        comments = fetch_issue_comments("kubeflow", "docs-agent", 9, request_fn)
        assert len(comments) == 101
        assert comments[0]["author"] == "a"
        assert comments[-1]["author"] == "b"

    def test_empty_comments_returns_empty(self):
        comments = fetch_issue_comments("kubeflow", "docs-agent", 9, lambda url, params: [])
        assert comments == []

    def test_malformed_comment_missing_user_body(self):
        def request_fn(url, params):
            return [
                {"created_at": "2026-01-16T10:00:00Z", "body": None},
                "not-a-dict",
            ]

        comments = fetch_issue_comments("kubeflow", "docs-agent", 9, request_fn)
        assert comments == [
            {
                "author": "unknown",
                "created_at": "2026-01-16",
                "body": "",
                "url": "",
            }
        ]

    def test_failed_fetch_returns_empty_and_warns(self, capsys):
        comments = fetch_issue_comments("kubeflow", "docs-agent", 9, lambda url, params: None)
        assert comments == []
        captured = capsys.readouterr()
        assert "WARNING: Failed to fetch comments" in captured.out
        assert "kubeflow/docs-agent#9" in captured.out

    def test_failed_second_page_keeps_first_page_and_warns(self, capsys):
        def request_fn(url, params):
            if params["page"] == 1:
                return [_api_comment("a", "2026-01-01T00:00:00Z", "c0", "") for _ in range(100)]
            return None

        comments = fetch_issue_comments("kubeflow", "docs-agent", 9, request_fn)
        assert len(comments) == 100
        captured = capsys.readouterr()
        assert "WARNING: Failed to fetch comments" in captured.out

    def test_comment_segment_includes_url_when_present(self):
        segments = issue_content_segments(SAMPLE_STRUCTURED_RECORD)
        assert "https://github.com/kubeflow/kubeflow/issues/42#issuecomment-1001" in segments[1]
        assert "https://github.com/kubeflow/kubeflow/issues/42#issuecomment-1002" in segments[2]

    def test_comment_without_url_still_renders_old_format(self):
        record = build_issue_record(
            repo_name="kubeflow/kubeflow",
            repo_short_name="kubeflow",
            issue_number=42,
            title="KServe model not loading",
            url="https://github.com/kubeflow/kubeflow/issues/42",
            labels="",
            state="open",
            created_at="2026-01-15",
            updated_at="2026-01-20",
            body="Body.",
            comments=[{"author": "alice", "created_at": "2026-01-16", "body": "Hi."}],
        )
        segments = issue_content_segments(record)
        assert segments[1] == "**Comment by @alice** (2026-01-16):\nHi."
        assert "http" not in segments[1]

    def test_format_issue_markdown_includes_comment_url(self):
        md = format_issue_markdown(
            title="T",
            repo_name="kubeflow/docs-agent",
            issue_number=9,
            url="https://github.com/kubeflow/docs-agent/issues/9",
            labels="",
            state="open",
            created_at="2026-01-30",
            updated_at="2026-03-22",
            body="Body.",
            comments=[
                {
                    "author": "alice",
                    "created_at": "2026-01-16",
                    "body": "Hi.",
                    "url": "https://github.com/kubeflow/docs-agent/issues/9#issuecomment-1001",
                }
            ],
        )
        assert "https://github.com/kubeflow/docs-agent/issues/9#issuecomment-1001" in md
