"""Tests for reranker_client and compressor."""

from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"))

import reranker_client
import compressor


def test_reranker_client_reranks_hits():
    hits = [
        {"entity": {"content_text": "doc1"}},
        {"entity": {"content_text": "doc2"}},
    ]
    with patch("reranker_client.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"index": 1, "score": 0.9},
            {"index": 0, "score": 0.1},
        ]
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {"RERANKER_URL": "http://mock"}):
            reranked = reranker_client.rerank_hits("query", hits)

        assert len(reranked) == 2
        assert reranked[0]["entity"]["content_text"] == "doc2"
        assert reranked[1]["entity"]["content_text"] == "doc1"
        assert reranked[0]["distance"] == 0.9
        assert reranked[1]["distance"] == 0.1
        assert reranked[0]["reranked"] is True


def test_reranker_client_bypasses_if_no_url():
    hits = [{"entity": {"content_text": "doc1"}}]
    with patch.dict("os.environ", {"RERANKER_URL": ""}):
        reranked = reranker_client.rerank_hits("query", hits)
        assert reranked == hits


def test_compressor_compresses_hits_lexically():
    hits = [
        {
            "entity": {
                "content_text": (
                    "KServe is a standard Model Inference Platform on Kubernetes. "
                    "Unrelated boilerplate sentence that should be discarded when budget is tight. "
                    "An InferenceService defines Predictor, Transformer, and Explainer components."
                )
            }
        },
    ]
    # Query mentions InferenceService components
    query = "What is an InferenceService and its Predictor?"
    max_chars = 180  # Enough for intro + target sentence, but not boilerplate

    compressed = compressor.compress_hits(query, hits, max_chars)

    assert len(compressed) == 1
    content = compressed[0]["entity"]["content_text"]
    assert "KServe is a standard Model Inference Platform" in content  # Lead anchor
    assert "InferenceService defines Predictor" in content
    assert "Unrelated boilerplate sentence" not in content


def test_compressor_preserves_atomic_code_blocks():
    yaml_code = "```yaml\napiVersion: serving.kserve.io/v1beta1\nkind: InferenceService\nspec:\n  predictor:\n    model:\n      modelFormat:\n        name: sklearn\n```"
    hits = [
        {
            "entity": {
                "content_text": (
                    "Here is the manifest for serving an sklearn model with KServe.\n\n"
                    f"{yaml_code}\n\n"
                    "Some trailing notes that can be dropped if context budget is exceeded."
                )
            }
        }
    ]
    query = "InferenceService sklearn YAML manifest"
    max_chars = len(yaml_code) + 80

    compressed = compressor.compress_hits(query, hits, max_chars)

    assert len(compressed) == 1
    content = compressed[0]["entity"]["content_text"]
    # The entire code block must remain intact without broken lines or regex mutilation
    assert yaml_code in content


def test_compressor_budget_prevents_document_starvation():
    hits = [
        {
            "entity": {
                "content_text": ("Doc 1 has a very long explanation about Kubeflow Pipelines architecture. " * 10)
            }
        },
        {
            "entity": {
                "content_text": (
                    "Doc 2 explains external MySQL configuration. "
                    "Set external mysql secret credentials in the ml-pipeline config."
                )
            }
        },
    ]
    query = "How to configure external MySQL in Pipelines?"
    max_chars = 400

    compressed = compressor.compress_hits(query, hits, max_chars)

    # Both documents must be represented in compressed output (no starvation)
    assert len(compressed) == 2
    assert "Doc 1" in compressed[0]["entity"]["content_text"]
    assert "external MySQL" in compressed[1]["entity"]["content_text"]


def test_compressor_passes_through_when_under_budget():
    hits = [
        {"entity": {"content_text": "Short hit 1."}},
        {"entity": {"content_text": "Short hit 2."}},
    ]
    # Total chars is ~26, budget is 500
    compressed = compressor.compress_hits("query", hits, 500)
    assert compressed == hits


def test_compressor_bypasses_if_no_hits():
    compressed = compressor.compress_hits("query", [], 100)
    assert compressed == []


def test_compressor_bypasses_if_max_chars_zero():
    hits = [{"entity": {"content_text": "doc1"}}]
    compressed = compressor.compress_hits("query", hits, 0)
    assert compressed == hits


def test_wrapped_list_item_stays_one_block():
    text = (
        "- **parallelTrialCount**: The maximum number of HP sets that Katib\n"
        "  should train in parallel. The default value is 3.\n"
    )
    blocks = compressor._split_into_blocks(text)
    assert any("Katib should train in parallel. The default value is 3." in b for b in blocks)


def test_wrapped_sentence_is_not_cut():
    text = (
        "Katib searches the latest reported\n`accuracy` metrics for each Trial. Check the\nmetrics strategies example."
    )
    assert compressor._split_into_blocks(text) == [
        "Katib searches the latest reported `accuracy` metrics for each Trial.",
        "Check the metrics strategies example.",
    ]


def test_compressor_preserves_heading_and_list_hierarchy_without_code_blocks():
    """Headings and list items must not be space-joined into a single heading line."""
    doc = (
        "### Katib Configuration\n- Set `parallelTrialCount: 3` in the experiment spec.\n- Enable metrics collector.\n"
    )
    hits = [{"entity": {"content_text": doc}}]
    # Query mentions metrics collector; budget forces compression
    compressed = compressor.compress_hits("Katib metrics", hits, max_chars=80)

    assert len(compressed) == 1
    content = compressed[0]["entity"]["content_text"]

    # Heading must be followed by newlines and not consume list items into the header
    assert content.startswith("### Katib Configuration\n\n")
    assert "- Enable metrics collector." in content
    # The heading must not swallow the bullet point on the same line
    assert "### Katib Configuration - " not in content


def test_compressor_preserves_tables_and_prose_transitions():
    """Tables must retain newline boundaries with adjacent prose and headings."""
    doc = (
        "### Compatibility Matrix\n\n"
        "| Kubeflow | Kubernetes |\n"
        "| --- | --- |\n"
        "| v1.9 | v1.29 |\n\n"
        "Review requirements before installation."
    )
    hits = [{"entity": {"content_text": doc}}]
    compressed = compressor.compress_hits("Kubeflow v1.29 compatibility", hits, max_chars=150)

    assert len(compressed) == 1
    content = compressed[0]["entity"]["content_text"]

    assert "| Kubeflow | Kubernetes |" in content
    assert "| v1.9 | v1.29 |" in content
    # Table must not be joined inline with space
    assert "### Compatibility Matrix |" not in content
    assert "| v1.29 | Review" not in content


def test_join_blocks_helper_handles_all_markdown_boundaries():
    parts = [
        "### Section Header",
        "- First bullet point",
        "- Second bullet point",
        "Intro sentence for paragraph.",
        "Second sentence in paragraph.",
        "```yaml\nkey: value\n```",
        "Trailing prose after code.",
    ]
    joined = compressor._join_blocks(parts)

    expected = (
        "### Section Header\n\n"
        "- First bullet point\n"
        "- Second bullet point\n\n"
        "Intro sentence for paragraph. Second sentence in paragraph.\n\n"
        "```yaml\nkey: value\n```\n\n"
        "Trailing prose after code."
    )
    assert joined == expected
