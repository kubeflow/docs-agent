"""Version matching in mcp-server/intent_router.py."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs-agent-mcp" / "mcp-server"))

import intent_router  # noqa: E402


@pytest.mark.parametrize(
    "entity",
    [
        {"version": "1.1"},
        {"file_path": "releases/v1.1.0.md"},
        {"file_path": "releases/kubeflow-v1.1.md"},
        {"content_text": "Kubeflow 1.1 notes"},
    ],
)
def test_version_matches_on_number_boundaries(entity):
    assert intent_router.entity_matches_version(entity, ["1.1"])


@pytest.mark.parametrize(
    "entity",
    [
        {"version": "1.10"},
        {"file_path": "releases/trainer/v2.1.1.md"},
        {"content_text": "requires CUDA 11.1"},
    ],
)
def test_version_does_not_match_longer_numbers(entity):
    assert not intent_router.entity_matches_version(entity, ["1.1"])


def test_api_version_does_not_match_longer_suffix():
    assert intent_router.entity_matches_version({"content_text": "apiVersion: v1beta1"}, ["v1beta1"])
    assert not intent_router.entity_matches_version({"content_text": "apiVersion: v1beta10"}, ["v1beta1"])


def test_release_date_rerank_keeps_exact_version_in_top_k():
    query = "When was Kubeflow 1.1 released?"
    plan = intent_router.pick_search_plan(query)
    hits = [
        {"distance": 0.95, "entity": {"version": "1.10", "file_path": "releases/kubeflow-1.10.md"}},
        {"distance": 0.90, "entity": {"content_text": "requires CUDA 11.1", "file_path": "docs/gpu.md"}},
        {"distance": 0.80, "entity": {"version": "1.1", "file_path": "releases/kubeflow-1.1.md"}},
    ]

    top = intent_router.rerank_hits_after_search(plan, hits, query, top_k=2)

    assert top[0]["entity"]["file_path"] == "releases/kubeflow-1.1.md"
