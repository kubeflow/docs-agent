import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.reranking import RerankConfig, candidate_pool_limit, load_rerank_config_from_env, rerank_documents


def test_candidate_pool_limit_expands_and_caps():
    config = RerankConfig(enabled=True, candidate_multiplier=3, max_candidates=10)
    assert candidate_pool_limit(2, config) == 6
    assert candidate_pool_limit(4, config) == 10


def test_candidate_pool_limit_handles_invalid_top_k(caplog):
    config = RerankConfig(enabled=True, candidate_multiplier=3, max_candidates=10)

    with caplog.at_level(logging.WARNING):
        limit = candidate_pool_limit("invalid", config)

    assert limit == 3
    assert "Invalid top_k=invalid" in caplog.text


def test_rerank_documents_returns_empty_for_empty_input():
    config = RerankConfig()
    assert rerank_documents("kubeflow", [], config, top_k=5) == []


def test_rerank_documents_scoring_and_ordering():
    config = RerankConfig(
        enabled=True,
        similarity_weight=0.7,
        keyword_weight=0.2,
        metadata_weight=0.1,
        min_token_len=3,
    )

    docs = [
        {
            "similarity": 0.9,
            "file_path": "docs/kserve.md",
            "citation_url": "https://kubeflow.org/docs/components/kserve",
            "content_text": "kserve inference service gpu deployment",
        },
        {
            "similarity": 0.95,
            "file_path": "docs/pipelines.md",
            "citation_url": "https://kubeflow.org/docs/components/pipelines",
            "content_text": "pipeline runs and scheduling",
        },
    ]

    ranked = rerank_documents("kserve gpu", docs, config, top_k=2)

    assert len(ranked) == 2
    assert ranked[0]["file_path"] == "docs/kserve.md"
    assert ranked[0]["rerank_score"] >= ranked[1]["rerank_score"]


def test_rerank_documents_deterministic_tie_breaking():
    config = RerankConfig(enabled=True, similarity_weight=1.0, keyword_weight=0.0, metadata_weight=0.0)

    docs = [
        {
            "similarity": 0.5,
            "file_path": "z.md",
            "citation_url": "https://example.com/z",
            "content_text": "same",
        },
        {
            "similarity": 0.5,
            "file_path": "a.md",
            "citation_url": "https://example.com/a",
            "content_text": "same",
        },
    ]

    ranked = rerank_documents("anything", docs, config, top_k=2)

    assert [doc["file_path"] for doc in ranked] == ["a.md", "z.md"]


def test_load_rerank_config_from_env_fallbacks(monkeypatch, caplog):
    monkeypatch.setenv("RERANK_ENABLED", "not-a-bool")
    monkeypatch.setenv("RERANK_CANDIDATE_MULTIPLIER", "nan")
    monkeypatch.setenv("RERANK_SIMILARITY_WEIGHT", "bad")
    monkeypatch.setenv("RERANK_KEYWORD_WEIGHT", "bad")
    monkeypatch.setenv("RERANK_METADATA_WEIGHT", "bad")
    monkeypatch.setenv("RERANK_MAX_CANDIDATES", "bad")
    monkeypatch.setenv("RERANK_MIN_TOKEN_LEN", "bad")
    monkeypatch.setenv("RERANK_DEBUG_LOG", "bad")
    monkeypatch.setenv("RERANK_LOG_TOP_N", "bad")

    with caplog.at_level(logging.WARNING):
        config = load_rerank_config_from_env()

    assert config == RerankConfig()
    assert "Invalid boolean value" in caplog.text
    assert "Invalid integer" in caplog.text
    assert "Invalid float" in caplog.text


def test_rerank_disabled_preserves_similarity_order():
    config = RerankConfig(enabled=False)

    docs = [
        {"similarity": 0.8, "file_path": "one.md", "citation_url": "", "content_text": ""},
        {"similarity": 0.7, "file_path": "two.md", "citation_url": "", "content_text": ""},
    ]

    ranked = rerank_documents("kubeflow", docs, config, top_k=2)

    assert [doc["file_path"] for doc in ranked] == ["one.md", "two.md"]
    assert ranked[0]["keyword_score"] == 0.0
    assert ranked[0]["metadata_score"] == 0.0
