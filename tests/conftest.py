"""Shared fixtures for docs-agent tests."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_milvus_client():
    """Create a mock MilvusClient that returns configurable search results."""
    client = MagicMock()
    client.search.return_value = [[]]  # default: no results
    return client


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer that returns a fixed embedding."""
    model = MagicMock()
    # Return a fake 768-dim embedding
    model.encode.return_value = np.zeros(768, dtype=np.float32)
    return model


@pytest.fixture
def sample_milvus_hits():
    """Sample Milvus search results for testing output formatting."""
    return [[
        {
            "id": 1,
            "distance": 0.9234,
            "entity": {
                "content_text": "KServe provides serverless inference on Kubernetes.",
                "citation_url": "https://www.kubeflow.org/docs/kserve/",
                "file_path": "content/en/docs/kserve/overview.md",
            },
        },
        {
            "id": 2,
            "distance": 0.8567,
            "entity": {
                "content_text": "Install Kubeflow Pipelines using the standalone deployment.",
                "citation_url": "https://www.kubeflow.org/docs/pipelines/install/",
                "file_path": "content/en/docs/pipelines/install.md",
            },
        },
    ]]


@pytest.fixture
def sample_github_issue_api_response():
    """Sample GitHub API response for a single issue."""
    return {
        "number": 42,
        "title": "KServe model not loading",
        "body": "The model fails to load when using GPU.",
        "state": "open",
        "html_url": "https://github.com/kubeflow/kubeflow/issues/42",
        "created_at": "2026-01-15T10:00:00Z",
        "updated_at": "2026-01-20T14:30:00Z",
        "labels": [{"name": "kind/bug"}, {"name": "area/kserve"}],
        "comments": 2,
        "user": {"login": "testuser"},
    }


@pytest.fixture
def sample_github_pr_api_response():
    """Sample GitHub API response that represents a PR (has pull_request key)."""
    return {
        "number": 100,
        "title": "Fix typo in docs",
        "body": "Fixed a typo.",
        "state": "closed",
        "html_url": "https://github.com/kubeflow/kubeflow/pull/100",
        "created_at": "2026-01-10T10:00:00Z",
        "updated_at": "2026-01-11T10:00:00Z",
        "labels": [],
        "comments": 0,
        "user": {"login": "contributor"},
        "pull_request": {
            "url": "https://api.github.com/repos/kubeflow/kubeflow/pulls/100"
        },
    }


@pytest.fixture
def sample_issues_milvus_hits():
    """Sample Milvus search results from the issues_rag collection."""
    return [[
        {
            "id": 1,
            "distance": 0.8912,
            "entity": {
                "content_text": "[Issue #42] KServe model not loading | Repo: kubeflow/kubeflow\n\nThe model fails to load when using GPU.",
                "citation_url": "https://github.com/kubeflow/kubeflow/issues/42",
                "repo_name": "kubeflow/kubeflow",
                "issue_number": 42,
                "issue_state": "open",
                "issue_labels": "kind/bug, area/kserve",
            },
        },
        {
            "id": 2,
            "distance": 0.7845,
            "entity": {
                "content_text": "[Issue #100] Pipeline timeout on large datasets | Repo: kubeflow/pipelines\n\nPipeline runs time out after 30 minutes.",
                "citation_url": "https://github.com/kubeflow/pipelines/issues/100",
                "repo_name": "kubeflow/pipelines",
                "issue_number": 100,
                "issue_state": "closed",
                "issue_labels": "kind/bug",
            },
        },
    ]]
