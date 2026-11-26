"""Tests for concurrent ingestion lease coordination and fencing."""

import sys
import time
from pathlib import Path
import pytest

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))
sys.path.insert(0, str(PIPELINES_DIR / "utils"))

from lease_coordination import (
    InMemoryLeaseProvider,
    IngestionConflictError,
    StaleWorkerError,
)


def test_lease_acquire_and_release():
    provider = InMemoryLeaseProvider()
    token = provider.acquire("docs_v4", ttl_seconds=60)
    assert token
    provider.validate("docs_v4", token)
    provider.release("docs_v4", token)

    # After release, can acquire again immediately
    new_token = provider.acquire("docs_v4", ttl_seconds=60)
    assert new_token != token


def test_lease_mutual_exclusion():
    provider = InMemoryLeaseProvider()
    token1 = provider.acquire("docs_v4", ttl_seconds=60)
    assert token1

    # Second worker should fail with IngestionConflictError
    with pytest.raises(IngestionConflictError) as exc_info:
        provider.acquire("docs_v4", ttl_seconds=60)
    assert "currently held by active worker" in str(exc_info.value)

    # Separate collection can be acquired without conflict
    token_code = provider.acquire("code_v4", ttl_seconds=60)
    assert token_code


def test_lease_expiration_and_recovery():
    provider = InMemoryLeaseProvider()
    token = provider.acquire("docs_v4", ttl_seconds=1)
    time.sleep(1.1)

    # Validate fails on expired token
    with pytest.raises(StaleWorkerError):
        provider.validate("docs_v4", token)

    # New worker can now acquire because old lease expired
    new_token = provider.acquire("docs_v4", ttl_seconds=60)
    assert new_token


def test_lease_renew():
    provider = InMemoryLeaseProvider()
    token = provider.acquire("docs_v4", ttl_seconds=1)
    provider.renew("docs_v4", token, ttl_seconds=5)
    time.sleep(1.1)

    # Still valid because of renewal
    provider.validate("docs_v4", token)
    provider.release("docs_v4", token)


def test_stale_worker_fenced_after_release():
    provider = InMemoryLeaseProvider()
    token = provider.acquire("docs_v4", ttl_seconds=60)
    provider.release("docs_v4", token)

    with pytest.raises(StaleWorkerError):
        provider.validate("docs_v4", token)
