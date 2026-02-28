"""Shared pytest fixtures and configuration for docs-agent tests.

Heavy third-party libraries (pymilvus, sentence_transformers) are mocked at
import time so that the test suite can run in any CI environment without GPU
or a live Milvus instance.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock heavy dependencies BEFORE importing the module under test
# ---------------------------------------------------------------------------

def _install_mock_modules():
    """Inject lightweight mock modules for pymilvus and sentence_transformers.

    This must run before ``shared.rag_core`` is imported for the first time
    so that the module-level ``from pymilvus import ...`` succeeds.
    """
    # -- pymilvus -----------------------------------------------------------
    pymilvus = types.ModuleType("pymilvus")
    pymilvus.connections = MagicMock()

    # Collection must be a callable that returns a MagicMock with .load()
    # (not MagicMock itself, which would create spec-based mocks from args)
    def _fake_collection(*args, **kwargs):
        coll = MagicMock()
        coll.load = MagicMock()
        coll.search = MagicMock(return_value=[[]])
        return coll

    pymilvus.Collection = _fake_collection
    pymilvus.utility = MagicMock()
    sys.modules.setdefault("pymilvus", pymilvus)

    # -- sentence_transformers ----------------------------------------------
    st_mod = types.ModuleType("sentence_transformers")

    class _FakeVector(list):
        """A list subclass that supports .tolist() like a numpy array."""

        def tolist(self):
            return list(self)

    class _FakeEncoder:
        """Minimal stand-in for SentenceTransformer."""

        def __init__(self, model_name: str = "mock"):
            self.model_name = model_name

        def encode(self, text):  # noqa: ANN001, ANN201
            """Return a deterministic 768-dim vector (all zeros)."""
            return _FakeVector([0.0] * 768)

    st_mod.SentenceTransformer = _FakeEncoder  # type: ignore[attr-defined]
    sys.modules.setdefault("sentence_transformers", st_mod)


# Install mocks before any test imports shared.rag_core
_install_mock_modules()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Ensure each test starts with a clean singleton state."""
    from shared import rag_core  # noqa: E402

    # Reset the encoder singleton
    rag_core._encoder = None

    # Reset the MilvusConnectionManager singleton
    rag_core.MilvusConnectionManager._instance = None

    yield

    # Cleanup after test
    rag_core._encoder = None
    rag_core.MilvusConnectionManager._instance = None
