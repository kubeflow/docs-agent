"""Tests for the incremental documentation ingestion pipeline.

Both `store_milvus` (full run) and `store_milvus_incremental` write into the
same `kubeflow_docs` collection, so the incremental component must agree with
the full one on the vector dimension and on the schema compatibility guard.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))


class DataType:
    INT64 = "INT64"
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"


class FieldSchema:
    def __init__(self, name, dtype, **params):
        self.name = name
        self.dtype = dtype
        self.params = params


class CollectionSchema:
    def __init__(self, fields, description=""):
        self.fields = fields
        self.description = description


def fake_pymilvus_module():
    module = ModuleType("pymilvus")
    module.CollectionSchema = CollectionSchema
    module.DataType = DataType
    module.FieldSchema = FieldSchema
    module.Collection = lambda *args, **kwargs: None
    module.connections = SimpleNamespace(connect=lambda *args, **kwargs: None)
    module.utility = SimpleNamespace(has_collection=lambda *args, **kwargs: False)
    return module


def load_incremental_pipeline_module():
    pytest.importorskip("kfp", reason="pipeline component tests require the KFP SDK")
    pipeline_path = PIPELINES_DIR / "incremental-pipeline.py"
    spec = importlib.util.spec_from_file_location("incremental_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def docs_schema(dim=768):
    """The schema the full pipeline creates, parameterized by vector dim."""
    return CollectionSchema(
        [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="last_updated", dtype=DataType.INT64),
        ],
        description="RAG collection for documentation (v=1)",
    )


def embedded_record(dim=768):
    return {
        "file_unique_id": "website:content/en/docs/components/katib/example.md",
        "repo_name": "website",
        "file_path": "content/en/docs/components/katib/example.md",
        "file_name": "example.md",
        "citation_url": "https://www.kubeflow.org/docs/components/katib/example/",
        "chunk_index": 0,
        "content_text": "Katib Experiment evidence",
        "embedding": [0.0] * dim,
    }


def test_incremental_store_creates_collection_at_requested_dim(monkeypatch, tmp_path):
    """A non-768 embedding_dim must reach the created collection's vector field."""
    module = load_incremental_pipeline_module()
    pymilvus = fake_pymilvus_module()
    created = {}

    class FakeCollection:
        description = ""
        indexes = [object()]
        num_entities = 1

        def __init__(self, name, schema=None):
            if schema is not None:
                created["schema"] = schema

        def load(self):
            return None

        def insert(self, batch):
            return None

        def flush(self):
            return None

        def index(self):
            return object()

    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    monkeypatch.setattr(pymilvus.connections, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(pymilvus.utility, "has_collection", lambda name: False)
    monkeypatch.setattr(pymilvus, "Collection", FakeCollection)
    monkeypatch.setenv("MILVUS_PASSWORD", "test-password")

    input_path = tmp_path / "embedded.jsonl"
    input_path.write_text(json.dumps(embedded_record(1024)) + "\n")

    module.store_milvus_incremental.python_func(
        embedded_data=SimpleNamespace(path=str(input_path)),
        milvus_host="milvus.test",
        milvus_port="19530",
        collection_name="kubeflow_docs",
        embedding_dim=1024,
    )

    vector_field = next(f for f in created["schema"].fields if f.name == "vector")
    assert vector_field.params["dim"] == 1024


def test_incremental_store_rejects_embedding_dim_mismatch(monkeypatch, tmp_path):
    """Inserting 1024-dim vectors into a 768-dim collection must fail loudly."""
    module = load_incremental_pipeline_module()
    pymilvus = fake_pymilvus_module()

    class FakeCollection:
        description = "RAG collection for documentation (v=1)"
        schema = docs_schema(768)
        indexes = [object()]
        num_entities = 0

        def __init__(self, name, schema=None):
            pass

        def load(self):
            raise AssertionError("must fail before load on dim mismatch")

        def insert(self, batch):
            raise AssertionError("must fail before insert on dim mismatch")

    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    monkeypatch.setattr(pymilvus.connections, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(pymilvus.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(pymilvus, "Collection", FakeCollection)
    monkeypatch.setenv("MILVUS_PASSWORD", "test-password")

    input_path = tmp_path / "embedded.jsonl"
    input_path.write_text(json.dumps(embedded_record(1024)) + "\n")

    with pytest.raises(RuntimeError, match="vector_dim=768"):
        module.store_milvus_incremental.python_func(
            embedded_data=SimpleNamespace(path=str(input_path)),
            milvus_host="milvus.test",
            milvus_port="19530",
            collection_name="kubeflow_docs",
            embedding_dim=1024,
        )


def test_incremental_store_accepts_matching_schema(monkeypatch, tmp_path):
    """The guard must not reject a collection the full pipeline just created."""
    module = load_incremental_pipeline_module()
    pymilvus = fake_pymilvus_module()
    inserted = []

    class FakeCollection:
        description = "RAG collection for documentation (v=1)"
        schema = docs_schema(768)
        indexes = [object()]
        num_entities = 1

        def __init__(self, name, schema=None):
            pass

        def load(self):
            return None

        def insert(self, batch):
            inserted.extend(batch)

        def flush(self):
            return None

        def index(self):
            return object()

    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    monkeypatch.setattr(pymilvus.connections, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(pymilvus.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(pymilvus, "Collection", FakeCollection)
    monkeypatch.setenv("MILVUS_PASSWORD", "test-password")

    input_path = tmp_path / "embedded.jsonl"
    input_path.write_text(json.dumps(embedded_record(768)) + "\n")

    module.store_milvus_incremental.python_func(
        embedded_data=SimpleNamespace(path=str(input_path)),
        milvus_host="milvus.test",
        milvus_port="19530",
        collection_name="kubeflow_docs",
        embedding_dim=768,
    )

    assert len(inserted) == 1
    assert len(inserted[0]["vector"]) == 768
