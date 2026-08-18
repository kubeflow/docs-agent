"""Tests for the full documentation ingestion pipeline."""

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


def load_docs_pipeline_module():
    pytest.importorskip("kfp", reason="pipeline component tests require the KFP SDK")
    pipeline_path = PIPELINES_DIR / "kubeflow-pipeline.py"
    spec = importlib.util.spec_from_file_location("kubeflow_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_docs_schema():
    """Match the compatible pre-versioned schema currently used in Milvus."""
    return CollectionSchema(
        [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        ],
        description="",
    )


def test_store_accepts_compatible_legacy_schema_without_last_updated(monkeypatch, tmp_path):
    module = load_docs_pipeline_module()
    inserted = []
    pymilvus = fake_pymilvus_module()

    class FakeCollection:
        description = ""
        schema = legacy_docs_schema()
        indexes = [object()]
        num_entities = 1

        def load(self):
            return None

        def query(self, **kwargs):
            return []

        def delete(self, expr):
            raise AssertionError("No old records were returned, so delete must not run")

        def insert(self, batch):
            inserted.extend(batch)

        def flush(self):
            return None

        def has_index(self):
            return True

    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    monkeypatch.setattr(pymilvus.connections, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(pymilvus.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(pymilvus, "Collection", lambda name: FakeCollection())
    monkeypatch.setenv("MILVUS_PASSWORD", "test-password")

    input_path = tmp_path / "embedded.jsonl"
    record = {
        "file_unique_id": "website:content/en/docs/components/katib/example.md",
        "repo_name": "website",
        "file_path": "content/en/docs/components/katib/example.md",
        "file_name": "example.md",
        "citation_url": "https://www.kubeflow.org/docs/components/katib/example",
        "chunk_index": 0,
        "content_text": "Katib Experiment evidence",
        "embedding": [0.0] * 768,
    }
    input_path.write_text(json.dumps(record) + "\n")

    module.store_milvus.python_func(
        embedded_data=SimpleNamespace(path=str(input_path)),
        milvus_host="milvus.test",
        milvus_port="19530",
        collection_name="kubeflow_docs",
    )

    assert len(inserted) == 1
    assert "last_updated" not in inserted[0]
    assert inserted[0]["citation_url"] == record["citation_url"]


def test_docs_cleaner_preserves_markdown_link_adjacent_yaml(monkeypatch, tmp_path):
    module = load_docs_pipeline_module()

    class FakeEmbeddingResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [[0.0] * 768]

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeEmbeddingResponse())
    source_path = tmp_path / "docs.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "path": "content/en/docs/components/katib/configure-experiment.md",
                "file_name": "configure-experiment.md",
                "content": """---
title: Configure an Experiment
---
### Running Katib Experiment with Istio

Katib Experiment from [this directory](https://github.com/kubeflow/katib/tree/main/examples)
doesn't work with [Istio sidecar injection](https://istio.io/latest/docs/setup/additional-setup/sidecar-injection/#automatic-sidecar-injection).
Specify this annotation:

```yaml
metadata:
  annotations:
    \"sidecar.istio.io/inject\": \"false\"
```
""",
            }
        )
        + "\n"
    )
    output_path = tmp_path / "embedded.jsonl"

    module.chunk_and_embed.python_func(
        github_data=SimpleNamespace(path=str(source_path)),
        repo_name="website",
        base_url="https://www.kubeflow.org/docs",
        chunk_size=2000,
        chunk_overlap=60,
        embeddings_service_url="http://embeddings.test/embed",
        embedding_batch_size=8,
        embedded_data=SimpleNamespace(path=str(output_path)),
    )

    record = json.loads(output_path.read_text())
    assert "title: Configure an Experiment" not in record["content_text"]
    assert "this directory" in record["content_text"]
    assert '"sidecar.istio.io/inject": "false"' in record["content_text"]
    assert "metadata:\n annotations:" in record["content_text"]


def test_store_replaces_legacy_chunk_ids_by_repo_and_file_path(monkeypatch, tmp_path):
    module = load_docs_pipeline_module()
    queried = []
    deleted = []
    pymilvus = fake_pymilvus_module()

    class FakeCollection:
        description = ""
        schema = legacy_docs_schema()
        indexes = [object()]
        num_entities = 3

        def load(self):
            return None

        def query(self, **kwargs):
            queried.append(kwargs["expr"])
            return [{"id": 1}, {"id": 2}]

        def delete(self, expr):
            deleted.append(expr)

        def insert(self, batch):
            return None

        def flush(self):
            return None

        def has_index(self):
            return True

    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    monkeypatch.setattr(pymilvus.connections, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(pymilvus.utility, "has_collection", lambda name: True)
    monkeypatch.setattr(pymilvus, "Collection", lambda name: FakeCollection())
    monkeypatch.setenv("MILVUS_PASSWORD", "test-password")

    input_path = tmp_path / "embedded.jsonl"
    record = {
        "file_unique_id": "website:content/en/docs/components/katib/example.md",
        "repo_name": "website",
        "file_path": "content/en/docs/components/katib/example.md",
        "file_name": "example.md",
        "citation_url": "https://www.kubeflow.org/docs/components/katib/example",
        "chunk_index": 0,
        "content_text": "Katib Experiment evidence",
        "embedding": [0.0] * 768,
    }
    input_path.write_text(json.dumps(record) + "\n")

    module.store_milvus.python_func(
        embedded_data=SimpleNamespace(path=str(input_path)),
        milvus_host="milvus.test",
        milvus_port="19530",
        collection_name="kubeflow_docs",
    )

    assert queried == ['repo_name == "website" and file_path in ["content/en/docs/components/katib/example.md"]']
    assert deleted == queried
