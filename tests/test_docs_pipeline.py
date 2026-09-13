"""Tests for the v4 documentation ingestion pipeline wrappers."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))


def load_docs_pipeline_module():
    pytest.importorskip("kfp", reason="pipeline component tests require the KFP SDK")
    pipeline_path = PIPELINES_DIR / "kubeflow-pipeline.py"
    spec = importlib.util.spec_from_file_location("kubeflow_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_and_embed_preserves_sidecar_yaml(monkeypatch, tmp_path):
    module = load_docs_pipeline_module()

    def fake_post(*args, **kwargs):
        batch = (kwargs.get("json") or {}).get("inputs") or []

        class FakeEmbeddingResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return [[0.0] * 768 for _ in batch]

        return FakeEmbeddingResponse()

    monkeypatch.setattr("utils.requests.post", fake_post)
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
doesn't work with sidecar injection.
Specify this annotation:

```yaml
metadata:
  annotations:
    "sidecar.istio.io/inject": "false"
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
        target_tokens=350,
        overlap_tokens=50,
        embeddings_service_url="http://embeddings.test/embed",
        embedding_batch_size=8,
        embedded_data=SimpleNamespace(path=str(output_path)),
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines() if line]
    assert records
    content = "\n".join(record["content_text"] for record in records)
    assert "title: Configure an Experiment" not in content
    assert "sidecar.istio.io/inject" in content
    assert "false" in content


def test_store_milvus_delegates_rebuild_gates(monkeypatch, tmp_path):
    module = load_docs_pipeline_module()
    captured = {}

    def fake_store(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)

    import milvus_store

    monkeypatch.setattr(milvus_store, "store_embedded_records", fake_store)

    input_path = tmp_path / "embedded.jsonl"
    input_path.write_text("{}\n")

    module.store_milvus.python_func(
        embedded_data=SimpleNamespace(path=str(input_path)),
        milvus_host="milvus.test",
        milvus_port="19530",
        collection_name="kubeflow_docs",
        clean_rebuild=True,
        clean_rebuild_confirmation="DELETE kubeflow_docs",
        maintenance_lock_token="lock",
    )

    assert captured["path"] == str(input_path)
    assert captured["milvus_host"] == "milvus.test"
    assert captured["collection_name"] == "kubeflow_docs"
    assert captured["clean_rebuild"] is True
    assert captured["clean_rebuild_confirmation"] == "DELETE kubeflow_docs"
    assert captured["maintenance_lock_token"] == "lock"
