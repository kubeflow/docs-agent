"""Tests for the incremental documentation ingestion pipeline (Schema v4)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
UTILS_DIR = PIPELINES_DIR / "utils"

if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


def _ensure_ingest_dependencies():
    """Ensure optional third-party packages are available for offline test execution."""
    if "pymilvus" not in sys.modules:
        try:
            import pymilvus  # noqa: F401
        except ImportError:
            mod = ModuleType("pymilvus")
            mod.CollectionSchema = lambda *args, **kwargs: None
            mod.DataType = SimpleNamespace(
                INT64="INT64",
                VARCHAR="VARCHAR",
                FLOAT_VECTOR="FLOAT_VECTOR",
                SPARSE_FLOAT_VECTOR="SPARSE_FLOAT_VECTOR",
            )
            mod.FieldSchema = lambda *args, **kwargs: None
            mod.Function = lambda *args, **kwargs: None
            mod.FunctionType = SimpleNamespace(BM25="BM25")

            class _MilvusClient:
                def __init__(self, *args, **kwargs):
                    pass

                @staticmethod
                def prepare_index_params():
                    return MagicMock()

            mod.MilvusClient = _MilvusClient
            sys.modules["pymilvus"] = mod

    if "frontmatter" not in sys.modules:
        try:
            import frontmatter  # noqa: F401
        except ImportError:
            fm_mod = ModuleType("frontmatter")
            handlers_mod = ModuleType("frontmatter.default_handlers")
            handlers_mod.TOMLHandler = lambda *args, **kwargs: None
            handlers_mod.YAMLHandler = lambda *args, **kwargs: None
            fm_mod.default_handlers = handlers_mod

            def fake_parse(content, **kwargs):
                meta = {}
                body = content
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        for line in parts[1].splitlines():
                            if ":" in line:
                                k, v = line.split(":", 1)
                                meta[k.strip()] = v.strip().strip('"').strip("'")
                        body = parts[2].strip()
                return meta, body

            fm_mod.parse = fake_parse
            sys.modules["frontmatter"] = fm_mod
            sys.modules["frontmatter.default_handlers"] = handlers_mod

    if "markdownify" not in sys.modules:
        try:
            import markdownify  # noqa: F401
        except ImportError:
            md_mod = ModuleType("markdownify")
            md_mod.markdownify = lambda html, **kwargs: html
            sys.modules["markdownify"] = md_mod

    if "html_table_rescuer" not in sys.modules:
        try:
            import html_table_rescuer  # noqa: F401
        except ImportError:
            ht_mod = ModuleType("html_table_rescuer")
            ht_mod.ParseConfig = lambda *args, **kwargs: None
            ht_mod.RowspanStrategy = SimpleNamespace(REPEAT_VALUE="repeat")
            ht_mod.TableParser = lambda *args, **kwargs: None
            sys.modules["html_table_rescuer"] = ht_mod

    if "kfp" not in sys.modules:
        try:
            import kfp  # noqa: F401
        except ImportError:
            kfp_mod = ModuleType("kfp")
            dsl_mod = ModuleType("kfp.dsl")

            class _Dataset:
                pass

            class _InputOutput:
                def __class_getitem__(cls, item):
                    return cls

            class _Component:
                def __init__(self, func):
                    self.python_func = func

                def __call__(self, *args, **kwargs):
                    return self

                def after(self, *args):
                    return self

            def _component_decorator(*args, **kwargs):
                def decorator(func):
                    return _Component(func)

                return decorator

            def _pipeline_decorator(*args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            dsl_mod.component = _component_decorator
            dsl_mod.pipeline = _pipeline_decorator
            dsl_mod.Input = _InputOutput
            dsl_mod.Output = _InputOutput
            dsl_mod.Dataset = _Dataset

            kfp_mod.dsl = dsl_mod

            class _Compiler:
                def compile(self, *args, **kwargs):
                    pass

            kfp_mod.compiler = SimpleNamespace(Compiler=_Compiler)

            sys.modules["kfp"] = kfp_mod
            sys.modules["kfp.dsl"] = dsl_mod


def load_incremental_pipeline_module():
    _ensure_ingest_dependencies()
    pipeline_path = PIPELINES_DIR / "incremental-pipeline.py"
    spec = importlib.util.spec_from_file_location("incremental_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_specific_files_filters_non_doc_files(monkeypatch, tmp_path):
    module = load_incremental_pipeline_module()

    class FakeResponse:
        def __init__(self, content_str, name):
            import base64

            self._data = {
                "name": name,
                "content": base64.b64encode(content_str.encode("utf-8")).decode("utf-8"),
            }

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def fake_get(url, headers):
        if "overview.md" in url:
            return FakeResponse("# Overview\n\nWelcome to Kubeflow.", "overview.md")
        raise RuntimeError(f"Unexpected URL: {url}")

    monkeypatch.setattr("requests.get", fake_get)

    output_path = tmp_path / "downloaded.jsonl"
    file_paths = json.dumps(
        [
            "content/en/docs/overview.md",
            "content/en/docs/image.png",
            "README.txt",
        ]
    )

    module.download_specific_files.python_func(
        repo_owner="kubeflow",
        repo_name="website",
        file_paths=file_paths,
        github_token="",
        github_data=SimpleNamespace(path=str(output_path)),
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["file_name"] == "overview.md"
    assert "Welcome to Kubeflow." in records[0]["content"]


def test_delete_old_vectors_uses_document_id_filter(monkeypatch):
    module = load_incremental_pipeline_module()
    monkeypatch.setenv("MILVUS_PASSWORD", "test-pass")

    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    deleted_filters = []

    def fake_delete(collection_name, filter):
        deleted_filters.append(filter)
        return {"delete_count": 2}

    mock_client.delete.side_effect = fake_delete

    with patch("pymilvus.MilvusClient", return_value=mock_client):
        with patch("milvus_store.check_milvus_health"):
            file_paths = json.dumps(["content/en/docs/katib/early-stopping.md"])
            module.delete_old_vectors.python_func(
                file_paths=file_paths,
                repo_name="website",
                milvus_host="milvus.test",
                milvus_port="19530",
                collection_name="kubeflow_docs",
            )

    assert len(deleted_filters) == 1
    assert deleted_filters[0] == 'document_id in ["website:content/en/docs/katib/early-stopping.md"]'
    assert mock_client.flush.called
    assert mock_client.load_collection.called


def test_delete_old_vectors_handles_empty_or_invalid_paths(monkeypatch):
    module = load_incremental_pipeline_module()
    monkeypatch.setenv("MILVUS_PASSWORD", "test-pass")

    mock_client = MagicMock()
    with patch("pymilvus.MilvusClient", return_value=mock_client):
        # Empty JSON array
        module.delete_old_vectors.python_func(
            file_paths="[]",
            repo_name="website",
            milvus_host="milvus.test",
            milvus_port="19530",
            collection_name="kubeflow_docs",
        )
        assert not mock_client.delete.called

        # Invalid JSON
        module.delete_old_vectors.python_func(
            file_paths="invalid-json",
            repo_name="website",
            milvus_host="milvus.test",
            milvus_port="19530",
            collection_name="kubeflow_docs",
        )
        assert not mock_client.delete.called


def test_chunk_and_embed_incremental_produces_v4_records(monkeypatch, tmp_path):
    module = load_incremental_pipeline_module()

    class FakeEmbeddingResponse:
        def __init__(self, inputs=None):
            self.inputs = inputs or []

        def raise_for_status(self):
            return None

        def json(self):
            return [[0.1] * 768] * len(self.inputs)

    monkeypatch.setattr(
        "requests.post",
        lambda *args, **kwargs: FakeEmbeddingResponse(kwargs.get("json", {}).get("inputs", [])),
    )

    source_path = tmp_path / "docs.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "path": "content/en/docs/components/katib/early-stopping.md",
                "file_name": "early-stopping.md",
                "content": (
                    "---\n"
                    "title: Early Stopping\n"
                    "---\n\n"
                    "## Configuring Rules\n\n"
                    "Set median_stop_rule: true in the experiment spec.\n"
                ),
            }
        )
        + "\n"
    )

    output_path = tmp_path / "embedded.jsonl"

    module.chunk_and_embed_incremental.python_func(
        github_data=SimpleNamespace(path=str(source_path)),
        repo_name="website",
        base_url="https://www.kubeflow.org/docs",
        target_tokens=350,
        overlap_tokens=50,
        embeddings_service_url="http://embeddings.test/embed",
        embedding_batch_size=8,
        embedded_data=SimpleNamespace(path=str(output_path)),
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(records) > 0
    first = records[0]
    assert "file_unique_id" in first
    assert "content_text" in first
    assert "embedding" in first
    assert len(first["embedding"]) == 768
    assert "title" in first
    assert first["title"] == "Early Stopping"
    assert "section_path" in first


def test_chunk_and_embed_incremental_handles_empty_input(tmp_path):
    module = load_incremental_pipeline_module()

    source_path = tmp_path / "empty.jsonl"
    source_path.write_text("")
    output_path = tmp_path / "embedded.jsonl"

    module.chunk_and_embed_incremental.python_func(
        github_data=SimpleNamespace(path=str(source_path)),
        repo_name="website",
        base_url="https://www.kubeflow.org/docs",
        target_tokens=350,
        overlap_tokens=50,
        embeddings_service_url="http://embeddings.test/embed",
        embedding_batch_size=8,
        embedded_data=SimpleNamespace(path=str(output_path)),
    )

    assert output_path.exists()
    assert output_path.read_text() == ""


def test_store_milvus_incremental_delegates_to_store_embedded_records(monkeypatch, tmp_path):
    module = load_incremental_pipeline_module()

    input_path = tmp_path / "embedded.jsonl"
    input_path.write_text('{"dummy": true}\n')

    called = {}

    def fake_store(
        embedded_data_path,
        *,
        milvus_host,
        milvus_port,
        collection_name,
        clean_rebuild,
        clean_rebuild_confirmation,
        maintenance_lock_token,
    ):
        called["embedded_data_path"] = embedded_data_path
        called["clean_rebuild"] = clean_rebuild
        called["collection_name"] = collection_name
        return {"inserted_count": 1}

    import milvus_store

    monkeypatch.setattr(milvus_store, "store_embedded_records", fake_store)

    module.store_milvus_incremental.python_func(
        embedded_data=SimpleNamespace(path=str(input_path)),
        milvus_host="milvus.test",
        milvus_port="19530",
        collection_name="kubeflow_docs",
    )

    assert called["embedded_data_path"] == str(input_path)
    assert called["clean_rebuild"] is False
    assert called["collection_name"] == "kubeflow_docs"


def test_store_milvus_incremental_skips_empty_file(monkeypatch, tmp_path):
    module = load_incremental_pipeline_module()

    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("")

    import milvus_store

    monkeypatch.setattr(
        milvus_store,
        "store_embedded_records",
        lambda *args, **kwargs: pytest.fail("Should not call store on empty input"),
    )

    module.store_milvus_incremental.python_func(
        embedded_data=SimpleNamespace(path=str(input_path)),
        milvus_host="milvus.test",
        milvus_port="19530",
        collection_name="kubeflow_docs",
    )
