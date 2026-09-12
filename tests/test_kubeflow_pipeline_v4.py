"""Focused tests for production docs v4 Kubeflow pipeline helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

import canonical_rag_ingest  # noqa: E402
import hugo_ingest  # noqa: E402
from milvus_store import (  # noqa: E402
    APPROVED_DOCS_COLLECTION,
    BM25_INPUT_FIELD,
    CLEAN_REBUILD_CONFIRMATION,
    DENSE_DIM,
    DENSE_FIELD,
    DOCUMENT_ID_FIELD,
    MAINTENANCE_LOCK_ENV,
    SCHEMA_VERSION,
    SPARSE_FIELD,
    build_lean_v4_schema,
    build_v4_index_params,
    check_milvus_health,
    chunk_github_jsonl,
    compute_validation_metrics,
    embed_chunk_records,
    prepare_v4_insert_row,
    schema_version_matches,
    truncate_utf8,
    validate_clean_rebuild_gates,
    validate_embedding_vectors,
    validate_production_collection_name,
)
from pymilvus import DataType, FunctionType  # noqa: E402

pytestmark = pytest.mark.unit

KUBEFLOW_DOC = """+++
title = "Install Kubeflow Pipelines"
description = "Standalone install guide"
weight = 42
+++

## Prerequisites

Install the [KFP SDK](https://pypi.org/project/kfp/) before continuing.
"""

RELEASE_DOC = """+++
title = "Kubeflow Community Distribution 1.9"
version = "1.9"
+++

## Kubeflow Community Distribution 1.9

<div class="table-responsive">
<table class="table table-bordered">
  <tbody>
    <tr>
      <th class="table-light">Release Date</th>
      <td>
        2024-07-22
      </td>
    </tr>
  </tbody>
</table>
</div>
"""


def _write_github_jsonl(path: Path, *records: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TestIngestModules:
    def test_hugo_and_canonical_are_importable(self):
        assert hasattr(hugo_ingest, "parse_frontmatter")
        assert hasattr(canonical_rag_ingest, "build_milvus_records")


class TestSafetyGates:
    def test_accepts_approved_collection(self):
        validate_production_collection_name(APPROVED_DOCS_COLLECTION)

    def test_rejects_unexpected_collection(self):
        with pytest.raises(ValueError, match="Refusing unexpected collection name"):
            validate_production_collection_name("kubeflow_docs_hybrid_v4_candidate")

    def test_clean_rebuild_requires_confirmation(self, monkeypatch):
        monkeypatch.setenv(MAINTENANCE_LOCK_ENV, "lock-123")
        with pytest.raises(ValueError, match="typed confirmation"):
            validate_clean_rebuild_gates(
                clean_rebuild=True,
                clean_rebuild_confirmation="wrong",
                maintenance_lock_token="lock-123",
            )

    def test_clean_rebuild_requires_maintenance_lock_env(self, monkeypatch):
        monkeypatch.delenv(MAINTENANCE_LOCK_ENV, raising=False)
        with pytest.raises(RuntimeError, match=MAINTENANCE_LOCK_ENV):
            validate_clean_rebuild_gates(
                clean_rebuild=True,
                clean_rebuild_confirmation=CLEAN_REBUILD_CONFIRMATION,
                maintenance_lock_token="lock-123",
            )

    def test_clean_rebuild_requires_matching_token(self, monkeypatch):
        monkeypatch.setenv(MAINTENANCE_LOCK_ENV, "lock-123")
        with pytest.raises(RuntimeError, match="maintenance_lock_token"):
            validate_clean_rebuild_gates(
                clean_rebuild=True,
                clean_rebuild_confirmation=CLEAN_REBUILD_CONFIRMATION,
                maintenance_lock_token="other",
            )

    def test_clean_rebuild_passes_with_valid_gates(self, monkeypatch):
        monkeypatch.setenv(MAINTENANCE_LOCK_ENV, "lock-123")
        validate_clean_rebuild_gates(
            clean_rebuild=True,
            clean_rebuild_confirmation=CLEAN_REBUILD_CONFIRMATION,
            maintenance_lock_token="lock-123",
        )


class TestLeanV4Schema:
    def test_schema_marker_and_bm25_function(self):
        schema = build_lean_v4_schema()
        assert f"v={SCHEMA_VERSION}" in schema.description
        assert len(schema.functions) == 1
        function = schema.functions[0]
        assert function.type == FunctionType.BM25
        assert function.input_field_names == [BM25_INPUT_FIELD]
        assert function.output_field_names == [SPARSE_FIELD]

    def test_release_date_nullable(self):
        schema = build_lean_v4_schema()
        release_field = next(field for field in schema.fields if field.name == "release_date")
        assert release_field.dtype == DataType.INT64
        assert release_field.nullable is True

    def test_index_params_include_dense_and_sparse(self):
        params = build_v4_index_params()
        serialized = [item.to_dict() for item in params]
        dense = next(item for item in serialized if item["field_name"] == DENSE_FIELD)
        sparse = next(item for item in serialized if item["field_name"] == SPARSE_FIELD)
        assert dense["index_type"] == "FLAT"
        assert dense["metric_type"] == "COSINE"
        assert sparse["index_type"] == "SPARSE_INVERTED_INDEX"
        assert sparse["metric_type"] == "BM25"


class TestChunkAndEmbed:
    def test_chunk_github_jsonl_uses_canonical_parser(self, tmp_path):
        jsonl_path = tmp_path / "github.jsonl"
        _write_github_jsonl(
            jsonl_path,
            {
                "path": "content/en/docs/components/pipelines/install.md",
                "file_name": "install.md",
                "content": KUBEFLOW_DOC,
            },
        )
        records = chunk_github_jsonl(
            str(jsonl_path),
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        assert records[0]["section_path"]
        assert records[0]["parser_version"] == "1.0.0"
        assert records[0]["release_date"] is None

    def test_release_date_extracted_for_release_docs(self, tmp_path):
        jsonl_path = tmp_path / "release.jsonl"
        _write_github_jsonl(
            jsonl_path,
            {
                "path": "content/en/docs/kubeflow-distribution/releases/kubeflow-1.9.md",
                "file_name": "kubeflow-1.9.md",
                "content": RELEASE_DOC,
            },
        )
        records = chunk_github_jsonl(
            str(jsonl_path),
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert all(record["doc_type"] == "release" for record in records)
        assert all(record["release_date"] == 1721606400 for record in records)

    def test_embed_chunk_records_validates_768_dim(self, monkeypatch):
        records = [
            {
                "content_text": "Install Kubeflow Pipelines.",
            }
        ]

        def fake_embed(texts, url, batch_size):
            assert url == "http://tei/embed"
            return [[0.1] * DENSE_DIM for _ in texts]

        monkeypatch.setattr("milvus_store.embed_texts", fake_embed)
        embedded = embed_chunk_records(
            records,
            embeddings_service_url="http://tei/embed",
            embedding_batch_size=8,
        )
        assert len(embedded[0]["embedding"]) == DENSE_DIM

    def test_rejects_wrong_embedding_dimension(self):
        with pytest.raises(ValueError, match=f"expected {DENSE_DIM}-dim"):
            validate_embedding_vectors([[0.1, 0.2]])


class TestInsertRowAndMetrics:
    def test_truncate_utf8_respects_byte_limit(self):
        text = "€" * 20
        truncated = truncate_utf8(text, 10)
        assert len(truncated.encode("utf-8")) <= 10

    def test_prepare_v4_insert_row_maps_document_id(self):
        row = prepare_v4_insert_row(
            {
                "file_unique_id": "kubeflow/website:install.md",
                "content_text": "Install Kubeflow Pipelines.",
                "embedding": [0.1] * DENSE_DIM,
                "chunk_index": 0,
                "citation_url": "https://example/docs/install",
                "file_path": "content/en/docs/install.md",
                "title": "Install",
                "section_path": "Install > Prerequisites",
                "doc_type": "documentation",
                "version": "",
                "release_date": None,
            }
        )
        assert row[DOCUMENT_ID_FIELD] == "kubeflow/website:install.md"
        assert row[BM25_INPUT_FIELD].startswith("Install")
        assert len(row[DENSE_FIELD]) == DENSE_DIM
        assert "release_date" not in row

    def test_prepare_v4_insert_row_includes_release_date(self):
        row = prepare_v4_insert_row(
            {
                "file_unique_id": "kubeflow/website:release.md",
                "content_text": "Release notes",
                "embedding": [0.2] * DENSE_DIM,
                "chunk_index": 0,
                "citation_url": "https://example/docs/release",
                "file_path": "content/en/docs/release.md",
                "title": "Release",
                "section_path": "Release",
                "doc_type": "release",
                "version": "1.9",
                "release_date": 1721606400,
            }
        )
        assert row["release_date"] == 1721606400

    def test_compute_validation_metrics_compact(self):
        records = [
            {"doc_type": "release", "release_date": 1721606400},
            {"doc_type": "release", "release_date": None},
            {"doc_type": "documentation", "release_date": None},
        ]
        metrics = compute_validation_metrics(
            collection_name=APPROVED_DOCS_COLLECTION,
            records=records,
            inserted_count=3,
            entity_count=3,
            dense_ready=True,
            sparse_ready=True,
            clean_rebuild=True,
        )
        assert metrics["schema_version"] == SCHEMA_VERSION
        assert metrics["release_doc_count"] == 2
        assert metrics["release_date_count"] == 1
        assert metrics["release_date_fill_rate"] == 0.5
        assert metrics["bm25_index_ready"] is True


class TestMilvusHealth:
    def test_refuses_unhealthy_milvus(self):
        client = MagicMock()
        client.get_server_version.side_effect = RuntimeError("down")
        with pytest.raises(RuntimeError, match="Milvus health check failed"):
            check_milvus_health(client)

    def test_accepts_healthy_milvus(self):
        client = MagicMock()
        client.get_server_version.return_value = "2.6.22"
        check_milvus_health(client)


class TestPipelineCompile:
    def test_github_rag_pipeline_compiles(self, tmp_path, monkeypatch):
        monkeypatch.chdir(PIPELINES_DIR)
        output_path = tmp_path / "github_rag_pipeline.yaml"

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "kubeflow_pipeline",
            PIPELINES_DIR / "kubeflow-pipeline.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        import kfp

        kfp.compiler.Compiler().compile(
            pipeline_func=module.github_rag_pipeline,
            package_path=str(output_path),
        )
        assert output_path.is_file()
        payload = output_path.read_text(encoding="utf-8")
        assert "clean_rebuild" in payload
        assert "target_tokens" in payload
        assert "docs-rag-ingest" in payload
        assert "HUGO_INGEST_SOURCE" not in payload
        assert schema_version_matches(
            f"RAG lean hybrid collection for documentation (v={SCHEMA_VERSION}, hybrid=bm25+dense)"
        )
