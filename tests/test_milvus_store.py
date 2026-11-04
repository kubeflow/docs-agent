"""Tests for crash-safe document replacement in the Milvus ingestion writer."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
UTILS_DIR = PIPELINES_DIR / "utils"
COLLECTION = "kubeflow_docs"
DIM = 768


@pytest.fixture
def store(monkeypatch):
    """Load milvus_store with its heavy third-party and sibling imports stubbed."""
    pymilvus = ModuleType("pymilvus")
    for name in ("CollectionSchema", "DataType", "FieldSchema", "Function", "FunctionType", "MilvusClient"):
        setattr(pymilvus, name, MagicMock())

    canonical = ModuleType("canonical_rag_ingest")
    canonical.DEFAULT_OVERLAP_TOKENS = 50
    canonical.DEFAULT_TARGET_TOKENS = 400
    canonical.build_milvus_records = lambda *args, **kwargs: []

    utils = ModuleType("utils")
    utils.DEFAULT_EMBEDDING_BATCH_SIZE = 8
    utils.DOCS_COLLECTION = COLLECTION
    utils.embed_texts = lambda *args, **kwargs: []
    utils.truncate_for_tei = lambda text, *args, **kwargs: text

    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    monkeypatch.setitem(sys.modules, "canonical_rag_ingest", canonical)
    monkeypatch.setitem(sys.modules, "utils", utils)

    spec = importlib.util.spec_from_file_location("milvus_store_under_test", UTILS_DIR / "milvus_store.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("MILVUS_MAINTENANCE_LOCK", "lock")
    return module


class InjectedFailure(RuntimeError):
    pass


class FakeIterator:
    def __init__(self, rows, batch_size):
        self._rows = rows
        self._batch_size = batch_size
        self._offset = 0
        self.closed = False

    def next(self):
        page = self._rows[self._offset : self._offset + self._batch_size]
        self._offset += self._batch_size
        return page

    def close(self):
        self.closed = True


class FakeMilvus:
    """In-memory stand-in for the MilvusClient calls used by the writer."""

    def __init__(self, schema_description):
        self.rows = {}
        self.next_id = 1000
        self.exists = True
        self.description = schema_description
        self.insert_calls = 0
        self.delete_calls = []
        self.fail_inserts = set()
        self.fail_deletes = set()
        self.iterators = []
        self.queries = []
        self.dropped = False
        self.flushes = 0
        self.loads = 0

    # Seeding bypasses insert() so injected failures never fire during setup.
    def seed(self, rows):
        ids = []
        for row in rows:
            self.next_id += 1
            self.rows[self.next_id] = dict(row)
            ids.append(self.next_id)
        return ids

    def get_server_version(self):
        return "v2.6.14"

    def has_collection(self, name):
        return self.exists

    def describe_collection(self, name):
        return {"description": self.description}

    def drop_collection(self, name):
        self.dropped = True
        self.exists = False
        self.rows.clear()

    def create_collection(self, **kwargs):
        self.exists = True

    def insert(self, collection_name, data):
        self.insert_calls += 1
        if self.insert_calls in self.fail_inserts:
            raise InjectedFailure(f"insert boom #{self.insert_calls}")
        return {"insert_count": len(data), "ids": self.seed(data)}

    def delete(self, collection_name, ids=None, filter=None):
        assert filter is None, "replacement must delete by primary key, not by filter"
        call_number = len(self.delete_calls) + 1
        self.delete_calls.append(list(ids))
        if call_number in self.fail_deletes:
            raise InjectedFailure(f"delete boom #{call_number}")
        deleted = 0
        for pk in ids:
            if self.rows.pop(pk, None) is not None:
                deleted += 1
        return {"delete_count": deleted}

    def query_iterator(self, collection_name, batch_size, filter, output_fields):
        wanted = set(json.loads("[" + re.search(r"in \[(.*)\]", filter).group(1) + "]"))
        self.queries.append(wanted)
        matches = [{"id": pk} for pk, row in sorted(self.rows.items()) if row["document_id"] in wanted]
        iterator = FakeIterator(matches, batch_size)
        self.iterators.append(iterator)
        return iterator

    def flush(self, name):
        self.flushes += 1

    def load_collection(self, name):
        self.loads += 1

    def get_collection_stats(self, name):
        return {"row_count": len(self.rows)}

    def list_indexes(self, collection_name):
        return ["vector", "sparse_vector"]

    def snapshot(self):
        return {pk: dict(row) for pk, row in self.rows.items()}

    def document_chunks(self, document_id):
        return sorted(
            (row["chunk_index"], row["content_text"]) for row in self.rows.values() if row["document_id"] == document_id
        )


@pytest.fixture
def milvus(store):
    return FakeMilvus(store.SCHEMA_DESCRIPTION)


def make_records(generation, documents):
    """documents maps document_id -> chunk count."""
    return [
        {
            "file_unique_id": document_id,
            "content_text": f"{generation}:{document_id}:{index}",
            "embedding": [0.0] * DIM,
            "chunk_index": index,
            "citation_url": f"https://example.org/{document_id}",
            "file_path": f"{document_id}.md",
            "doc_type": "documentation",
        }
        for document_id, count in documents.items()
        for index in range(count)
    ]


def write_jsonl(tmp_path, records, name="embedded.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return str(path)


def seed_generation(store, milvus, generation, documents):
    rows = [store.prepare_v4_insert_row(record) for record in make_records(generation, documents)]
    return milvus.seed(rows)


def run_store(store, milvus, path, *, clean_rebuild=False):
    return store.store_embedded_records(
        path,
        milvus_host="milvus",
        milvus_port="19530",
        collection_name=COLLECTION,
        clean_rebuild=clean_rebuild,
        clean_rebuild_confirmation=f"DROP {COLLECTION}" if clean_rebuild else "",
        maintenance_lock_token="lock" if clean_rebuild else "",
        client=milvus,
    )


def assert_single_generation(milvus, generation, documents):
    expected = {
        document_id: [(index, f"{generation}:{document_id}:{index}") for index in range(count)]
        for document_id, count in documents.items()
    }
    actual = {document_id: milvus.document_chunks(document_id) for document_id in documents}
    assert actual == expected
    assert len(milvus.rows) == sum(documents.values())


OLD_DOCS = {"doc-a": 3, "doc-b": 2}
NEW_DOCS = {"doc-a": 4, "doc-b": 2}


def test_successful_replacement_deletes_only_previous_generation(store, milvus, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INSERT_BATCH_SIZE", 3)
    old_ids = seed_generation(store, milvus, "old", OLD_DOCS)
    untouched_ids = seed_generation(store, milvus, "other", {"doc-untouched": 2})
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))

    metrics = run_store(store, milvus, path)

    assert not set(old_ids) & set(milvus.rows)
    assert set(untouched_ids) <= set(milvus.rows)
    for document_id, count in NEW_DOCS.items():
        assert milvus.document_chunks(document_id) == [(index, f"new:{document_id}:{index}") for index in range(count)]
    assert milvus.document_chunks("doc-untouched") == [(0, "other:doc-untouched:0"), (1, "other:doc-untouched:1")]
    assert len(milvus.rows) == sum(NEW_DOCS.values()) + 2
    assert sorted(pk for call in milvus.delete_calls for pk in call) == sorted(old_ids)
    assert metrics["record_count"] == 6
    assert metrics["inserted_count"] == 6
    assert metrics["entity_count"] == 8
    assert metrics["clean_rebuild"] is False
    assert milvus.flushes == 1
    assert all(iterator.closed for iterator in milvus.iterators)


def test_first_batch_insert_failure_keeps_old_documents(store, milvus, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INSERT_BATCH_SIZE", 3)
    seed_generation(store, milvus, "old", OLD_DOCS)
    before = milvus.snapshot()
    milvus.fail_inserts = {1}
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))

    with pytest.raises(InjectedFailure, match="insert boom #1"):
        run_store(store, milvus, path)

    assert milvus.snapshot() == before
    assert milvus.delete_calls == []
    assert_single_generation(milvus, "old", OLD_DOCS)


def test_later_batch_insert_failure_rolls_back_new_rows_only(store, milvus, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INSERT_BATCH_SIZE", 2)
    seed_generation(store, milvus, "old", OLD_DOCS)
    before = milvus.snapshot()
    milvus.fail_inserts = {3}
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))

    with pytest.raises(InjectedFailure, match="insert boom #3"):
        run_store(store, milvus, path)

    assert milvus.snapshot() == before
    assert_single_generation(milvus, "old", OLD_DOCS)
    rolled_back = [pk for call in milvus.delete_calls for pk in call]
    assert len(rolled_back) == 4
    assert not set(rolled_back) & set(before)


def test_cleanup_failure_preserves_original_error_and_logs(store, milvus, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "INSERT_BATCH_SIZE", 2)
    seed_generation(store, milvus, "old", OLD_DOCS)
    old_snapshot = milvus.snapshot()
    milvus.fail_inserts = {2}
    milvus.fail_deletes = {1}
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))

    with pytest.raises(InjectedFailure, match="insert boom #2"):
        run_store(store, milvus, path)

    assert old_snapshot.items() <= milvus.snapshot().items()
    out = capsys.readouterr().out
    assert "failed to roll back 2 inserted chunks" in out
    assert "delete boom #1" in out
    # The stranded rows are the first insert batch of the failed run.
    assert len(milvus.rows) == len(old_snapshot) + 2


def test_old_row_deletion_failure_keeps_new_rows_and_propagates(store, milvus, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INSERT_BATCH_SIZE", 3)
    monkeypatch.setattr(store, "PRIMARY_KEY_BATCH_SIZE", 2)
    old_ids = seed_generation(store, milvus, "old", OLD_DOCS)
    milvus.fail_deletes = {2}
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))

    with pytest.raises(InjectedFailure, match="delete boom #2"):
        run_store(store, milvus, path)

    # Both batches of the old keys were attempted in order; only the first committed.
    assert len(milvus.delete_calls) == 2
    assert milvus.delete_calls[0] == old_ids[:2]
    assert not set(old_ids[:2]) & set(milvus.rows)
    assert set(old_ids[2:]) <= set(milvus.rows)
    # Every replacement chunk is still stored, so each document is complete.
    for document_id, count in NEW_DOCS.items():
        new_chunks = [c for c in milvus.document_chunks(document_id) if c[1].startswith("new:")]
        assert new_chunks == [(i, f"new:{document_id}:{i}") for i in range(count)]


def test_rerun_after_insert_failure_converges_to_one_generation(store, milvus, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INSERT_BATCH_SIZE", 2)
    seed_generation(store, milvus, "old", OLD_DOCS)
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))
    milvus.fail_inserts = {2}

    with pytest.raises(InjectedFailure):
        run_store(store, milvus, path)
    assert_single_generation(milvus, "old", OLD_DOCS)

    milvus.fail_inserts = set()
    run_store(store, milvus, path)

    assert_single_generation(milvus, "new", NEW_DOCS)


def test_rerun_after_crash_between_insert_and_delete_converges(store, milvus, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "INSERT_BATCH_SIZE", 3)
    seed_generation(store, milvus, "old", OLD_DOCS)
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))
    milvus.fail_deletes = {1}

    with pytest.raises(InjectedFailure):
        run_store(store, milvus, path)
    # Both generations are visible: duplicates rather than data loss.
    assert len(milvus.rows) == sum(OLD_DOCS.values()) + sum(NEW_DOCS.values())

    milvus.fail_deletes = set()
    run_store(store, milvus, path)

    assert_single_generation(milvus, "new", NEW_DOCS)


def test_all_old_keys_are_found_and_deleted_across_pages_and_batches(store, milvus, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PRIMARY_KEY_BATCH_SIZE", 4)
    monkeypatch.setattr(store, "DELETE_BATCH_SIZE", 2)
    old_docs = {f"doc-{n}": 5 for n in range(5)}
    old_ids = seed_generation(store, milvus, "old", old_docs)
    new_docs = {f"doc-{n}": 1 for n in range(5)}
    path = write_jsonl(tmp_path, make_records("new", new_docs))

    run_store(store, milvus, path)

    assert len(milvus.queries) == 3
    deleted = [pk for call in milvus.delete_calls for pk in call]
    assert sorted(deleted) == sorted(old_ids)
    assert len(deleted) == len(set(deleted)) == 25
    assert all(len(call) <= 4 for call in milvus.delete_calls)
    assert_single_generation(milvus, "new", new_docs)


def test_document_ids_with_quotes_are_escaped_in_query_filter(store, milvus, tmp_path):
    tricky = 'docs/"quoted"\\path.md'
    seed_generation(store, milvus, "old", {tricky: 2})
    path = write_jsonl(tmp_path, make_records("new", {tricky: 1}))

    run_store(store, milvus, path)

    assert milvus.queries == [{tricky}]
    assert_single_generation(milvus, "new", {tricky: 1})


def test_empty_input_raises_before_touching_milvus(store, milvus, tmp_path):
    seed_generation(store, milvus, "old", OLD_DOCS)
    before = milvus.snapshot()
    path = write_jsonl(tmp_path, [])

    with pytest.raises(ValueError, match="No embedded records found"):
        run_store(store, milvus, path)

    assert milvus.snapshot() == before
    assert milvus.insert_calls == 0
    assert milvus.delete_calls == []
    assert milvus.queries == []


def test_clean_rebuild_drops_and_inserts_without_incremental_replacement(store, milvus, tmp_path):
    seed_generation(store, milvus, "old", OLD_DOCS)
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))

    metrics = run_store(store, milvus, path, clean_rebuild=True)

    assert milvus.dropped is True
    assert milvus.queries == []
    assert milvus.delete_calls == []
    assert_single_generation(milvus, "new", NEW_DOCS)
    assert metrics["clean_rebuild"] is True
    assert metrics["inserted_count"] == 6
    assert metrics["entity_count"] == 6


def test_clean_rebuild_still_requires_confirmation(store, milvus, tmp_path):
    seed_generation(store, milvus, "old", OLD_DOCS)
    before = milvus.snapshot()
    path = write_jsonl(tmp_path, make_records("new", NEW_DOCS))

    with pytest.raises(ValueError, match="typed confirmation"):
        store.store_embedded_records(
            path,
            milvus_host="milvus",
            milvus_port="19530",
            collection_name=COLLECTION,
            clean_rebuild=True,
            clean_rebuild_confirmation="yes",
            maintenance_lock_token="lock",
            client=milvus,
        )

    assert milvus.snapshot() == before
    assert milvus.dropped is False
