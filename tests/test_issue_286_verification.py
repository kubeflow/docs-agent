"""Integration test verifying that the race condition reported in Issue #286

is prevented by the new lease coordination layer.
"""

import sys
import threading
import time
from pathlib import Path

import pytest
PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))
sys.path.insert(0, str(PIPELINES_DIR / "utils"))

from lease_coordination import InMemoryLeaseProvider, IngestionConflictError, StaleWorkerError


class SimulatedMilvusCollection:
    def __init__(self):
        self.lock = threading.Lock()
        self.documents = {}  # doc_id -> list of chunk versions

    def write_document(self, doc_id, version):
        with self.lock:
            # Simulate append / replace
            if doc_id not in self.documents:
                self.documents[doc_id] = []
            self.documents[doc_id].append(version)


def test_race_condition_prevented_by_lease_provider():
    """Verify that two concurrent ingestion workers targeting the same collection

    cannot execute overlapping mutations. One succeeds, the other is safely rejected.
    """
    lease_provider = InMemoryLeaseProvider()
    collection = SimulatedMilvusCollection()
    collection_name = "docs_v4"

    barrier_start = threading.Barrier(2)
    results = {}

    def worker(worker_id, doc_version):
        barrier_start.wait()
        try:
            token = lease_provider.acquire(collection_name, ttl_seconds=5)
            try:
                # Simulate document replacement mutation
                time.sleep(0.05)
                lease_provider.validate(collection_name, token)
                collection.write_document("doc_1", doc_version)
                results[worker_id] = "SUCCESS"
            finally:
                lease_provider.release(collection_name, token)
        except IngestionConflictError:
            results[worker_id] = "REJECTED_CONFLICT"

    t1 = threading.Thread(target=worker, args=("worker_1", "v1"))
    t2 = threading.Thread(target=worker, args=("worker_2", "v2"))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one worker succeeded and one was safely rejected
    assert ("SUCCESS" in results.values())
    assert ("REJECTED_CONFLICT" in results.values())

    # In Milvus, only ONE version exists (zero conflicting generations!)
    assert len(collection.documents["doc_1"]) == 1
    assert collection.documents["doc_1"][0] in ("v1", "v2")
    print("\n[VERIFIED] Issue #286 Race Condition is successfully prevented! Conflicting write rejected.")


def test_stale_worker_fenced_cannot_corrupt():
    """Verify that a worker whose lease expired cannot commit writes (fencing)."""
    lease_provider = InMemoryLeaseProvider()
    collection = SimulatedMilvusCollection()
    collection_name = "docs_v4"

    # Worker 1 acquires a very short lease (0.1s)
    token1 = lease_provider.acquire(collection_name, ttl_seconds=0.1)

    # Worker 1 pauses / experiences network freeze
    time.sleep(0.15)

    # Worker 2 now acquires the lease
    token2 = lease_provider.acquire(collection_name, ttl_seconds=5)

    # Worker 1 wakes up and tries to validate/write
    with pytest.raises(StaleWorkerError):
        lease_provider.validate(collection_name, token1)
        collection.write_document("doc_1", "v1_stale")

    # Worker 2 writes safely
    lease_provider.validate(collection_name, token2)
    collection.write_document("doc_1", "v2_fresh")
    lease_provider.release(collection_name, token2)

    # Verify only fresh data was committed
    assert collection.documents["doc_1"] == ["v2_fresh"]
    print("[VERIFIED] Stale worker was safely fenced from corrupting collection state.")
