"""Lease coordination and fencing for concurrent Milvus ingestion workers.

Provides mutual exclusion and stale-worker fencing to prevent overlapping
ingestion runs from concurrently mutating the same Milvus collection and
corrupting document generations.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Protocol, runtime_checkable


class IngestionConflictError(RuntimeError):
    """Raised when an ingestion worker fails to acquire the collection lease."""


class StaleWorkerError(RuntimeError):
    """Raised when an active worker detects its lease was lost or superseded."""


@runtime_checkable
class CollectionLeaseProvider(Protocol):
    """Protocol for exclusive collection lease coordination."""

    def acquire(self, collection_name: str, *, ttl_seconds: int = 300) -> str:
        """Acquire an exclusive lease for the target collection.

        Returns an acquisition token (fencing token) on success.
        Raises IngestionConflictError if another active worker holds the lease.
        """
        ...

    def renew(self, collection_name: str, token: str, *, ttl_seconds: int = 300) -> None:
        """Renew an active lease.

        Raises StaleWorkerError if the lease was lost or expired.
        """
        ...

    def release(self, collection_name: str, token: str) -> None:
        """Release the lease upon completion.

        No-op if the token is already expired or relinquished.
        """
        ...

    def validate(self, collection_name: str, token: str) -> None:
        """Validate that the worker still holds a valid unexpired lease.

        Raises StaleWorkerError if the lease has expired or was acquired by another worker.
        """
        ...


class InMemoryLeaseProvider:
    """Thread-safe in-memory lease provider for local tests and standalone runs."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        # collection_name -> (holder_id, token, expiration_timestamp)
        self._leases: dict[str, tuple[str, str, float]] = {}

    def acquire(self, collection_name: str, *, ttl_seconds: int = 300) -> str:
        now = time.time()
        token = str(uuid.uuid4())
        holder_id = f"worker-{os.getpid()}-{token[:8]}"
        with self._lock:
            current = self._leases.get(collection_name)
            if current is not None:
                _, _, expires_at = current
                if now < expires_at:
                    raise IngestionConflictError(
                        f"Collection '{collection_name}' is currently held by active worker until "
                        f"{expires_at:.1f} (remaining: {expires_at - now:.1f}s)"
                    )
            self._leases[collection_name] = (holder_id, token, now + ttl_seconds)
            return token

    def renew(self, collection_name: str, token: str, *, ttl_seconds: int = 300) -> None:
        now = time.time()
        with self._lock:
            current = self._leases.get(collection_name)
            if current is None or current[1] != token:
                raise StaleWorkerError(
                    f"Worker lease token for collection '{collection_name}' is invalid or superseded."
                )
            holder_id, _, expires_at = current
            if now >= expires_at:
                raise StaleWorkerError(
                    f"Lease for collection '{collection_name}' expired at {expires_at:.1f}."
                )
            self._leases[collection_name] = (holder_id, token, now + ttl_seconds)

    def release(self, collection_name: str, token: str) -> None:
        with self._lock:
            current = self._leases.get(collection_name)
            if current is not None and current[1] == token:
                del self._leases[collection_name]

    def validate(self, collection_name: str, token: str) -> None:
        now = time.time()
        with self._lock:
            current = self._leases.get(collection_name)
            if current is None or current[1] != token:
                raise StaleWorkerError(
                    f"Worker lease token for collection '{collection_name}' is invalid or superseded."
                )
            _, _, expires_at = current
            if now >= expires_at:
                raise StaleWorkerError(
                    f"Lease for collection '{collection_name}' expired at {expires_at:.1f}."
                )
