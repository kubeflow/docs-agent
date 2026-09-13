"""Talk to Milvus: connect, search, and (for docs auto) plan → search → rerank."""

from __future__ import annotations

import os
import threading

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker

from embeddings_client import embed_query
from intent_router import (
    downgrade_plan_for_collection,
    pick_search_plan,
    rerank_hits_after_search,
    retrieval_metadata,
)
from rag_collections import (
    CODE_COLLECTION,
    DOCS_COLLECTION,
    DENSE_DIM,
    DENSE_FIELD,
    ISSUES_COLLECTION,
    SPARSE_FIELD,
)

def _env(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


CLUSTER_MILVUS_URI = _env("CLUSTER_MILVUS_URI", "http://milvus-milvus.ml-infra.svc.cluster.local:19530")
LOCAL_MILVUS_URI = _env("LOCAL_MILVUS_URI", "http://127.0.0.1:19530")

MILVUS_LOCAL_MODE = os.getenv("MILVUS_LOCAL_MODE", "").lower() in ("1", "true", "yes")
MILVUS_URI = _env("MILVUS_URI", LOCAL_MILVUS_URI if MILVUS_LOCAL_MODE else CLUSTER_MILVUS_URI)
MILVUS_USER = _env("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
EMBEDDINGS_URL = _env(
    "EMBEDDINGS_URL",
    "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed",
)

SEARCH_MODE = os.getenv("SEARCH_MODE", "dense").strip().lower()
ISSUES_SEARCH_MODE = os.getenv("ISSUES_SEARCH_MODE", "dense").strip().lower()
CODE_SEARCH_MODE = os.getenv("CODE_SEARCH_MODE", "dense").strip().lower()

COLLECTION_NAME = os.getenv("COLLECTION_NAME") or DOCS_COLLECTION
ISSUES_COLLECTION_NAME = os.getenv("ISSUES_COLLECTION_NAME", ISSUES_COLLECTION)
CODE_COLLECTION_NAME = os.getenv("CODE_COLLECTION_NAME", CODE_COLLECTION)

HYBRID_RANKER = os.getenv("HYBRID_RANKER", "rrf").strip().lower()
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
HYBRID_SPARSE_WEIGHT = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))

client: MilvusClient | None = None
_connect_lock = threading.Lock()
_schema_cache: dict[str, dict] = {}


def connect() -> None:
    """Open one Milvus client for this process."""
    global client
    if client is not None:
        return
    with _connect_lock:
        if client is not None:
            return
        if not MILVUS_PASSWORD and not MILVUS_LOCAL_MODE:
            raise RuntimeError(
                "MILVUS_PASSWORD is required (set via Kubernetes secret, not ConfigMap)"
            )
        client = MilvusClient(uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)


def _fields(collection_name: str) -> set[str]:
    connect()
    if collection_name not in _schema_cache:
        _schema_cache[collection_name] = client.describe_collection(collection_name)
    info = _schema_cache[collection_name]
    return {field.get("name") for field in (info.get("fields") or []) if field.get("name")}


def collection_has_bm25(collection_name: str) -> bool:
    """True when this collection has a sparse_vector field (BM25 / hybrid)."""
    return SPARSE_FIELD in _fields(collection_name)


def collection_has_release_fields(collection_name: str) -> bool:
    """True when this collection stores release_date (used to filter/rerank release notes)."""
    return "release_date" in _fields(collection_name)


def _search_mode_for(collection_name: str) -> str:
    if collection_name == COLLECTION_NAME:
        return SEARCH_MODE
    if collection_name == ISSUES_COLLECTION_NAME:
        return ISSUES_SEARCH_MODE
    if collection_name == CODE_COLLECTION_NAME:
        return CODE_SEARCH_MODE
    return "dense"


def _require_embedding(query: str) -> list[float]:
    try:
        embedding = embed_query(query, url=EMBEDDINGS_URL or None)
    except Exception as exc:
        raise RuntimeError(f"Embeddings service request failed: {exc}") from exc
    if len(embedding) != DENSE_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {DENSE_DIM}, got {len(embedding)}"
        )
    return embedding


def _ranker():
    if HYBRID_RANKER == "weighted":
        return WeightedRanker(HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT)
    return RRFRanker(k=HYBRID_RRF_K)


def dense_search(
    collection_name: str,
    embedding: list[float],
    top_k: int,
    output_fields: list[str],
    filter_expr: str = "",
) -> list[dict]:
    params = {
        "collection_name": collection_name,
        "data": [embedding],
        "anns_field": DENSE_FIELD,
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filter_expr:
        params["filter"] = filter_expr
    return client.search(**params)[0]


def bm25_search(
    collection_name: str,
    query: str,
    top_k: int,
    output_fields: list[str],
    filter_expr: str = "",
) -> list[dict]:
    params = {
        "collection_name": collection_name,
        "data": [query],
        "anns_field": SPARSE_FIELD,
        "search_params": {"metric_type": "BM25"},
        "limit": top_k,
        "output_fields": output_fields,
    }
    if filter_expr:
        params["filter"] = filter_expr
    return client.search(**params)[0]


def hybrid_search(
    collection_name: str,
    query: str,
    embedding: list[float],
    top_k: int,
    output_fields: list[str],
    filter_expr: str = "",
    candidate_depth: int | None = None,
) -> list[dict]:
    per_leg = candidate_depth or top_k
    extra = {"limit": per_leg}
    if filter_expr:
        extra["expr"] = filter_expr

    dense_req = AnnSearchRequest(
        data=[embedding],
        anns_field=DENSE_FIELD,
        param={"metric_type": "COSINE"},
        **extra,
    )
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field=SPARSE_FIELD,
        param={"metric_type": "BM25"},
        **extra,
    )
    return client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_req, sparse_req],
        ranker=_ranker(),
        limit=top_k,
        output_fields=output_fields,
    )[0]


def _load(collection_name: str) -> None:
    connect()
    try:
        client.load_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(f"Milvus load_collection failed for {collection_name}: {exc}") from exc


def search_docs_auto(
    query: str,
    top_k: int,
    output_fields: list[str],
) -> tuple[list[dict], dict]:
    """Plan → search → rerank. Used when SEARCH_MODE=auto."""
    plan = pick_search_plan(query)
    _load(COLLECTION_NAME)
    plan = downgrade_plan_for_collection(
        plan,
        has_bm25=collection_has_bm25(COLLECTION_NAME),
        has_release_fields=collection_has_release_fields(COLLECTION_NAME),
    )

    fetch_limit = plan.candidate_depth or top_k
    filter_expr = plan.filter_expr
    filter_fallback = False
    embedding = None
    if plan.retrieval_mode in ("dense", "hybrid"):
        embedding = _require_embedding(query)

    try:
        if plan.retrieval_mode == "bm25":
            hits = bm25_search(
                COLLECTION_NAME, query, fetch_limit, output_fields, filter_expr=filter_expr
            )
            if not hits and filter_expr:
                hits = bm25_search(
                    COLLECTION_NAME, query, fetch_limit, output_fields, filter_expr=""
                )
                filter_fallback = True
        elif plan.retrieval_mode == "hybrid":
            hits = hybrid_search(
                COLLECTION_NAME,
                query,
                embedding,
                top_k,
                output_fields,
                filter_expr=filter_expr,
                candidate_depth=plan.candidate_depth,
            )
        else:
            hits = dense_search(
                COLLECTION_NAME, embedding, top_k, output_fields, filter_expr=filter_expr
            )
    except Exception as exc:
        kind = "hybrid_search" if plan.retrieval_mode == "hybrid" else "search"
        raise RuntimeError(f"Milvus {kind} failed for {COLLECTION_NAME}: {exc}") from exc

    if plan.retrieval_mode == "bm25":
        hits = rerank_hits_after_search(plan, hits, query, top_k)
        meta = retrieval_metadata(
            plan,
            candidate_depth=fetch_limit,
            filter_expr=filter_expr or None,
            filter_fallback=filter_fallback,
        )
        return hits, meta
    if plan.retrieval_mode == "hybrid":
        meta = retrieval_metadata(
            plan,
            candidate_depth=plan.candidate_depth or top_k,
            filter_expr=filter_expr or None,
        )
        return hits, meta
    return hits, retrieval_metadata(plan)


def search_collection(
    collection_name: str,
    query: str,
    top_k: int,
    output_fields: list[str],
    filter_expr: str = "",
) -> list[dict]:
    """Embed the query and search (dense, or hybrid when that collection's mode is hybrid)."""
    mode = (
        "hybrid"
        if _search_mode_for(collection_name) == "hybrid" and collection_has_bm25(collection_name)
        else "dense"
    )
    _load(collection_name)
    embedding = _require_embedding(query)
    use_hybrid = mode == "hybrid"
    try:
        if use_hybrid:
            hits = hybrid_search(
                collection_name,
                query,
                embedding,
                top_k,
                output_fields,
                filter_expr=filter_expr,
            )
        else:
            hits = dense_search(
                collection_name, embedding, top_k, output_fields, filter_expr=filter_expr
            )
    except Exception as exc:
        kind = "hybrid_search" if use_hybrid else "search"
        raise RuntimeError(f"Milvus {kind} failed for {collection_name}: {exc}") from exc
    return hits
