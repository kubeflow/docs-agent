"""Pick a docs search plan from the query text.

This is regex matching in the MCP server — not an LLM parameter.
Rerank helpers run only after Milvus returns hits.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace

from rag_collections import SPARSE_FIELD

# ConfigMap still uses these env names. candidate_depth = fetch this many, then rerank to top_k.
LATEST_RELEASE_FETCH_LIMIT = int(os.getenv("AUTO_TEMPORAL_CANDIDATE_DEPTH", "50"))
KEYWORD_FETCH_LIMIT = int(os.getenv("AUTO_BM25_CANDIDATE_DEPTH", "30"))

COMPARE_WORDS = re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.I)
RELEASE_DATE_WORDS = re.compile(r"\b(when was|release date|ga date|released)\b", re.I)
LATEST_OR_CURRENT_WORDS = re.compile(
    r"\b(latest|current|newest|most recent|supported|(?:new|recent|upcoming|next|last) release)\b",
    re.I,
)
HOW_WHY_EXPLAIN_WORDS = re.compile(r"\b(how|why|explain|overview|architecture)\b", re.I)
VERSION_NUMBER = re.compile(
    r"\b(?:v?\d+\.\d+(?:\.\d+)?(?:\.\d+)?|v1beta\d+|v1alpha\d+)\b",
    re.I,
)
DOTTED_CONFIG_KEY = re.compile(r"\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,}\b", re.I)
ERROR_WORDS = re.compile(
    r"\b(?:error|exception|errno|failed|failure|timeout|crashloop|oom)\b",
    re.I,
)


@dataclass(frozen=True)
class RetrievalPlan:
    """How to search this query.

    candidate_depth: fetch this many hits, then rerank/trim down to top_k.
    """

    intent: str
    retrieval_mode: str
    reason: str
    candidate_depth: int | None = None
    filter_expr: str = ""
    rerank_by_release_date: bool = False
    prefer_version_match: bool = False
    # Also run an unfiltered hybrid search and keep some of its hits. A "latest" question
    # is often about a component ("is X in the latest release?"), and the release notes
    # alone do not answer it.
    add_general_hits: bool = False


def downgrade_plan_for_collection(
    plan: RetrievalPlan,
    *,
    has_bm25: bool,
    has_release_fields: bool,
) -> RetrievalPlan:
    """If this collection cannot do the planned search, fall back.

    has_bm25: collection has sparse_vector. Without it, BM25/hybrid become dense.
    has_release_fields: collection has release_date. Without it, skip the
    release filter and post-search date/version rerank.
    """
    if not has_bm25:
        if plan.retrieval_mode in ("hybrid", "bm25"):
            return replace(
                plan,
                retrieval_mode="dense",
                reason=f"{plan.reason}; dense fallback (collection lacks {SPARSE_FIELD})",
                filter_expr="",
                rerank_by_release_date=False,
                prefer_version_match=False,
                candidate_depth=None,
            )
        return plan

    if not has_release_fields:
        return replace(
            plan,
            filter_expr="",
            rerank_by_release_date=False,
            prefer_version_match=False,
            add_general_hits=False,
        )

    return plan


def pick_search_plan(query: str) -> RetrievalPlan:
    """First matching branch wins. Same rules as before."""
    versions = VERSION_NUMBER.findall(query)

    # Comparing two versions, or "compare / vs".
    if COMPARE_WORDS.search(query) or len(versions) >= 2:
        return RetrievalPlan(
            intent="comparison",
            retrieval_mode="hybrid",
            reason="comparison query (two versions or compare/versus phrasing)",
            candidate_depth=KEYWORD_FETCH_LIMIT,
        )

    # "When was X released?" — keyword search, then keep version-matching hits.
    if RELEASE_DATE_WORDS.search(query):
        return RetrievalPlan(
            intent="release_date",
            retrieval_mode="bm25",
            reason="release-date question",
            candidate_depth=LATEST_RELEASE_FETCH_LIMIT,
            prefer_version_match=True,
        )

    # "Latest / current / newest" — keyword search release notes, then sort by date.
    if LATEST_OR_CURRENT_WORDS.search(query):
        return RetrievalPlan(
            intent="temporal",
            retrieval_mode="bm25",
            reason="temporal/latest-current query",
            candidate_depth=LATEST_RELEASE_FETCH_LIMIT,
            filter_expr='doc_type == "release"',
            rerank_by_release_date=True,
            add_general_hits=True,
        )

    # Exact version, dotted config key, or error wording — keyword search.
    if VERSION_NUMBER.search(query) or DOTTED_CONFIG_KEY.search(query) or ERROR_WORDS.search(query):
        return RetrievalPlan(
            intent="exact",
            retrieval_mode="bm25",
            reason="exact version, config key, or error term",
            candidate_depth=KEYWORD_FETCH_LIMIT,
        )

    # How / why / explain — meaning + keywords together.
    if HOW_WHY_EXPLAIN_WORDS.search(query):
        return RetrievalPlan(
            intent="conceptual",
            retrieval_mode="hybrid",
            reason="conceptual/explanatory query",
        )

    return RetrievalPlan(
        intent="general",
        retrieval_mode="hybrid",
        reason="default hybrid for general docs query",
    )


def boost_release_docs(hits: list[dict]) -> list[dict]:
    """Move release-note chunks ahead of other hits. Runs after search."""
    release_hits = []
    other_hits = []
    for hit in hits:
        if hit.get("entity", {}).get("doc_type") == "release":
            release_hits.append(hit)
        else:
            other_hits.append(hit)
    if not release_hits:
        return hits
    return release_hits + other_hits


def rerank_by_release_date(hits: list[dict], top_k: int) -> list[dict]:
    """Newest release_date first; tie-break by search score. Runs after search."""
    dated: list[dict] = []
    undated: list[dict] = []
    for hit in hits:
        release_date = hit.get("entity", {}).get("release_date")
        if release_date is not None:
            dated.append(hit)
        else:
            undated.append(hit)

    if not dated:
        return hits[:top_k]

    dated.sort(
        key=lambda hit: (
            -int(hit["entity"]["release_date"]),
            -float(hit.get("distance", 0.0)),
        )
    )
    return (dated + undated)[:top_k]


def entity_matches_version(entity: dict, versions: list[str]) -> bool:
    entity_version = str(entity.get("version", "")).lower()
    content = str(entity.get("content_text", "")).lower()
    file_path = str(entity.get("file_path", "")).lower()
    for version in versions:
        normalized = version.lower().lstrip("v")
        if (
            normalized in entity_version
            or normalized in content
            or normalized in file_path
            or version.lower() in content
        ):
            return True
    return False


def rerank_for_version_match(hits: list[dict], query: str, top_k: int) -> list[dict]:
    """Keep chunks that mention the version in the query. Runs after search."""
    versions = VERSION_NUMBER.findall(query)
    if not versions:
        return hits[:top_k]

    matched = [hit for hit in hits if entity_matches_version(hit.get("entity", {}), versions)]
    if matched:
        return matched[:top_k]
    return hits[:top_k]


def rerank_hits_after_search(
    plan: RetrievalPlan,
    hits: list[dict],
    query: str,
    top_k: int,
) -> list[dict]:
    """Reorder or trim Milvus hits. Call this only after search returns."""
    if plan.intent == "temporal":
        hits = boost_release_docs(hits)
    if plan.rerank_by_release_date:
        return rerank_by_release_date(hits, top_k)
    if plan.prefer_version_match:
        return rerank_for_version_match(hits, query, top_k)
    return hits[:top_k]


def merge_release_and_general_hits(
    release_hits: list[dict],
    general_hits: list[dict],
    top_k: int,
) -> list[dict]:
    """Release notes first, then general hits, sharing top_k.

    General hits get up to half the slots (rounded down), so the newest release notes
    still lead. Either side fills the other's unused slots.
    """

    def key(hit: dict):
        entity = hit.get("entity", {})
        return hit.get("id") or (entity.get("file_path"), entity.get("content_text"))

    seen = {key(hit) for hit in release_hits}
    general = [hit for hit in general_hits if key(hit) not in seen]
    general_slots = min(len(general), top_k // 2)
    release_slots = min(len(release_hits), top_k - general_slots)
    general_slots = min(len(general), top_k - release_slots)
    return release_hits[:release_slots] + general[:general_slots]


def retrieval_metadata(plan: RetrievalPlan, **extra: object) -> dict:
    meta = {
        "retrieval_mode": plan.retrieval_mode,
        "intent": plan.intent,
        "reason": plan.reason,
    }
    meta.update(extra)
    return meta


# ConfigMap env names (same values as the fetch limits above).
AUTO_TEMPORAL_CANDIDATE_DEPTH = LATEST_RELEASE_FETCH_LIMIT
AUTO_BM25_CANDIDATE_DEPTH = KEYWORD_FETCH_LIMIT
