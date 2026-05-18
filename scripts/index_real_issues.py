"""Fetch real GitHub issues, chunk, embed, and index into Milvus.

This script implements the issues ingestion pipeline end-to-end:
  1. Fetch issues from kubeflow repos via GitHub API
  2. Chunk using comment-boundary-aware splitting (issues_utils)
  3. Embed with sentence-transformers/all-mpnet-base-v2
  4. Store into issues_rag Milvus collection

Usage:
    # Port-forward first: kubectl port-forward svc/my-release-milvus -n docs-agent 19530:19530
    python scripts/index_real_issues.py [--max-issues 50] [--repos kubeflow/kubeflow,kubeflow/pipelines]

Set GITHUB_TOKEN env var for higher rate limits.
"""

import argparse
import os
import re
import sys
import time

import requests
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

# Add pipelines to path for issues_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from issues_utils import build_metadata_prefix, parse_issue_metadata, split_issue_into_chunks

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "issues_rag"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_REPOS = [
    "kubeflow/kubeflow",
    "kubeflow/pipelines",
    "kubeflow/manifests",
    "kubeflow/katib",
]

GITHUB_API = "https://api.github.com"


def fetch_issues(repo: str, max_issues: int, token: str | None) -> list[dict]:
    """Fetch issues from a GitHub repo (excludes pull requests)."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    issues = []
    page = 1
    per_page = min(max_issues, 100)

    while len(issues) < max_issues:
        url = f"{GITHUB_API}/repos/{repo}/issues"
        params = {
            "state": "all",
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "direction": "desc",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=30)

        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - int(time.time()), 10)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        for item in batch:
            if "pull_request" in item:
                continue  # Skip PRs
            if len(issues) >= max_issues:
                break
            issues.append(item)

        page += 1

    return issues


def format_issue_content(issue: dict, repo: str, comments: list[dict]) -> str:
    """Format a GitHub issue into the download_github_issues output format."""
    labels = ", ".join(l["name"] for l in issue.get("labels", []))
    state = issue["state"]

    lines = [
        f"# {issue['title']}",
        f"**Repository:** {repo}",
        f"**Issue:** #{issue['number']}",
        f"**State:** {state}",
        f"**Labels:** {labels}",
        f"**URL:** {issue['html_url']}",
        "",
        issue.get("body") or "(no description)",
    ]

    for c in comments:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"**Comment by {c['user']['login']}** ({c['created_at'][:10]}):")
        lines.append(c.get("body") or "")

    return "\n".join(lines)


def fetch_comments(repo: str, issue_number: int, token: str | None) -> list[dict]:
    """Fetch comments for a specific issue."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{GITHUB_API}/repos/{repo}/issues/{issue_number}/comments"
    resp = requests.get(url, headers=headers, params={"per_page": 30}, timeout=30)

    if resp.status_code == 403:
        return []  # Skip on rate limit

    resp.raise_for_status()
    return resp.json()


COLLECTION_FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="issue_number", dtype=DataType.INT64),
    FieldSchema(name="issue_state", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="issue_labels", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="chunk_index", dtype=DataType.INT64),
    FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="last_updated", dtype=DataType.INT64),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
]


def ensure_collection(fresh: bool = False) -> Collection:
    """Create or get the issues_rag collection.

    Args:
        fresh: If True, drop and recreate. If False (default), reuse existing.
    """
    if utility.has_collection(COLLECTION_NAME):
        if fresh:
            utility.drop_collection(COLLECTION_NAME)
            print(f"Dropped existing {COLLECTION_NAME}")
        else:
            print(f"Reusing existing {COLLECTION_NAME} (incremental mode)")
            col = Collection(COLLECTION_NAME)
            # Ensure index exists before loading
            if not col.has_index():
                print("  Creating missing index...")
                col.create_index("vector", {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": min(1024, max(1, col.num_entities))},
                })
            col.load()
            return col

    schema = CollectionSchema(COLLECTION_FIELDS, "RAG collection for GitHub issues")
    collection = Collection(COLLECTION_NAME, schema)
    print(f"Created collection: {COLLECTION_NAME}")
    return collection


def get_indexed_issue_ids(collection: Collection, repo: str) -> set[int]:
    """Return set of issue_numbers already indexed for a given repo."""
    try:
        results = collection.query(
            expr=f'repo_name == "{repo}"',
            output_fields=["issue_number"],
            limit=16384,
        )
        return {r["issue_number"] for r in results}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="Index real GitHub issues into Milvus")
    parser.add_argument("--max-issues", type=int, default=25, help="Max issues per repo")
    parser.add_argument("--repos", type=str, default=",".join(DEFAULT_REPOS))
    parser.add_argument("--fresh", action="store_true",
                        help="Drop and recreate collection (default: incremental)")
    args = parser.parse_args()

    repos = [r.strip() for r in args.repos.split(",")]
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("WARNING: No GITHUB_TOKEN set. Rate limits will be very low (60 req/hr).")

    # Connect to Milvus
    print("Connecting to Milvus...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create or reuse collection
    collection = ensure_collection(fresh=args.fresh)

    timestamp = int(time.time())
    total_chunks = 0

    for repo in repos:
        print(f"\n{'='*60}")
        print(f"Fetching issues from {repo}...")

        # In incremental mode, skip already-indexed issues
        existing_ids = set()
        if not args.fresh:
            existing_ids = get_indexed_issue_ids(collection, repo)
            if existing_ids:
                print(f"  Already indexed: {len(existing_ids)} issues")

        issues = fetch_issues(repo, args.max_issues, token)
        # Filter out already-indexed issues
        issues = [i for i in issues if i["number"] not in existing_ids]
        print(f"  Got {len(issues)} new issues to index")

        records = []
        for issue in issues:
            # Fetch comments
            comments = fetch_comments(repo, issue["number"], token)

            # Format into pipeline output format
            content = format_issue_content(issue, repo, comments)

            # Parse metadata and chunk using pipeline utilities
            metadata = parse_issue_metadata(content)
            prefix = build_metadata_prefix(metadata)
            chunks = split_issue_into_chunks(content, prefix, chunk_size=1500, chunk_overlap=150)

            labels = ", ".join(l["name"] for l in issue.get("labels", []))

            for idx, chunk in enumerate(chunks):
                # Truncate to fit VARCHAR(2000)
                chunk_text = chunk[:2000]
                embedding = model.encode(chunk_text).tolist()

                records.append({
                    "file_unique_id": f"{repo}:issues/{issue['number']}",
                    "repo_name": repo,
                    "issue_number": issue["number"],
                    "issue_state": issue["state"],
                    "issue_labels": labels[:1024],
                    "citation_url": issue["html_url"],
                    "chunk_index": idx,
                    "content_text": chunk_text,
                    "vector": embedding,
                    "last_updated": timestamp,
                    "source_type": "issue",
                })

            total_chunks += len(chunks)
            print(f"  #{issue['number']}: {issue['title'][:60]} ({len(chunks)} chunks, {len(comments)} comments)")

        if records:
            # Insert in batches of 100
            for i in range(0, len(records), 100):
                batch = records[i:i+100]
                collection.insert(batch)
            print(f"  Inserted {len(records)} records from {repo}")

    collection.flush()

    # Create index if needed (fresh mode or first run)
    if not collection.has_index():
        print("\nCreating vector index...")
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": min(1024, max(1, collection.num_entities))},
        }
        collection.create_index("vector", index_params)
    collection.load()

    print(f"\nTotal: {collection.num_entities} chunks from {len(repos)} repos")

    # Sanity search
    print("\n--- Sanity check: 'pipeline timeout error' ---")
    q_emb = model.encode("pipeline timeout error").tolist()
    results = collection.search(
        data=[q_emb],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["issue_number", "repo_name", "issue_state", "content_text"],
    )
    for hits in results:
        for hit in hits:
            print(f"  #{hit.entity.get('issue_number')} [{hit.entity.get('repo_name')}] "
                  f"(score: {hit.distance:.4f}): {hit.entity.get('content_text', '')[:80]}...")

    print("\nDone! issues_rag is ready for search_github_issues MCP tool.")


if __name__ == "__main__":
    main()
