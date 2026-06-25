"""KFP pipeline for ingesting GitHub issues into Milvus for RAG.

Wires the existing download_github_issues component into a complete pipeline
with issues-aware chunking and a dedicated issues_rag Milvus collection.

Components are self-contained per KFP convention. The download_github_issues
component is copied from kubeflow-pipeline.py (lines 67-213) since KFP
@dsl.component functions must be self-contained with all imports inside.
"""

import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

try:
    import kfp.kubernetes as k8s
except ImportError:  # pragma: no cover
    k8s = None

from utils import DEFAULT_EMBEDDING_BATCH_SIZE, ISSUES_COLLECTION

@dsl.component(
    base_image="docker.io/library/python:3.9",
    packages_to_install=["requests"]
)
def download_github_issues(
    repos: str,
    labels: str,
    state: str,
    max_issues_per_repo: int,
    github_token: str,
    issues_data: dsl.Output[dsl.Dataset]
):
    """Fetch GitHub issues and comments from multiple repos for RAG indexing.

    Args:
        repos: Comma-separated list of repos (e.g., "kubeflow/kubeflow,kubeflow/pipelines")
        labels: Comma-separated labels to filter (e.g., "kind/bug,kind/question")
        state: Issue state - "open", "closed", or "all"
        max_issues_per_repo: Maximum issues to fetch per repository
        github_token: GitHub personal access token for API authentication
        issues_data: Output dataset path
    """
    import requests
    import json
    import time
    import os

    def resolve_github_token(token):
        for candidate in (token, os.environ.get("Github_Pat"), os.environ.get("GITHUB_TOKEN")):
            if candidate and str(candidate).strip():
                return str(candidate).strip()
        return ""

    github_token = resolve_github_token(github_token)
    if github_token:
        print("Using authenticated GitHub API requests")
    else:
        print("WARNING: No github_token or Github_Pat env set; rate limits will be low (60 req/hr)")

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    all_issues = []

    def api_request(url, params=None):
        """Make GitHub API request with rate limit handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers)

                # Handle rate limiting
                if resp.status_code == 403:
                    remaining = resp.headers.get("X-RateLimit-Remaining", "0")
                    if remaining == "0":
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                        wait_time = max(reset_time - int(time.time()), 60)
                        print(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(min(wait_time, 300))  # Max 5 min wait
                        continue

                if resp.status_code == 200:
                    return resp.json()
                else:
                    print(f"API error: HTTP {resp.status_code}")
                    return None

            except Exception as e:
                print(f"Request failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def fetch_comments(owner, name, issue_number):
        """Fetch all comments for a single issue."""
        comments_url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}/comments"
        comments_text = ""
        page = 1

        while True:
            comments = api_request(comments_url, {"per_page": 100, "page": page})
            if not comments:
                break

            for comment in comments:
                author = comment.get("user", {}).get("login", "unknown")
                created = comment.get("created_at", "")[:10]
                body = comment.get("body", "") or ""
                comments_text += f"\n\n---\n**Comment by @{author}** ({created}):\n{body}"

            if len(comments) < 100:
                break
            page += 1

        return comments_text

    for repo in repos.split(","):
        repo = repo.strip()
        if "/" not in repo:
            print(f"Skipping invalid repo format: {repo}")
            continue

        owner, name = repo.split("/", 1)
        print(f"Fetching issues from {owner}/{name}...")

        page = 1
        repo_issues = []

        while len(repo_issues) < max_issues_per_repo:
            url = f"https://api.github.com/repos/{owner}/{name}/issues"
            params = {
                "state": state,
                "labels": labels,
                "per_page": 100,
                "page": page
            }

            issues = api_request(url, params)
            if not issues:
                break

            for issue in issues:
                if "pull_request" in issue:
                    continue

                labels_str = ", ".join([l["name"] for l in issue.get("labels", [])])
                issue_url = issue.get("html_url", "")
                created_at = issue.get("created_at", "")[:10]
                updated_at = issue.get("updated_at", "")[:10]

                # Build issue content with full metadata
                content = f"# {issue['title']}\n\n"
                content += f"**Repository:** {repo}\n"
                content += f"**Issue:** #{issue['number']}\n"
                content += f"**URL:** {issue_url}\n"
                content += f"**Labels:** {labels_str}\n"
                content += f"**State:** {issue['state']}\n"
                content += f"**Created:** {created_at}\n"
                content += f"**Updated:** {updated_at}\n\n"
                content += issue.get("body", "") or ""

                # Fetch and append comments
                if issue.get("comments", 0) > 0:
                    comments = fetch_comments(owner, name, issue["number"])
                    content += comments

                repo_issues.append({
                    "path": f"issues/{name}/{issue['number']}",
                    "content": content,
                    "file_name": f"issue-{name}-{issue['number']}.md",
                    "url": issue_url
                })

                if len(repo_issues) >= max_issues_per_repo:
                    break

            page += 1

        all_issues.extend(repo_issues)
        print(f"  Fetched {len(repo_issues)} issues from {repo}")

    print(f"Total issues fetched: {len(all_issues)}")

    with open(issues_data.path, 'w', encoding='utf-8') as f:
        for issue_data in all_issues:
            f.write(json.dumps(issue_data, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "langchain-text-splitters"],
)
def chunk_and_embed_issues(
    issues_data: dsl.Input[dsl.Dataset],
    chunk_size: int,
    chunk_overlap: int,
    embeddings_service_url: str,
    embedding_batch_size: int,
    embedded_data: dsl.Output[dsl.Dataset],
):
    """Chunk GitHub issues at comment boundaries and generate embeddings.

    Chunking strategy:
    - Parse metadata (repo, issue number, state, labels, URL) from content
    - Prepend a metadata prefix to every chunk for self-contained retrieval
    - Split at comment boundaries (--- separators) first
    - Subdivide oversized segments with RecursiveCharacterTextSplitter
    - Short issues (body + comments < chunk_size) become a single chunk
    """
    import json
    import re
    import requests
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"Using embeddings service: {embeddings_service_url}")
    embedding_batch_size = max(1, int(embedding_batch_size))

    def parse_metadata(content):
        """Extract structured metadata from issue content."""
        title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        repo_match = re.search(r'\*\*Repository:\*\*\s*(.+)', content)
        number_match = re.search(r'\*\*Issue:\*\*\s*#(\d+)', content)
        url_match = re.search(r'\*\*URL:\*\*\s*(.+)', content)
        labels_match = re.search(r'\*\*Labels:\*\*[ \t]*(.*)', content)
        state_match = re.search(r'\*\*State:\*\*\s*(\w+)', content)
        return {
            "title": title_match.group(1).strip() if title_match else "",
            "repo_name": repo_match.group(1).strip() if repo_match else "",
            "issue_number": int(number_match.group(1)) if number_match else 0,
            "issue_state": state_match.group(1).strip() if state_match else "",
            "issue_labels": labels_match.group(1).strip() if labels_match else "",
            "citation_url": url_match.group(1).strip() if url_match else "",
        }

    def build_prefix(meta):
        """Build metadata prefix for each chunk."""
        parts = [f"[Issue #{meta['issue_number']}] {meta['title']}"]
        if meta["repo_name"]:
            parts.append(f"Repo: {meta['repo_name']}")
        if meta["issue_state"]:
            parts.append(f"State: {meta['issue_state']}")
        if meta["issue_labels"]:
            parts.append(f"Labels: {meta['issue_labels']}")
        return " | ".join(parts) + "\n\n"

    records = []
    with open(issues_data.path, 'r', encoding='utf-8') as f:
        for line in f:
            issue = json.loads(line)
            content = issue["content"]
            metadata = parse_metadata(content)
            prefix = build_prefix(metadata)
            effective_size = chunk_size - len(prefix)
            if effective_size < 100:
                effective_size = 100

            # Split oversized segments with room left for metadata prefix (matches issues_utils)
            overlap = min(chunk_overlap, max(0, effective_size - 1))
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=effective_size,
                chunk_overlap=overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            # Split at comment boundaries or keep as single chunk
            if len(content) <= effective_size:
                chunks = [prefix + content]
            else:
                segments = re.split(r'\n\n---\n', content)
                chunks = []
                for segment in segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    if len(segment) <= effective_size:
                        chunks.append(prefix + segment)
                    else:
                        for sub in splitter.split_text(segment):
                            chunks.append(prefix + sub)
                if not chunks:
                    chunks = [prefix + content[:effective_size]]

            for chunk_idx, chunk_text in enumerate(chunks):
                records.append({
                    "file_unique_id": f"{metadata['repo_name']}:issues/{metadata['issue_number']}",
                    "repo_name": metadata["repo_name"],
                    "issue_number": metadata["issue_number"],
                    "issue_state": metadata["issue_state"],
                    "issue_labels": metadata["issue_labels"],
                    "citation_url": metadata["citation_url"],
                    "chunk_index": chunk_idx,
                    "content_text": chunk_text[:2000],
                    "source_type": "issue",
                })

            print(f"Issue #{metadata['issue_number']}: {len(chunks)} chunks")

    print(f"Total chunks: {len(records)}; requesting embeddings from TEI service...")

    max_tei_chars = 1000
    for i in range(0, len(records), embedding_batch_size):
        batch = records[i:i + embedding_batch_size]
        texts = [r["content_text"][:max_tei_chars] for r in batch]
        response = requests.post(
            embeddings_service_url,
            json={"inputs": texts},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        vectors = response.json()
        for idx, vector in enumerate(vectors):
            batch[idx]["embedding"] = vector

    with open(embedded_data.path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="docker.io/library/python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_issues_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str
):
    """Store issue embeddings in a dedicated Milvus collection.

    Creates the issues_rag collection with issue-specific schema fields
    (issue_number, issue_state, issue_labels, source_type).
    """
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    import json
    import os
    import re
    from datetime import datetime

    SCHEMA_VERSION = 1
    SCHEMA_DESCRIPTION = f"RAG collection for GitHub issues (v={SCHEMA_VERSION})"
    DELETE_BATCH_SIZE = 100

    milvus_user = os.environ.get("MILVUS_USER", "root")
    milvus_password = os.environ.get("MILVUS_PASSWORD", "")
    if not milvus_password:
        raise RuntimeError("MILVUS_PASSWORD must be set via pipeline secret (not in source code)")

    connections.connect(
        "default",
        host=milvus_host,
        port=milvus_port,
        user=milvus_user,
        password=milvus_password,
    )

    fields = [
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

    schema = CollectionSchema(fields, SCHEMA_DESCRIPTION)
    collection_existed = utility.has_collection(collection_name)

    if collection_existed:
        collection = Collection(collection_name)
        existing_desc = collection.description or ""
        version_match = re.search(r'\bv=(\d+)\b', existing_desc)
        existing_version = int(version_match.group(1)) if version_match else None
        if existing_version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version mismatch for {collection_name}. "
                f"Expected v={SCHEMA_VERSION}, found v={existing_version} (description: '{existing_desc}'). "
                f"Run a migration job to drop+recreate before re-indexing."
            )
        print(f"Using existing collection: {collection_name} (schema v={SCHEMA_VERSION})")
    else:
        collection = Collection(collection_name, schema)
        print(f"Created collection: {collection_name} (schema v={SCHEMA_VERSION})")

    records = []
    timestamp = int(datetime.now().timestamp())

    with open(embedded_data.path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "file_unique_id": record["file_unique_id"],
                "repo_name": record["repo_name"],
                "issue_number": record["issue_number"],
                "issue_state": record["issue_state"],
                "issue_labels": record["issue_labels"],
                "citation_url": record["citation_url"],
                "chunk_index": record["chunk_index"],
                "content_text": record["content_text"],
                "vector": record["embedding"],
                "last_updated": timestamp,
                "source_type": record["source_type"],
            })

    if records:
        # Delete existing chunks only when collection pre-existed and has an index loadable
        if collection_existed and len(collection.indexes) > 0:
            collection.load()
            unique_ids = sorted(set(r["file_unique_id"] for r in records))
            current_batch_ids = None
            batches_deleted = 0
            try:
                for i in range(0, len(unique_ids), DELETE_BATCH_SIZE):
                    current_batch_ids = unique_ids[i:i + DELETE_BATCH_SIZE]
                    quoted = ", ".join(f'"{uid}"' for uid in current_batch_ids)
                    expr = f"file_unique_id in [{quoted}]"
                    collection.delete(expr)
                    batches_deleted += 1
                collection.flush()
                print(f"Deleted old chunks for {len(unique_ids)} issues across {batches_deleted} batches")
            except Exception as e:
                print(f"ERROR during delete phase: {e}")
                print(f"Failed batch unique_ids ({len(current_batch_ids) if current_batch_ids else 0} ids): {current_batch_ids}")
                print(f"Batches successfully deleted before failure (unflushed): {batches_deleted - 1}")
                raise

        # Insert new chunks (failure-aware: capture failing batch inside the loop)
        batch_size = 1000
        inserted = 0
        current_batch = None
        current_batch_start = 0
        try:
            for i in range(0, len(records), batch_size):
                current_batch_start = i
                current_batch = records[i:i + batch_size]
                collection.insert(current_batch)
                inserted += len(current_batch)
            current_batch = None  # successful insert phase
            collection.flush()
        except Exception as e:
            print(f"ERROR during insert/flush. Inserted={inserted}/{len(records)}. Error: {e}")
            if current_batch is not None:
                failed_ids = sorted(set(r["file_unique_id"] for r in current_batch))
                print(f"Failed insert batch starts at record {current_batch_start}, file_unique_ids: {failed_ids}")
            else:
                print("Failure occurred during flush() after all batches were submitted")
            raise

        # Create index if not already present (post-flush so num_entities reflects total)
        if len(collection.indexes) == 0:
            nlist = max(16, min(1024, collection.num_entities))
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": nlist}
            }
            collection.create_index("vector", index_params, timeout=120)
        collection.load()
        print(f"Inserted {len(records)} records. Total: {collection.num_entities}")


@dsl.pipeline(
    name="github-issues-rag",
    description="RAG pipeline for GitHub issues ingestion"
)
def github_issues_rag_pipeline(
    repos: str = "kubeflow/kubeflow,kubeflow/pipelines,kubeflow/manifests",
    labels: str = "",
    state: str = "all",
    max_issues_per_repo: int = 200,
    github_token: str = "",
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    embeddings_service_url: str = (
        "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed"
    ),
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    milvus_host: str = "milvus-milvus.ml-infra.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = ISSUES_COLLECTION,
):
    download_task = download_github_issues(
        repos=repos,
        labels=labels,
        state=state,
        max_issues_per_repo=max_issues_per_repo,
        github_token=github_token,
    )

    if k8s is not None:
        k8s.use_secret_as_env(
            download_task,
            secret_name="github-pat",
            secret_key_to_env={"Github_Pat": "Github_Pat"},
        )

    chunk_task = chunk_and_embed_issues(
        issues_data=download_task.outputs["issues_data"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings_service_url=embeddings_service_url,
        embedding_batch_size=embedding_batch_size,
    )

    store_task = store_issues_milvus(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
    )

    if k8s is not None:
        k8s.use_secret_as_env(
            store_task,
            secret_name="milvus-auth",
            secret_key_to_env={
                "MILVUS_USER": "MILVUS_USER",
                "MILVUS_PASSWORD": "MILVUS_PASSWORD",
            },
        )


if __name__ == "__main__":
    import os
    os.environ['KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT'] = 'true'
    kfp.compiler.Compiler().compile(
        pipeline_func=github_issues_rag_pipeline,
        package_path="github_issues_rag_pipeline.yaml"
    )
    print("Compiled: github_issues_rag_pipeline.yaml")
