import os
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

try:
    import kfp.kubernetes as k8s
except ImportError:  # pragma: no cover - optional at compile time
    k8s = None

from utils import DEFAULT_EMBEDDING_BATCH_SIZE, DOCS_COLLECTION

from milvus_store import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
)

INGEST_IMAGE = os.getenv(
    "DOCS_INGEST_IMAGE",
    "ghcr.io/kubeflow/docs-rag-ingest:v0.1.0",
)

@dsl.component(
    base_image="docker.io/library/python:3.11-slim",
    packages_to_install=["requests==2.33.0", "beautifulsoup4==4.15.0"],
)
def download_github_directory(
    repo_owner: str,
    repo_name: str,
    directory_path: str,
    exclude_paths: str,
    github_token: str,
    github_data: dsl.Output[dsl.Dataset],
):
    """Download every .md/.html file under directory_path at the default branch head.

    One tree listing, then one raw download per file. Any file that cannot be read fails
    the step: a partial download must not reach the store step, which removes documents
    that are missing from the run.
    """
    import json
    import os
    import time

    import requests
    from bs4 import BeautifulSoup

    token = ""
    for candidate in (github_token, os.environ.get("Github_Pat"), os.environ.get("GITHUB_TOKEN")):
        if candidate and str(candidate).strip():
            token = str(candidate).strip()
            break
    if not token:
        print("WARNING: no github_token or Github_Pat; GitHub allows 60 API requests per hour")
    headers = {"Authorization": f"token {token}"} if token else {}

    def get(url, **kwargs):
        last_error = ""
        for attempt in range(4):
            try:
                resp = requests.get(url, headers=headers, timeout=60, **kwargs)
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return resp
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (403, 429) and resp.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(resp.headers.get("X-RateLimit-Reset", "0") or 0)
                wait = min(max(reset - int(time.time()), 30), 600)
                print(f"GitHub rate limit reached; waiting {wait}s")
                time.sleep(wait)
            elif resp.status_code >= 500 or resp.status_code == 429:
                time.sleep(2**attempt)
            else:
                break
        raise RuntimeError(f"GET {url} failed: {last_error}")

    repo_api = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    branch = get(repo_api).json()["default_branch"]
    commit_sha = get(f"{repo_api}/commits/{branch}").json()["sha"]
    tree = get(f"{repo_api}/git/trees/{commit_sha}", params={"recursive": "1"}).json()
    if tree.get("truncated"):
        raise RuntimeError(f"Tree listing of {repo_owner}/{repo_name} is truncated; narrow directory_path")

    prefix = directory_path.strip("/")
    prefix = f"{prefix}/" if prefix else ""
    excluded = tuple(part.strip().strip("/") for part in exclude_paths.split(",") if part.strip())

    def wanted(path):
        if not path.startswith(prefix) or not path.endswith((".md", ".html")):
            return False
        return not any(path == ex or path.startswith(f"{ex}/") for ex in excluded)

    paths = sorted(item["path"] for item in tree["tree"] if item["type"] == "blob" and wanted(item["path"]))
    if not paths:
        raise RuntimeError(f"No .md/.html files under '{prefix}' in {repo_owner}/{repo_name}@{commit_sha[:7]}")
    print(f"{repo_owner}/{repo_name}@{commit_sha[:7]} ({branch}): {len(paths)} files under '{prefix}'")

    with open(github_data.path, "w", encoding="utf-8") as out:
        for path in paths:
            raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{commit_sha}/{path}"
            content = get(raw_url).content.decode("utf-8", errors="replace")
            if path.endswith(".html"):
                content = BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)
            record = {
                "path": path,
                "content": content,
                "file_name": path.rsplit("/", 1)[-1],
                "commit_sha": commit_sha,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Downloaded {len(paths)} files")


@dsl.component(base_image=INGEST_IMAGE)
def chunk_and_embed(
    github_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    base_url: str,
    target_tokens: int,
    overlap_tokens: int,
    doc_type: str,
    embeddings_service_url: str,
    embedding_batch_size: int,
    embedded_data: dsl.Output[dsl.Dataset],
):
    import json

    from milvus_store import chunk_github_jsonl, embed_chunk_records

    print(f"Using embeddings service: {embeddings_service_url}")
    records = chunk_github_jsonl(
        github_data.path,
        repo_name=repo_name,
        base_url=base_url,
        target_tokens=int(target_tokens),
        overlap_tokens=int(overlap_tokens),
        doc_type=doc_type,
    )
    records = embed_chunk_records(
        records,
        embeddings_service_url=embeddings_service_url,
        embedding_batch_size=embedding_batch_size,
    )
    print(f"Embedded {len(records)} chunks")
    with open(embedded_data.path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@dsl.component(base_image=INGEST_IMAGE)
def store_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    clean_rebuild: bool,
    clean_rebuild_confirmation: str,
    maintenance_lock_token: str,
    prune_removed: bool,
    repo_name: str,
    directory_path: str,
):
    from milvus_store import store_embedded_records

    # document_id is "<repo_name>:<file path>"; exclude_paths under it are pruned too.
    directory = directory_path.strip("/")
    prune_prefix = f"{repo_name}:{directory}/" if directory else f"{repo_name}:"

    store_embedded_records(
        embedded_data.path,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        clean_rebuild=clean_rebuild,
        clean_rebuild_confirmation=clean_rebuild_confirmation,
        maintenance_lock_token=maintenance_lock_token,
        prune_prefix=prune_prefix if prune_removed else "",
    )


@dsl.pipeline(
    name="github-rag",
    description="RAG pipeline for processing GitHub documentation"
)
def github_rag_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website", 
    directory_path: str = "content/en/docs",
    # Comma-separated paths (files or directories) to skip under directory_path.
    exclude_paths: str = "",
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
    # Empty = infer from the Hugo path (documentation, release, nav, redirect). Set it for
    # other sources, e.g. "distribution" for kubeflow/community-distribution.
    doc_type: str = "",
    # Remove indexed pages of this source that are gone upstream. Needs a full download
    # of the source, which is what this pipeline does.
    prune_removed: bool = True,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    embeddings_service_url: str = (
        "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed"
    ),
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    milvus_host: str = "milvus-milvus.ml-infra.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = DOCS_COLLECTION,
    clean_rebuild: bool = False,
    clean_rebuild_confirmation: str = "",
    maintenance_lock_token: str = "",
):
    # Download GitHub directory
    download_task = download_github_directory(
        repo_owner=repo_owner,
        repo_name=repo_name,
        directory_path=directory_path,
        exclude_paths=exclude_paths,
        github_token=github_token,
    )

    if k8s is not None:
        k8s.use_secret_as_env(
            download_task,
            secret_name="github-pat",
            secret_key_to_env={"Github_Pat": "Github_Pat"},
        )
    
    # Chunk and embed the content
    chunk_task = chunk_and_embed(
        github_data=download_task.outputs["github_data"],
        repo_name=repo_name,
        base_url=base_url,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        doc_type=doc_type,
        embeddings_service_url=embeddings_service_url,
        embedding_batch_size=embedding_batch_size,
    )
    
    # Store in Milvus
    store_task = store_milvus(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        clean_rebuild=clean_rebuild,
        clean_rebuild_confirmation=clean_rebuild_confirmation,
        maintenance_lock_token=maintenance_lock_token,
        prune_removed=prune_removed,
        repo_name=repo_name,
        directory_path=directory_path,
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
        k8s.use_secret_as_env(
            store_task,
            secret_name="milvus-maintenance-lock",
            secret_key_to_env={
                "MILVUS_MAINTENANCE_LOCK": "MILVUS_MAINTENANCE_LOCK",
            },
        )


if __name__ == "__main__":
    import os
    # Set environment variable to disable caching by default
    os.environ['KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT'] = 'true'
    
    # Compile the pipeline with caching disabled by default
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_pipeline,
        package_path="github_rag_pipeline.yaml"
    )