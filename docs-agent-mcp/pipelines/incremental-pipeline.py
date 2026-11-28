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
    base_image="docker.io/library/python:3.9",
    packages_to_install=["requests", "beautifulsoup4"],
)
def download_specific_files(
    repo_owner: str,
    repo_name: str,
    file_paths: str,  # JSON string of file paths list
    github_token: str,
    github_data: dsl.Output[dsl.Dataset],
):
    import base64
    import json
    import os
    import requests
    from bs4 import BeautifulSoup

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

    try:
        file_paths_list = json.loads(file_paths)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file_paths: {file_paths}")
        file_paths_list = []

    print(f"Processing {len(file_paths_list)} changed files")

    files = []

    for file_path in file_paths_list:
        if not (file_path.endswith(".md") or file_path.endswith(".html")):
            print(f"Skipping non-doc file: {file_path}")
            continue

        try:
            api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            file_data = response.json()

            if not file_data or "content" not in file_data:
                print(f"Skipping unreadable file: {file_path}")
                continue

            content = base64.b64decode(file_data["content"]).decode("utf-8")

            if file_path.endswith(".html"):
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text(separator=" ", strip=True)

            files.append(
                {
                    "path": file_path,
                    "content": content,
                    "file_name": file_data["name"],
                }
            )
            print(f"Downloaded: {file_path}")

        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
            continue

    print(f"Successfully downloaded {len(files)} files")

    with open(github_data.path, "w", encoding="utf-8") as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + "\n")


@dsl.component(base_image=INGEST_IMAGE)
def delete_old_vectors(
    file_paths: str,  # JSON string of file paths list
    repo_name: str,
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
):
    import json
    import os
    from pymilvus import MilvusClient
    from milvus_store import (
        DELETE_BATCH_SIZE,
        DOCUMENT_ID_FIELD,
        check_milvus_health,
        milvus_uri,
    )

    milvus_user = os.environ.get("MILVUS_USER", "root")
    milvus_password = os.environ.get("MILVUS_PASSWORD", "")
    if not milvus_password:
        raise RuntimeError("MILVUS_PASSWORD must be set via pipeline secret (not in source code)")

    try:
        file_paths_list = json.loads(file_paths)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file_paths: {file_paths}")
        return

    if not file_paths_list:
        print("No file paths provided for deletion.")
        return

    milvus = MilvusClient(
        uri=milvus_uri(milvus_host, milvus_port),
        user=milvus_user,
        password=milvus_password,
    )
    check_milvus_health(milvus)

    if not milvus.has_collection(collection_name):
        print(f"Collection {collection_name} does not exist yet; skipping deletion.")
        return

    doc_ids = [f"{repo_name}:{fp.strip()}" for fp in file_paths_list if fp and fp.strip()]
    if not doc_ids:
        return

    deleted = 0
    for start in range(0, len(doc_ids), DELETE_BATCH_SIZE):
        batch = doc_ids[start : start + DELETE_BATCH_SIZE]
        quoted = ", ".join(f'"{doc_id}"' for doc_id in batch)
        result = milvus.delete(
            collection_name=collection_name,
            filter=f"{DOCUMENT_ID_FIELD} in [{quoted}]",
        )
        deleted += int(result.get("delete_count", 0) or 0)

    if deleted:
        milvus.flush(collection_name)
        milvus.load_collection(collection_name)
        print(f"Deleted {deleted} existing chunks for {len(doc_ids)} documents from {collection_name}")
    else:
        print(f"No existing chunks found for {len(doc_ids)} documents in {collection_name}")


@dsl.component(base_image=INGEST_IMAGE)
def chunk_and_embed_incremental(
    github_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    base_url: str,
    target_tokens: int,
    overlap_tokens: int,
    embeddings_service_url: str,
    embedding_batch_size: int,
    embedded_data: dsl.Output[dsl.Dataset],
):
    import json
    import os
    from milvus_store import chunk_github_jsonl, embed_chunk_records

    print(f"Using embeddings service: {embeddings_service_url}")

    if not os.path.exists(github_data.path) or os.path.getsize(github_data.path) == 0:
        print("No changed files to chunk/embed.")
        with open(embedded_data.path, "w", encoding="utf-8") as handle:
            pass
        return

    records = chunk_github_jsonl(
        github_data.path,
        repo_name=repo_name,
        base_url=base_url,
        target_tokens=int(target_tokens),
        overlap_tokens=int(overlap_tokens),
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
def store_milvus_incremental(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
):
    import os
    from milvus_store import store_embedded_records

    if not os.path.exists(embedded_data.path) or os.path.getsize(embedded_data.path) == 0:
        print("No records to insert for this incremental run.")
        return

    store_embedded_records(
        embedded_data.path,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        clean_rebuild=False,
        clean_rebuild_confirmation="",
        maintenance_lock_token="",
    )


@dsl.pipeline(
    name="github-rag-incremental-build",
    description="Incremental RAG pipeline for processing only changed GitHub files",
)
def github_rag_incremental_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website",
    changed_files: str = "[]",  # JSON string of changed file paths
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    embeddings_service_url: str = ("http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed"),
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    milvus_host: str = "milvus-milvus.ml-infra.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = DOCS_COLLECTION,
):
    # Step 1: Delete old vectors for changed/deleted files
    delete_task = delete_old_vectors(
        file_paths=changed_files,
        repo_name=repo_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
    )

    if k8s is not None:
        k8s.use_secret_as_env(
            delete_task,
            secret_name="milvus-auth",
            secret_key_to_env={
                "MILVUS_USER": "MILVUS_USER",
                "MILVUS_PASSWORD": "MILVUS_PASSWORD",
            },
        )

    # Step 2: Download only the changed files
    download_task = download_specific_files(
        repo_owner=repo_owner,
        repo_name=repo_name,
        file_paths=changed_files,
        github_token=github_token,
    )

    # Step 3: Chunk and embed the changed files
    chunk_task = chunk_and_embed_incremental(
        github_data=download_task.outputs["github_data"],
        repo_name=repo_name,
        base_url=base_url,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        embeddings_service_url=embeddings_service_url,
        embedding_batch_size=embedding_batch_size,
    )

    # Step 4: Store new vectors in Milvus (after deletion is complete)
    store_task = store_milvus_incremental(
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

    # Ensure deletion happens before insertion
    store_task.after(delete_task)


if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_incremental_pipeline,
        package_path="github_rag_incremental_pipeline.yaml",
    )
