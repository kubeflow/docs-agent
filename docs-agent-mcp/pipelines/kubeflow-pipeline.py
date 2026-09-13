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
    packages_to_install=["requests", "beautifulsoup4"]
)
def download_github_directory(
    repo_owner: str,
    repo_name: str,
    directory_path: str,
    github_token: str,
    github_data: dsl.Output[dsl.Dataset]
):
    import os
    import requests
    import json
    import base64
    import time
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

    def api_request(url, params=None):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers)

                if resp.status_code == 403:
                    remaining = resp.headers.get("X-RateLimit-Remaining", "0")
                    if remaining == "0":
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                        wait_time = max(reset_time - int(time.time()), 60)
                        print(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(min(wait_time, 300))
                        continue
                    print(f"Forbidden (403) for {url}: {resp.text[:200]}")

                if resp.status_code == 200:
                    return resp.json()

                print(f"API error: HTTP {resp.status_code} for {url}")
                return None

            except Exception as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

        return None

    def get_files_recursive(url):
        files = []
        items = api_request(url)
        if not items or not isinstance(items, list):
            return files

        for item in items:
            if item['type'] == 'file' and (item['name'].endswith('.md') or item['name'].endswith('.html')):
                file_data = api_request(item['url'])
                if not file_data or 'content' not in file_data:
                    print(f"Skipping unreadable file: {item['path']}")
                    continue
                content = base64.b64decode(file_data['content']).decode('utf-8')

                if item['name'].endswith('.html'):
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text(separator=' ', strip=True)

                files.append({
                    'path': item['path'],
                    'content': content,
                    'file_name': item['name']
                })
            elif item['type'] == 'dir':
                files.extend(get_files_recursive(item['url']))

        return files

    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}"
    files = get_files_recursive(api_url)
    print(f"Downloaded {len(files)} files")

    with open(github_data.path, 'w', encoding='utf-8') as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + '\n')


@dsl.component(base_image=INGEST_IMAGE)
def chunk_and_embed(
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

    from milvus_store import chunk_github_jsonl, embed_chunk_records

    print(f"Using embeddings service: {embeddings_service_url}")
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
def store_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    clean_rebuild: bool,
    clean_rebuild_confirmation: str,
    maintenance_lock_token: str,
):
    from milvus_store import store_embedded_records

    store_embedded_records(
        embedded_data.path,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        clean_rebuild=clean_rebuild,
        clean_rebuild_confirmation=clean_rebuild_confirmation,
        maintenance_lock_token=maintenance_lock_token,
    )


@dsl.pipeline(
    name="github-rag",
    description="RAG pipeline for processing GitHub documentation"
)
def github_rag_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website", 
    directory_path: str = "content/en/docs",
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
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
        github_token=github_token
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