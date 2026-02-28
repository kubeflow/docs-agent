"""KFP pipeline: Platform Architecture ingestion for docs-agent.

This pipeline ingests infrastructure and deployment documentation from
``kubeflow/manifests`` and platform-specific docs into a **dedicated Milvus
partition** (``platform_arch`` by default).  Using a partition instead of
dropping and recreating the collection (as the main ``kubeflow-pipeline.py``
does — see Issue #10) means the existing docs-RAG data is preserved.

Sources ingested:
  1. ``kubeflow/manifests`` — Kustomize overlays, YAML docs, READMEs
  2. Platform deployment guides (e.g. ``kubeflow/website`` infra sections)

Usage::

    python pipelines/platform-architecture-pipeline.py
    # produces: platform_architecture_pipeline.yaml
"""

import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *


# ====================================================================
# Component 1 — Download manifests / platform docs from GitHub
# ====================================================================


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "beautifulsoup4"],
)
def download_platform_docs(
    repo_owner: str,
    repo_name: str,
    directory_path: str,
    github_token: str,
    file_extensions: str,  # comma-separated, e.g. ".md,.yaml,.yml"
    github_data: dsl.Output[dsl.Dataset],
):
    """Recursively download documentation and manifest files from a GitHub repo."""
    import requests
    import json
    import base64
    from bs4 import BeautifulSoup

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    api_url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        f"/contents/{directory_path}"
    )
    allowed_ext = tuple(ext.strip() for ext in file_extensions.split(","))

    def get_files_recursive(url: str, depth: int = 0, max_depth: int = 8):
        """Walk the GitHub Contents API, collecting matching files."""
        if depth > max_depth:
            return []

        files = []
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            items = response.json()

            for item in items:
                if item["type"] == "file" and item["name"].endswith(allowed_ext):
                    file_resp = requests.get(item["url"], headers=headers, timeout=30)
                    file_resp.raise_for_status()
                    file_data = file_resp.json()
                    content = base64.b64decode(file_data["content"]).decode("utf-8")

                    # Strip HTML if needed
                    if item["name"].endswith(".html"):
                        soup = BeautifulSoup(content, "html.parser")
                        content = soup.get_text(separator=" ", strip=True)

                    files.append(
                        {
                            "path": item["path"],
                            "content": content,
                            "file_name": item["name"],
                        }
                    )
                elif item["type"] == "dir":
                    files.extend(
                        get_files_recursive(item["url"], depth + 1, max_depth)
                    )
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return files

    all_files = get_files_recursive(api_url)
    print(f"Downloaded {len(all_files)} files from {repo_owner}/{repo_name}/{directory_path}")

    with open(github_data.path, "w", encoding="utf-8") as f:
        json.dump(all_files, f, ensure_ascii=False)


# ====================================================================
# Component 2 — Chunk and embed
# ====================================================================


@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "torch",
        "sentence-transformers",
        "langchain-text-splitters",
    ],
)
def chunk_and_embed_platform(
    github_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    base_url: str,
    chunk_size: int,
    chunk_overlap: int,
    embedded_data: dsl.Output[dsl.Dataset],
):
    """Split documents into chunks and create embeddings."""
    import json
    import hashlib

    import torch
    from sentence_transformers import SentenceTransformer
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"CUDA available: {torch.cuda.is_available()}")

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    with open(github_data.path, "r", encoding="utf-8") as f:
        files = json.load(f)

    records = []
    for file_data in files:
        content = file_data.get("content", "")
        if not content.strip():
            continue

        file_unique_id = hashlib.sha256(
            f"{repo_name}/{file_data['path']}".encode()
        ).hexdigest()[:16]

        # Build a citation URL
        citation_url = f"{base_url}/{file_data['path']}"

        chunks = splitter.split_text(content)
        for chunk_idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            records.append(
                {
                    "file_unique_id": file_unique_id,
                    "repo_name": repo_name,
                    "file_path": file_data["path"],
                    "file_name": file_data["file_name"],
                    "citation_url": citation_url[:1024],
                    "chunk_index": chunk_idx,
                    "content_text": chunk[:2000],
                    "embedding": embedding,
                }
            )

    print(f"Created {len(records)} total chunks from {len(files)} files")

    with open(embedded_data.path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ====================================================================
# Component 3 — Store in Milvus using partitions (NOT drop+recreate)
# ====================================================================


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus", "numpy"],
)
def store_milvus_partition(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    partition_name: str,
):
    """Upsert embedded data into a Milvus partition.

    Unlike the main pipeline's ``store_milvus`` component which drops the
    entire collection on every run, this component:

    1. Creates the collection only if it doesn't exist.
    2. Creates / reuses the target partition.
    3. Deletes stale records in the partition (by ``file_unique_id``) before
       inserting fresh ones, achieving an upsert-like workflow.
    """
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        Partition,
        connections,
        utility,
    )
    import json
    from datetime import datetime

    connections.connect("default", host=milvus_host, port=milvus_port)

    # -- Ensure collection exists (shared schema with the docs pipeline) ----
    # NOTE: When multiple pipeline runs target the same partition concurrently,
    # the delete-then-insert below is NOT atomic.  Serialise runs via KFP
    # caching or an external lock if concurrent ingestion is expected.
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="last_updated", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, "RAG collection for Kubeflow documentation")
        collection = Collection(collection_name, schema)
        print(f"Created new collection: {collection_name}")
    else:
        collection = Collection(collection_name)
        print(f"Using existing collection: {collection_name}")

    # -- Ensure partition exists --------------------------------------------
    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name)
        print(f"Created partition: {partition_name}")
    else:
        print(f"Partition already exists: {partition_name}")

    partition = Partition(collection, partition_name)

    # -- Load records -------------------------------------------------------
    records = []
    file_ids_seen = set()
    timestamp = int(datetime.now().timestamp())

    with open(embedded_data.path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            file_ids_seen.add(record["file_unique_id"])
            records.append(
                {
                    "file_unique_id": record["file_unique_id"],
                    "repo_name": record["repo_name"],
                    "file_path": record["file_path"],
                    "file_name": record["file_name"],
                    "citation_url": record["citation_url"],
                    "chunk_index": record["chunk_index"],
                    "content_text": record["content_text"],
                    "vector": record["embedding"],
                    "last_updated": timestamp,
                }
            )

    # -- Delete stale records for files we're about to re-ingest ------------
    if file_ids_seen:
        collection.load()
        for fid in file_ids_seen:
            try:
                # Use json.dumps for safe quoting to prevent expression injection
                safe_fid = json.dumps(fid)
                partition.delete(f"file_unique_id == {safe_fid}")
            except Exception as e:
                print(f"Warning: could not delete stale records for {fid}: {e}")
        collection.flush()
        print(f"Cleaned stale records for {len(file_ids_seen)} files")

    # -- Insert new records -------------------------------------------------
    if records:
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            partition.insert(batch)

        collection.flush()

        # Ensure index exists
        if not collection.has_index():
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": min(1024, max(1, len(records)))},
            }
            collection.create_index("vector", index_params)

        collection.load()
        print(
            f"Inserted {len(records)} records into partition '{partition_name}'. "
            f"Total entities in collection: {collection.num_entities}"
        )

    connections.disconnect("default")


# ====================================================================
# Pipeline definition
# ====================================================================


@dsl.pipeline(
    name="platform-architecture-rag",
    description=(
        "Ingest platform architecture docs (kubeflow/manifests, deployment "
        "guides) into a dedicated Milvus partition for the agentic router."
    ),
)
def platform_architecture_pipeline(
    # -- Source 1: kubeflow/manifests --
    manifests_repo_owner: str = "kubeflow",
    manifests_repo_name: str = "manifests",
    manifests_directory: str = "docs/",
    manifests_file_ext: str = ".md,.yaml,.yml",
    # -- Source 2: website platform/infra docs --
    website_repo_owner: str = "kubeflow",
    website_repo_name: str = "website",
    website_directory: str = "content/en/docs/started",
    website_file_ext: str = ".md,.html",
    # -- Common --
    github_token: str = "",
    base_url_manifests: str = "https://github.com/kubeflow/manifests/blob/master",
    base_url_website: str = "https://www.kubeflow.org/docs/started",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    milvus_host: str = "milvus-standalone-final.docs-agent.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = "docs_rag",
    partition_name: str = "platform_arch",
):
    # -- Download from both sources in parallel --
    manifests_download = download_platform_docs(
        repo_owner=manifests_repo_owner,
        repo_name=manifests_repo_name,
        directory_path=manifests_directory,
        github_token=github_token,
        file_extensions=manifests_file_ext,
    )

    website_download = download_platform_docs(
        repo_owner=website_repo_owner,
        repo_name=website_repo_name,
        directory_path=website_directory,
        github_token=github_token,
        file_extensions=website_file_ext,
    )

    # -- Chunk & embed each source --
    manifests_embed = chunk_and_embed_platform(
        github_data=manifests_download.outputs["github_data"],
        repo_name="manifests",
        base_url=base_url_manifests,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    website_embed = chunk_and_embed_platform(
        github_data=website_download.outputs["github_data"],
        repo_name="website-platform",
        base_url=base_url_website,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # -- Store into the platform_arch partition --
    store_manifests = store_milvus_partition(
        embedded_data=manifests_embed.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        partition_name=partition_name,
    )

    store_website = store_milvus_partition(
        embedded_data=website_embed.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        partition_name=partition_name,
    )
    # store_website runs after store_manifests to avoid race conditions
    # on the shared partition (concurrent inserts can cause index issues)
    store_website.after(store_manifests)


if __name__ == "__main__":
    import os

    os.environ["KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT"] = "true"
    kfp.compiler.Compiler().compile(
        pipeline_func=platform_architecture_pipeline,
        package_path="platform_architecture_pipeline.yaml",
    )
    print("Compiled: platform_architecture_pipeline.yaml")
