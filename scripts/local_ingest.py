"""
Local data ingestion script for development.

This replicates what the Kubeflow Pipeline does, but runs locally.
In production, use the KFP pipeline (pipelines/kubeflow-pipeline.py) instead.

Usage:
    source .env
    python scripts/local_ingest.py
"""

import os
import re
import json
import base64
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection


def connect_milvus():
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    connections.connect("default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")


def create_collection(name):
    if utility.has_collection(name):
        Collection(name).drop()
        print(f"Dropped existing collection: {name}")

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
    schema = CollectionSchema(fields, "RAG collection for documentation")
    collection = Collection(name, schema)
    print(f"Created collection: {name}")
    return collection


def fetch_docs(repo_owner, repo_name, directory_path, token=None, max_files=20):
    headers = {"Authorization": f"token {token}"} if token else {}
    files = []

    def fetch_recursive(path):
        if len(files) >= max_files:
            return
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        items = response.json()

        for item in items:
            if len(files) >= max_files:
                return
            if item["type"] == "file" and item["name"].endswith(".md"):
                try:
                    file_resp = requests.get(item["url"], headers=headers)
                    file_resp.raise_for_status()
                    content = base64.b64decode(file_resp.json()["content"]).decode("utf-8")
                    files.append({
                        "path": item["path"],
                        "content": content,
                        "file_name": item["name"],
                    })
                    print(f"  Fetched: {item['path']}")
                except Exception as e:
                    print(f"  Error fetching {item['path']}: {e}")
            elif item["type"] == "dir":
                fetch_recursive(item["path"])

    fetch_recursive(directory_path)
    print(f"Fetched {len(files)} files from GitHub")
    return files


def clean_content(content):
    content = re.sub(r"^\s*[+-]{3,}.*?[+-]{3,}\s*", "", content, flags=re.DOTALL | re.MULTILINE)
    content = re.sub(r"\{\{.*?\}\}", "", content, flags=re.DOTALL)
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"https?://[^\s]+", "", content)
    content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def chunk_and_embed(files, model, repo_name, base_url, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    records = []
    timestamp = int(datetime.now().timestamp())

    for f in files:
        content = clean_content(f["content"])
        if len(content) < 50:
            print(f"  Skipping (too short): {f['file_name']}")
            continue

        chunks = splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            records.append({
                "file_unique_id": f"{repo_name}:{f['path']}",
                "repo_name": repo_name,
                "file_path": f["path"],
                "file_name": f["file_name"],
                "citation_url": f"{base_url}/{f['file_name'].replace('.md', '')}"[:1024],
                "chunk_index": idx,
                "content_text": chunk[:2000],
                "vector": embedding,
                "last_updated": timestamp,
            })
        print(f"  Chunked: {f['file_name']} -> {len(chunks)} chunks")

    print(f"Total: {len(records)} vectors")
    return records


def store_vectors(collection, records):
    if not records:
        print("No records to store")
        return

    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        collection.insert(batch)

    collection.flush()
    collection.create_index(
        "vector",
        {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
    )
    collection.load()
    print(f"Stored {len(records)} vectors in Milvus")


def main():
    collection_name = os.getenv("MILVUS_COLLECTION", "docs_rag")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    github_token = os.getenv("GITHUB_TOKEN", "")

    connect_milvus()
    collection = create_collection(collection_name)

    print("\nLoading embedding model...")
    model = SentenceTransformer(embedding_model)

    print("\nFetching docs from GitHub...")
    files = fetch_docs("kubeflow", "website", "content/en/docs", github_token)

    print("\nChunking and embedding...")
    records = chunk_and_embed(files, model, "website", "https://www.kubeflow.org/docs")

    print("\nStoring in Milvus...")
    store_vectors(collection, records)

    print("\nDone!")


if __name__ == "__main__":
    main()
