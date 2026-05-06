import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *


@dsl.component(
    base_image="docker.io/python:3.9",
    packages_to_install=["requests"]
)
def download_github_code(
    repos: str,
    directory_paths: str,
    file_extensions: str,
    github_token: str,
    code_data: dsl.Output[dsl.Dataset]
):
    """Fetch code files from GitHub repositories for RAG indexing.

    Recursively downloads files matching given extensions from specified
    directories across multiple repositories.

    Args:
        repos: Comma-separated repos (e.g., "kubeflow/manifests,kubeflow/pipelines").
        directory_paths: Comma-separated directory paths to crawl per repo
            (e.g., "apps/pipeline,common/istio").
        file_extensions: Comma-separated extensions to include
            (e.g., "yaml,yml,py,json").
        github_token: GitHub personal access token.
        code_data: Output dataset path.
    """
    import requests
    import json
    import base64
    import time

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    extensions = [ext.strip().lstrip(".") for ext in file_extensions.split(",")]

    def api_request(url, params=None):
        """GitHub API request with rate limit handling and retries."""
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

                if resp.status_code == 200:
                    return resp.json()
                else:
                    print(f"API error: HTTP {resp.status_code} for {url}")
                    return None

            except Exception as e:
                print(f"Request failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)

        return None

    def get_files_recursive(owner, name, path):
        """Recursively fetch files from a GitHub directory."""
        files = []
        url = f"https://api.github.com/repos/{owner}/{name}/contents/{path}"
        items = api_request(url)

        if not items or not isinstance(items, list):
            return files

        for item in items:
            if item["type"] == "file":
                ext = item["name"].rsplit(".", 1)[-1].lower() if "." in item["name"] else ""
                # Also include extensionless files like Dockerfile, Makefile
                include_no_ext = item["name"] in ("Dockerfile", "Makefile", "Kustomization")
                if ext in extensions or include_no_ext:
                    try:
                        file_resp = api_request(item["url"])
                        if file_resp and "content" in file_resp:
                            content = base64.b64decode(file_resp["content"]).decode("utf-8")
                            files.append({
                                "path": item["path"],
                                "content": content,
                                "file_name": item["name"],
                                "repo": f"{owner}/{name}",
                            })
                    except Exception as e:
                        print(f"Error decoding {item['path']}: {e}")
            elif item["type"] == "dir":
                files.extend(get_files_recursive(owner, name, item["path"]))

        return files

    all_files = []
    dirs = [d.strip() for d in directory_paths.split(",")]

    for repo in repos.split(","):
        repo = repo.strip()
        if "/" not in repo:
            print(f"Skipping invalid repo format: {repo}")
            continue

        owner, name = repo.split("/", 1)
        print(f"Fetching code from {owner}/{name}...")

        for dir_path in dirs:
            print(f"  Scanning {dir_path}/ ...")
            files = get_files_recursive(owner, name, dir_path)
            all_files.extend(files)
            print(f"  Found {len(files)} files in {dir_path}/")

    print(f"Total code files fetched: {len(all_files)}")

    with open(code_data.path, "w", encoding="utf-8") as f:
        for file_data in all_files:
            f.write(json.dumps(file_data, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["sentence-transformers==3.3.1", "transformers==4.44.2", "langchain-text-splitters", "pyyaml"]
)
def chunk_and_embed_code(
    code_data: dsl.Input[dsl.Dataset],
    chunk_size: int,
    chunk_overlap: int,
    embedded_data: dsl.Output[dsl.Dataset]
):
    """Chunk code files using YAML-aware and Python AST-aware parsing, then embed.

    Routes files to the appropriate parser:
    - .yaml/.yml: Split at --- boundaries, extract K8s metadata
    - .py: Split at function/class boundaries using ast
    - .json: Index as single chunk
    - Others: Text-based fallback

    Oversized chunks are sub-split with RecursiveCharacterTextSplitter.
    """
    import json
    import re
    import ast as ast_module
    import yaml
    import torch
    from sentence_transformers import SentenceTransformer
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    print(f"Model loaded on {device}")

    # --- Inline chunking logic (KFP components can't import local modules) ---

    def parse_yaml_documents(content, file_path=""):
        chunks = []
        file_name = file_path.rsplit("/", 1)[-1] if file_path else ""
        if file_name in ("kustomization.yaml", "kustomization.yml"):
            file_type = "kustomize"
        else:
            file_type = "yaml"

        raw_docs = re.split(r'^---\s*$', content, flags=re.MULTILINE)
        for raw_doc in raw_docs:
            raw_doc = raw_doc.strip()
            if not raw_doc or len(raw_doc) < 10:
                continue

            resource_kind = ""
            resource_name = ""
            resource_namespace = ""
            try:
                parsed = yaml.safe_load(raw_doc)
                if isinstance(parsed, dict):
                    resource_kind = str(parsed.get("kind", ""))
                    metadata = parsed.get("metadata", {})
                    if isinstance(metadata, dict):
                        resource_name = str(metadata.get("name", ""))
                        resource_namespace = str(metadata.get("namespace", ""))
            except yaml.YAMLError:
                pass

            chunks.append({
                "content": raw_doc,
                "resource_kind": resource_kind,
                "resource_name": resource_name,
                "resource_namespace": resource_namespace,
                "file_type": file_type,
            })
        return chunks

    def parse_python_ast(content, file_path=""):
        chunks = []
        lines = content.split("\n")
        try:
            tree = ast_module.parse(content)
        except SyntaxError:
            return [{
                "content": content,
                "resource_kind": "",
                "resource_name": "",
                "resource_namespace": "",
                "file_type": "python",
            }]

        nodes = []
        for node in ast_module.iter_child_nodes(tree):
            if isinstance(node, (ast_module.FunctionDef, ast_module.AsyncFunctionDef, ast_module.ClassDef)):
                nodes.append(node)

        if not nodes:
            return [{
                "content": content,
                "resource_kind": "module",
                "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
                "resource_namespace": "",
                "file_type": "python",
            }]

        first_node_line = nodes[0].lineno
        if first_node_line > 1:
            header = "\n".join(lines[:first_node_line - 1]).strip()
            if header and len(header) >= 10:
                chunks.append({
                    "content": header,
                    "resource_kind": "module_header",
                    "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
                    "resource_namespace": "",
                    "file_type": "python",
                })

        for i, node in enumerate(nodes):
            start_line = node.lineno - 1
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno - 1
            if i + 1 < len(nodes):
                next_start = nodes[i + 1].lineno - 1
                if nodes[i + 1].decorator_list:
                    next_start = nodes[i + 1].decorator_list[0].lineno - 1
                end_line = next_start
            else:
                end_line = len(lines)

            chunk_content = "\n".join(lines[start_line:end_line]).rstrip()
            if not chunk_content or len(chunk_content) < 10:
                continue

            if isinstance(node, ast_module.ClassDef):
                kind = "class"
            elif isinstance(node, ast_module.AsyncFunctionDef):
                kind = "async_function"
            else:
                kind = "function"

            chunks.append({
                "content": chunk_content,
                "resource_kind": kind,
                "resource_name": node.name,
                "resource_namespace": "",
                "file_type": "python",
            })
        return chunks

    def chunk_code_file(content, file_path, c_size, c_overlap):
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        if ext in ("yaml", "yml"):
            raw_chunks = parse_yaml_documents(content, file_path)
        elif ext == "py":
            raw_chunks = parse_python_ast(content, file_path)
        elif ext == "json":
            raw_chunks = [{
                "content": content,
                "resource_kind": "",
                "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
                "resource_namespace": "",
                "file_type": "json",
            }]
        else:
            raw_chunks = [{
                "content": content,
                "resource_kind": "",
                "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
                "resource_namespace": "",
                "file_type": ext if ext else "text",
            }]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        result = []
        for chunk in raw_chunks:
            text = chunk["content"]
            if len(text) <= c_size:
                result.append(chunk)
            else:
                for sub_text in splitter.split_text(text):
                    result.append({
                        "content": sub_text,
                        "resource_kind": chunk["resource_kind"],
                        "resource_name": chunk["resource_name"],
                        "resource_namespace": chunk["resource_namespace"],
                        "file_type": chunk["file_type"],
                    })
        return result

    # --- End inline chunking logic ---

    records = []

    with open(code_data.path, "r", encoding="utf-8") as f:
        for line in f:
            file_data = json.loads(line)
            content = file_data["content"]
            file_path = file_data["path"]
            repo = file_data.get("repo", "")

            # Light cleaning: normalize whitespace but preserve structure
            # (YAML/Python are whitespace-sensitive, so no aggressive cleaning)
            content = content.rstrip()
            if len(content) < 10:
                print(f"Skipping tiny file: {file_path} ({len(content)} chars)")
                continue

            file_unique_id = f"{repo}:{file_path}"
            # Citation URL: link to GitHub blob
            citation_url = f"https://github.com/{repo}/blob/main/{file_path}"

            chunks = chunk_code_file(content, file_path, chunk_size, chunk_overlap)

            print(f"File: {file_path} -> {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks):
                embedding = model.encode(chunk["content"]).tolist()
                records.append({
                    "file_unique_id": file_unique_id,
                    "repo_name": repo,
                    "file_path": file_path,
                    "file_name": file_data["file_name"],
                    "citation_url": citation_url[:1024],
                    "chunk_index": chunk_idx,
                    "content_text": chunk["content"][:2000],
                    "embedding": embedding,
                    "resource_kind": chunk["resource_kind"][:128],
                    "resource_name": chunk["resource_name"][:256],
                    "resource_namespace": chunk["resource_namespace"][:256],
                    "file_type": chunk["file_type"][:64],
                })

    print(f"Created {len(records)} total code chunks")

    with open(embedded_data.path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="docker.io/python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_code_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str
):
    """Store code embeddings in the code_rag Milvus collection.

    Creates the collection with the code_rag schema if it doesn't exist.
    Uses upsert-style logic: drops and recreates to avoid schema drift.
    """
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    import json
    from datetime import datetime

    connections.connect("default", host=milvus_host, port=milvus_port)

    # Drop existing collection to ensure schema matches
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    # code_rag schema: docs_rag fields + resource_kind, resource_name,
    # resource_namespace, file_type
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
        # Code-specific fields
        FieldSchema(name="resource_kind", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="resource_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="resource_namespace", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=64),
    ]

    schema = CollectionSchema(fields, "RAG collection for Kubeflow code and manifests")
    collection = Collection(collection_name, schema)
    print(f"Created new collection: {collection_name}")

    records = []
    timestamp = int(datetime.now().timestamp())

    with open(embedded_data.path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "file_unique_id": record["file_unique_id"],
                "repo_name": record["repo_name"],
                "file_path": record["file_path"],
                "file_name": record["file_name"],
                "citation_url": record["citation_url"],
                "chunk_index": record["chunk_index"],
                "content_text": record["content_text"],
                "vector": record["embedding"],
                "last_updated": timestamp,
                "resource_kind": record.get("resource_kind", ""),
                "resource_name": record.get("resource_name", ""),
                "resource_namespace": record.get("resource_namespace", ""),
                "file_type": record.get("file_type", ""),
            })

    if records:
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.insert(batch)

        collection.flush()

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": min(1024, len(records))},
        }
        collection.create_index("vector", index_params)
        collection.load()
        print(f"Inserted {len(records)} records. Total: {collection.num_entities}")
    else:
        print("No records to insert")


@dsl.pipeline(
    name="code-rag",
    description="RAG pipeline for ingesting Kubeflow code and YAML manifests"
)
def code_rag_pipeline(
    repos: str = "kubeflow/manifests",
    directory_paths: str = "apps/pipeline/upstream,apps/katib,common/istio,apps/jupyter",
    file_extensions: str = "yaml,yml,py,json",
    github_token: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    milvus_host: str = "milvus-standalone-final.docs-agent.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = "code_rag"
):
    download_task = download_github_code(
        repos=repos,
        directory_paths=directory_paths,
        file_extensions=file_extensions,
        github_token=github_token,
    )

    chunk_task = chunk_and_embed_code(
        code_data=download_task.outputs["code_data"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    store_task = store_code_milvus(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    import os
    os.environ["KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT"] = "true"

    kfp.compiler.Compiler().compile(
        pipeline_func=code_rag_pipeline,
        package_path="code_rag_pipeline.yaml",
    )
