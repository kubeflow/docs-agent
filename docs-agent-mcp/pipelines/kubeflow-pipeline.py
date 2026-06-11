import kfp
from kfp import dsl
from typing import Optional


# ---------------------------------------------------------------------------
# Step 1: Download GitHub docs
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "beautifulsoup4"]
)
def download_github_directory(
    repo_owner: str,
    repo_name: str,
    directory_path: str,
    github_token: str,
    github_data: dsl.Output[dsl.Dataset],
):
    import requests
    import json
    import base64
    from bs4 import BeautifulSoup

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}"

    def get_files_recursive(url):
        files = []
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            items = response.json()
            for item in items:
                if item["type"] == "file" and (
                    item["name"].endswith(".md") or item["name"].endswith(".html")
                ):
                    file_response = requests.get(item["url"], headers=headers, timeout=30)
                    file_response.raise_for_status()
                    file_data = file_response.json()
                    content = base64.b64decode(file_data["content"]).decode("utf-8")
                    if item["name"].endswith(".html"):
                        soup = BeautifulSoup(content, "html.parser")
                        content = soup.get_text(separator=" ", strip=True)
                    files.append({
                        "path": item["path"],
                        "content": content,
                        "file_name": item["name"],
                    })
                elif item["type"] == "dir":
                    files.extend(get_files_recursive(item["url"]))
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return files

    files = get_files_recursive(api_url)
    print(f"Downloaded {len(files)} files")

    with open(github_data.path, "w", encoding="utf-8") as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Step 2: Chunk and embed
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "langchain-text-splitters"]
)
def chunk_and_embed(
    github_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    base_url: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings_service_url: str,
    embedding_batch_size: int,
    embedded_data: dsl.Output[dsl.Dataset],
):
    import json
    import os
    import re
    import requests
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"Using external embeddings service at: {embeddings_service_url}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    records = []
    with open(github_data.path, "r", encoding="utf-8") as f:
        for line in f:
            file_data = json.loads(line)
            content = file_data["content"]

            # Clean Hugo frontmatter, templates, HTML, URLs
            content = re.sub(r"^\s*[+\-]{3,}.*?[+\-]{3,}\s*", "", content, flags=re.DOTALL | re.MULTILINE)
            content = re.sub(r"\{\{.*?\}\}", "", content, flags=re.DOTALL)
            content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"https?://\S+", "", content)
            content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
            content = re.sub(r"\s+", " ", content).strip()

            if len(content) < 50:
                print(f"Skipping (too short after cleaning): {file_data['path']}")
                continue

            # Build citation URL
            path_parts = file_data["path"].split("/")
            if "content/en/docs" in file_data["path"]:
                docs_index = path_parts.index("docs")
                url_path = "/".join(path_parts[docs_index + 1:])
                url_path = os.path.splitext(url_path)[0]
                citation_url = f"{base_url}/{url_path}"
            else:
                citation_url = f"{base_url}/{file_data['path']}"

            file_unique_id = f"{repo_name}:{file_data['path']}"
            chunks = text_splitter.split_text(content)
            print(f"{file_data['path']} -> {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks):
                records.append({
                    "file_unique_id": f"{file_unique_id}:{chunk_idx}",
                    "repo_name": repo_name,
                    "file_path": file_data["path"],
                    "file_name": file_data["file_name"],
                    "citation_url": citation_url,
                    "chunk_index": chunk_idx,
                    "content_text": chunk[:1000],
                })

    print(f"Total chunks created: {len(records)}. Generating embeddings via service...")

    # Embed in batches to avoid huge payloads and allow progress reporting
    for i in range(0, len(records), embedding_batch_size):
        batch = records[i:i + embedding_batch_size]
        texts = [r["content_text"] for r in batch]
        
        try:
            response = requests.post(
                embeddings_service_url,
                json={"inputs": texts},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            embeddings = response.json()
            
            for idx, emb in enumerate(embeddings):
                batch[idx]["embedding"] = emb
        except Exception as e:
            print(f"Failed to generate embeddings for batch starting at {i}: {e}")
            raise e

    print(f"Successfully generated embeddings for all {len(records)} chunks.")
    with open(embedded_data.path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Step 3: Store directly in Milvus (no Feast)
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pymilvus>=2.4.0"]
)
def store_in_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_uri: str,
    collection_name: str,
    milvus_batch_size: int,
):
    import json
    from pymilvus import MilvusClient

    print(f"Connecting to Milvus at {milvus_uri}")
    client = MilvusClient(uri=milvus_uri)
    print("Connected.")

    # Load records in batches
    batch = []
    total = 0

    with open(embedded_data.path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            batch.append({
                "file_unique_id": r["file_unique_id"][:512],
                "repo_name":      r["repo_name"][:256],
                "file_path":      r["file_path"][:512],
                "file_name":      r["file_name"][:256],
                "citation_url":   r["citation_url"][:512],
                "chunk_index":    r["chunk_index"],
                "content_text":   r["content_text"][:4096],
                "vector":         r["embedding"],
            })
            if len(batch) >= milvus_batch_size:
                client.insert(collection_name=collection_name, data=batch)
                total += len(batch)
                print(f"Inserted batch — total so far: {total}")
                batch = []

    if batch:
        client.insert(collection_name=collection_name, data=batch)
        total += len(batch)

    print(f"Done. Total records inserted: {total}")
    client.close()


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
@dsl.pipeline(
    name="github-rag-milvus",
    description="RAG pipeline: GitHub docs -> chunk+embed -> Milvus (no Feast)",
)
def github_rag_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website",
    directory_path: str = "content/en/docs",
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embeddings_service_url: str = "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed",
    embedding_batch_size: int = 32,
    milvus_uri: str = "http://milvus-milvus.ml-infra.svc.cluster.local:19530",
    collection_name: str = "kubeflow_docs",
    milvus_batch_size: int = 100,
):
    download_task = download_github_directory(
        repo_owner=repo_owner,
        repo_name=repo_name,
        directory_path=directory_path,
        github_token=github_token,
    )

    chunk_task = chunk_and_embed(
        github_data=download_task.outputs["github_data"],
        repo_name=repo_name,
        base_url=base_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings_service_url=embeddings_service_url,
        embedding_batch_size=embedding_batch_size,
    )

    store_in_milvus(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_uri=milvus_uri,
        collection_name=collection_name,
        milvus_batch_size=milvus_batch_size,
    )


if __name__ == "__main__":
    import os
    os.environ["KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT"] = "true"
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_pipeline,
        package_path="github_rag_pipeline.yaml",
    )
    print("Compiled -> github_rag_pipeline.yaml")