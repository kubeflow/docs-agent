import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["requests", "beautifulsoup4"]
)
def download_github_directory(
    repo_owner: str,
    repo_name: str,
    directory_path: str,
    github_token: str,
    github_data: dsl.Output[dsl.Dataset]
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
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            items = response.json()

            for item in items:
                if item['type'] == 'file' and (item['name'].endswith('.md') or item['name'].endswith('.html')):
                    file_response = requests.get(item['url'], headers=headers)
                    file_response.raise_for_status()
                    file_data = file_response.json()
                    content = base64.b64decode(file_data['content']).decode('utf-8')

                    # Extract text from HTML files
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
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return files

    files = get_files_recursive(api_url)
    print(f"Downloaded {len(files)} files")

    with open(github_data.path, 'w', encoding='utf-8') as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["requests"]
)
@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["sentence-transformers", "langchain-text-splitters"]
)
def chunk_and_embed(
    github_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    base_url: str,
    chunk_size: int,
    chunk_overlap: int,
    embedded_data: dsl.Output[dsl.Dataset]
):
    import json
    import os
    import re
    from sentence_transformers import SentenceTransformer
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("Model loaded on CPU")

    records = []

    with open(github_data.path, 'r', encoding='utf-8') as f:
        for line in f:
            file_data = json.loads(line)
            content = file_data['content']

            # AGGRESSIVE CLEANING FOR BETTER EMBEDDINGS

            # Remove Hugo frontmatter (both --- and +++ styles)
            content = re.sub(r'^\s*[+\-]{3,}.*?[+\-]{3,}\s*', '', content, flags=re.DOTALL | re.MULTILINE)

            # Remove Hugo template syntax
            content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)

            # Remove HTML comments and tags
            content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', ' ', content)

            # Remove navigation/menu artifacts
            content = re.sub(r'\b(Get Started|Contribute|GenAI|Home|Menu|Navigation)\b', '', content, flags=re.IGNORECASE)

            # Clean up URLs and links
            content = re.sub(r'https?://[^\s]+', '', content)
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Convert [text](url) to text

            # Remove excessive whitespace and normalize
            content = re.sub(r'\s+', ' ', content)  # Multiple spaces to single
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines to double
            content = content.strip()

            # Skip files that are too short after cleaning
            if len(content) < 50:
                print(f"Skipping file after cleaning: {file_data['path']} ({len(content)} chars)")
                continue

            # Build citation URL (same as before)
            path_parts = file_data['path'].split('/')
            if 'content/en/docs' in file_data['path']:
                docs_index = path_parts.index('docs')
                url_path = '/'.join(path_parts[docs_index+1:])
                url_path = os.path.splitext(url_path)[0]
                citation_url = f"{base_url}/{url_path}"
            else:
                citation_url = f"{base_url}/{file_data['path']}"

            file_unique_id = f"{repo_name}:{file_data['path']}"

            # Create splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            # Split into chunks
            chunks = text_splitter.split_text(content)

            print(f"File: {file_data['path']} -> {len(chunks)} chunks (avg: {sum(len(c) for c in chunks)/len(chunks):.0f} chars)")

            # Create embeddings
            for chunk_idx, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                records.append({
                    'file_unique_id': f"{file_unique_id}:{chunk_idx}",
                    'repo_name': repo_name,
                    'file_path': file_data['path'],
                    'file_name': file_data['file_name'],
                    'citation_url': citation_url,
                    'chunk_index': chunk_idx,
                    'content_text': chunk,
                    'embedding': embedding
                })

    print(f"Created {len(records)} total chunks")

    with open(embedded_data.path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["feast[milvus]", "pandas", "marshmallow>=3.13.0"]
)
def store_via_feast(
    embedded_data: dsl.Input[dsl.Dataset],
    feast_online_store_host: str,
    feast_project: str,
):
    import json
    import os
    import pandas as pd
    import inspect
    from datetime import datetime, timedelta
    from feast import FeatureStore, Entity, FeatureView, Field, FileSource
    from feast.types import String, Int64, Float32, Array, UnixTimestamp

    # Patch Feast VARCHAR limit (hardcoded 512 -> 4096) and reload module
    import importlib
    import feast.infra.online_stores.milvus_online_store.milvus as milvus_mod
    src_file = inspect.getfile(milvus_mod)
    with open(src_file, "r") as f:
        content = f.read()
    if "max_length=512" in content:
        with open(src_file, "w") as f:
            f.write(content.replace("max_length=512", "max_length=4096"))
        print("Patched Feast VARCHAR limit to 4096")
    importlib.reload(milvus_mod)
    print("Reloaded Feast Milvus module")

    # Drop existing collection so it gets recreated with the patched schema
    from pymilvus import connections, utility
    milvus_host = feast_online_store_host.replace("http://", "").replace("https://", "")
    connections.connect("default", host=milvus_host, port="19530")
    collection_name = f"{feast_project}_docs_rag"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")
    connections.disconnect("default")

    feast_dir = "/tmp/feast_repo"
    os.makedirs(f"{feast_dir}/data", exist_ok=True)

    with open(f"{feast_dir}/feature_store.yaml", "w") as f:
        f.write(f"""project: {feast_project}
provider: local
registry: data/registry.db
online_store:
  type: milvus
  host: {feast_online_store_host}
  port: 19530
  username: root
  password: Milvus
  vector_enabled: true
  embedding_dim: 768
  index_type: IVF_FLAT
  metric_type: COSINE
offline_store:
  type: file
entity_key_serialization_version: 3
auth:
  type: no_auth
""")

    # Define Feast objects inline
    doc_chunk = Entity(name="doc_chunk", join_keys=["file_unique_id"])
    docs_source = FileSource(path="data/embedded_docs.parquet", timestamp_field="event_timestamp")
    docs_rag = FeatureView(
        name="docs_rag",
        entities=[doc_chunk],
        schema=[
            Field(name="file_unique_id", dtype=String),
            Field(name="repo_name", dtype=String),
            Field(name="file_path", dtype=String),
            Field(name="file_name", dtype=String),
            Field(name="citation_url", dtype=String),
            Field(name="chunk_index", dtype=Int64),
            Field(name="content_text", dtype=String),
            Field(name="vector", dtype=Array(Float32), vector_index=True, vector_search_metric="COSINE"),
            Field(name="event_timestamp", dtype=UnixTimestamp),
        ],
        source=docs_source,
        ttl=timedelta(days=30),
    )

    records = []
    with open(embedded_data.path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "file_unique_id": record['file_unique_id'],
                "repo_name": record["repo_name"],
                "file_path": record["file_path"],
                "file_name": record["file_name"],
                "citation_url": record["citation_url"],
                "chunk_index": record["chunk_index"],
                "content_text": record["content_text"],
                "vector": record["embedding"],
                "event_timestamp": datetime.now(),
            })

    df = pd.DataFrame(records)

    store = FeatureStore(repo_path=feast_dir)
    store.apply([doc_chunk, docs_source, docs_rag])
    store.write_to_online_store(feature_view_name="docs_rag", df=df)
    print(f"Wrote {len(records)} records to Feast online store")


@dsl.pipeline(
    name="github-rag-feast",
    description="RAG pipeline: GitHub docs -> chunk -> embed -> Feast"
)
def github_rag_feast_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website",
    directory_path: str = "content/en",
    github_token: str = "",
    base_url: str = "https://<YOUR_DOCS_BASE_URL>",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    feast_online_store_host: str = "http://milvus.<YOUR_NAMESPACE>.svc.cluster.local",
    feast_project: str = "kubeflow_docs",
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
    )

    store_via_feast(
        embedded_data=chunk_task.outputs["embedded_data"],
        feast_online_store_host=feast_online_store_host,
        feast_project=feast_project,
    )



if __name__ == "__main__":
    import os
    os.environ['KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT'] = 'true'
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_feast_pipeline,
        package_path="github_rag_pipeline.yaml",
    )