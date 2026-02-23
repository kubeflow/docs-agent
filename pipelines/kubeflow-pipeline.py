import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

@dsl.component(
    base_image="python:3.9",
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
    base_image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["sentence-transformers", "langchain"]
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
    import torch
    from sentence_transformers import SentenceTransformer
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    print(f"Model loaded on {device}")

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
                    'file_unique_id': file_unique_id,
                    'repo_name': repo_name,
                    'file_path': file_data['path'],
                    'file_name': file_data['file_name'],
                    'citation_url': citation_url[:1024],
                    'chunk_index': chunk_idx,
                    'content_text': chunk[:2000],
                    'embedding': embedding
                })

    print(f"Created {len(records)} total chunks")

    with open(embedded_data.path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    repo_name: str,
    source_type: str = "docs"
):
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    import json
    from datetime import datetime

    connections.connect("default", host=milvus_host, port=milvus_port)

    # Use a clean partition name
    partition_name = f"{source_type}__{repo_name.replace('/', '_').replace('-', '_')}"

    # Schema definition (keep consistent with current)
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
        FieldSchema(name="last_updated", dtype=DataType.INT64)
    ]
    schema = CollectionSchema(fields, "RAG collection for documentation")

    # Create collection if it doesn't exist, instead of dropping it
    if not utility.has_collection(collection_name):
        collection = Collection(collection_name, schema)
        print(f"Created new collection: {collection_name}")
    else:
        collection = Collection(collection_name)
        print(f"Using existing collection: {collection_name}")

    # Drop existing partition to ensure fresh data for THIS source
    if collection.has_partition(partition_name):
        collection.drop_partition(partition_name)
        print(f"Dropped existing partition: {partition_name}")

    # Create and load the partition
    collection.create_partition(partition_name)
    print(f"Created partition: {partition_name}")

    # Rest of your existing code remains the same...
    records = []
    timestamp = int(datetime.now().timestamp())

    with open(embedded_data.path, 'r', encoding='utf-8') as f:
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
                "last_updated": timestamp
            })

    if records:
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.insert(batch, partition_name=partition_name)

        collection.flush()

        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT", 
            "params": {"nlist": min(1024, len(records))}
        }
        collection.create_index("vector", index_params)
        collection.load()
        print(f"✅ Inserted {len(records)} records. Total: {collection.num_entities}")


@dsl.pipeline(
    name="github-rag",
    description="RAG pipeline for processing GitHub documentation"
)
def github_rag_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website", 
    directory_path: str = "content/en",
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    milvus_host: str = "milvus-standalone-final.santhosh.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = "docs_rag",
    source_type: str = "docs"
):
    # Download GitHub directory
    download_task = download_github_directory(
        repo_owner=repo_owner,
        repo_name=repo_name,
        directory_path=directory_path,
        github_token=github_token
    )
    
    # Chunk and embed the content
    chunk_task = chunk_and_embed(
        github_data=download_task.outputs["github_data"],
        repo_name=repo_name,
        base_url=base_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Store in Milvus
    store_task = store_milvus(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        repo_name=repo_name,
        source_type=source_type
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