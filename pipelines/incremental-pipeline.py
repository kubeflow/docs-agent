import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "beautifulsoup4"]
)
def download_specific_files(
    repo_owner: str,
    repo_name: str,
    file_paths: str,  # JSON string of file paths list
    github_token: str,
    github_data: dsl.Output[dsl.Dataset]
):
    import requests
    import json
    import base64
    from bs4 import BeautifulSoup

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    
    # Parse the file paths from JSON string
    try:
        file_paths_list = json.loads(file_paths)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file_paths: {file_paths}")
        file_paths_list = []
    
    print(f"Processing {len(file_paths_list)} changed files")
    
    files = []
    
    for file_path in file_paths_list:
        # Skip non-documentation files
        if not (file_path.endswith('.md') or file_path.endswith('.html')):
            print(f"Skipping non-doc file: {file_path}")
            continue
            
        try:
            # Get file content from GitHub API
            api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            file_data = response.json()
            
            # Decode content
            content = base64.b64decode(file_data['content']).decode('utf-8')
            
            # Extract text from HTML files
            if file_path.endswith('.html'):
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text(separator=' ', strip=True)
            
            files.append({
                'path': file_path,
                'content': content,
                'file_name': file_data['name']
            })
            print(f"Downloaded: {file_path}")
            
        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
            continue
    
    print(f"Successfully downloaded {len(files)} files")
    
    # Save to output dataset
    with open(github_data.path, 'w', encoding='utf-8') as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus"]
)
def delete_old_vectors(
    file_paths: str,  # JSON string of file paths list
    repo_name: str,
    milvus_host: str,
    milvus_port: str,
    collection_name: str
):
    from pymilvus import connections, Collection
    import json
    
    # Connect to Milvus
    connections.connect("default", host=milvus_host, port=milvus_port)
    
    # Parse file paths
    try:
        file_paths_list = json.loads(file_paths)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file_paths: {file_paths}")
        return
    
    # Check if collection exists
    try:
        collection = Collection(collection_name)
        collection.load()
        print(f"Connected to collection: {collection_name}")
        
        # Delete old vectors for each changed file
        deleted_count = 0
        for file_path in file_paths_list:
            file_unique_id = f"{repo_name}:{file_path}"
            
            # Delete vectors with matching file_unique_id
            expr = f'file_unique_id == "{file_unique_id}"'
            try:
                # Get count before deletion for logging
                query_result = collection.query(
                    expr=expr,
                    output_fields=["id"],
                    limit=10000
                )
                count_before = len(query_result)
                
                if count_before > 0:
                    # Delete the vectors
                    collection.delete(expr)
                    collection.flush()
                    deleted_count += count_before
                    print(f"Deleted {count_before} vectors for file: {file_path}")
                else:
                    print(f"No existing vectors found for file: {file_path}")
                    
            except Exception as e:
                print(f"Error deleting vectors for {file_path}: {e}")
                continue
        
        print(f"✅ Total deleted vectors: {deleted_count}")
        
    except Exception as e:
        print(f"Error connecting to collection {collection_name}: {e}")
        print("Collection might not exist yet - this is okay for first run")


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["requests", "langchain"]
)
def chunk_and_embed_incremental(
    github_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    base_url: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_service_url: str,
    embedded_data: dsl.Output[dsl.Dataset]
):
    import json
    import os
    import re
    import requests
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    BATCH_SIZE = 64  # chunks per embedding-service call

    def get_embeddings(texts: list) -> list:
        """Call the centralised embedding service (ADR-004)."""
        resp = requests.post(
            f"{embedding_service_url}/embed",
            json={"texts": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    # ── Phase 1: clean + chunk ───────────────────────────────────────
    records_pending = []   # records WITHOUT embeddings yet

    with open(github_data.path, 'r', encoding='utf-8') as f:
        for line in f:
            file_data = json.loads(line)
            content = file_data['content']

            # AGGRESSIVE CLEANING FOR BETTER EMBEDDINGS (same as original)
            
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

            # Build citation URL
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

            # Collect records (embedding added in Phase 2)
            for chunk_idx, chunk in enumerate(chunks):
                records_pending.append({
                    'file_unique_id': file_unique_id,
                    'repo_name': repo_name,
                    'file_path': file_data['path'],
                    'file_name': file_data['file_name'],
                    'citation_url': citation_url[:1024],
                    'chunk_index': chunk_idx,
                    'content_text': chunk[:2000],
                })

    # ── Phase 2: batch-embed via the embedding service ──────────────
    all_chunks = [r['content_text'] for r in records_pending]
    print(f"Embedding {len(all_chunks)} chunks in batches of {BATCH_SIZE}...")

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        embeddings = get_embeddings(batch)
        for j, emb in enumerate(embeddings):
            records_pending[i + j]['embedding'] = emb

    print(f"Created {len(records_pending)} total chunks for incremental update")

    with open(embedded_data.path, 'w', encoding='utf-8') as f:
        for record in records_pending:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_milvus_incremental(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str
):
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    import json
    from datetime import datetime

    connections.connect("default", host=milvus_host, port=milvus_port)

    # Check if collection exists, if not create it
    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} doesn't exist, creating it...")
        
        # Enhanced schema with 768 dimensions
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
        collection = Collection(collection_name, schema)
        print(f"Created new collection: {collection_name}")
    else:
        collection = Collection(collection_name)
        print(f"Using existing collection: {collection_name}")

    # Load collection
    collection.load()

    # Prepare records for insertion
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
        # Insert new records
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.insert(batch)

        collection.flush()

        # Create/update index if needed
        try:
            # Check if index exists
            index_info = collection.index()
            if not index_info:
                print("Creating index...")
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT", 
                    "params": {"nlist": min(1024, max(100, len(records)))}
                }
                collection.create_index("vector", index_params)
                collection.load()
                print("Index created successfully")
            else:
                print("Index already exists")
        except Exception as e:
            print(f"Index operation result: {e}")

        print(f"✅ Inserted {len(records)} new records. Total collection size: {collection.num_entities}")
    else:
        print("No records to insert")


@dsl.pipeline(
    name="github-rag-incremental-build",
    description="Incremental RAG pipeline for processing only changed GitHub files"
)
def github_rag_incremental_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "website", 
    changed_files: str = '[]',  # JSON string of changed file paths
    github_token: str = "",
    base_url: str = "https://www.kubeflow.org/docs",
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    embedding_service_url: str = "http://embedding-service.docs-agent.svc.cluster.local:8080",
    milvus_host: str = "milvus-standalone-final.docs-agent.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = "docs_rag"
):
    # Step 1: Delete old vectors for changed files
    delete_task = delete_old_vectors(
        file_paths=changed_files,
        repo_name=repo_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name
    )
    
    # Step 2: Download only the changed files
    download_task = download_specific_files(
        repo_owner=repo_owner,
        repo_name=repo_name,
        file_paths=changed_files,
        github_token=github_token
    )
    
    # Step 3: Chunk and embed the changed files (via embedding service - ADR-004)
    chunk_task = chunk_and_embed_incremental(
        github_data=download_task.outputs["github_data"],
        repo_name=repo_name,
        base_url=base_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_service_url=embedding_service_url,
    )
    
    # Step 4: Store new vectors in Milvus (after deletion is complete)
    store_task = store_milvus_incremental(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name
    )
    
    # Ensure deletion happens before insertion
    store_task.after(delete_task)


if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_incremental_pipeline,
        package_path="github_rag_incremental_pipeline.yaml"
    )