import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

try:
    import kfp.kubernetes as k8s
except ImportError:  # pragma: no cover - optional at compile time
    k8s = None

from utils import DEFAULT_EMBEDDING_BATCH_SIZE, DOCS_COLLECTION

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


@dsl.component(
    base_image="docker.io/library/python:3.9",
    packages_to_install=["requests"]
)
def download_github_issues(
    repos: str,
    labels: str,
    state: str,
    max_issues_per_repo: int,
    github_token: str,
    issues_data: dsl.Output[dsl.Dataset]
):
    """Fetch GitHub issues and comments from multiple repos for RAG indexing.
    
    Args:
        repos: Comma-separated list of repos (e.g., "kubeflow/kubeflow,kubeflow/pipelines")
        labels: Comma-separated labels to filter (e.g., "kind/bug,kind/question")
        state: Issue state - "open", "closed", or "all"
        max_issues_per_repo: Maximum issues to fetch per repository
        github_token: GitHub personal access token for API authentication
        issues_data: Output dataset path
    """
    import requests
    import json
    import time
    import os

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
    all_issues = []

    def api_request(url, params=None):
        """Make GitHub API request with rate limit handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers)
                
                # Handle rate limiting
                if resp.status_code == 403:
                    remaining = resp.headers.get("X-RateLimit-Remaining", "0")
                    if remaining == "0":
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                        wait_time = max(reset_time - int(time.time()), 60)
                        print(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(min(wait_time, 300))  # Max 5 min wait
                        continue
                
                if resp.status_code == 200:
                    return resp.json()
                else:
                    print(f"API error: HTTP {resp.status_code}")
                    return None
                    
            except Exception as e:
                print(f"Request failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None

    def fetch_comments(owner, name, issue_number):
        """Fetch all comments for a single issue."""
        comments_url = f"https://api.github.com/repos/{owner}/{name}/issues/{issue_number}/comments"
        comments_text = ""
        page = 1
        
        while True:
            comments = api_request(comments_url, {"per_page": 100, "page": page})
            if not comments:
                break
                
            for comment in comments:
                author = comment.get("user", {}).get("login", "unknown")
                created = comment.get("created_at", "")[:10]
                body = comment.get("body", "") or ""
                comments_text += f"\n\n---\n**Comment by @{author}** ({created}):\n{body}"
            
            if len(comments) < 100:
                break
            page += 1
        
        return comments_text

    for repo in repos.split(","):
        repo = repo.strip()
        if "/" not in repo:
            print(f"Skipping invalid repo format: {repo}")
            continue

        owner, name = repo.split("/", 1)
        print(f"Fetching issues from {owner}/{name}...")

        page = 1
        repo_issues = []

        while len(repo_issues) < max_issues_per_repo:
            url = f"https://api.github.com/repos/{owner}/{name}/issues"
            params = {
                "state": state,
                "labels": labels,
                "per_page": 100,
                "page": page
            }

            issues = api_request(url, params)
            if not issues:
                break

            for issue in issues:
                if "pull_request" in issue:
                    continue

                labels_str = ", ".join([l["name"] for l in issue.get("labels", [])])
                issue_url = issue.get("html_url", "")
                created_at = issue.get("created_at", "")[:10]
                updated_at = issue.get("updated_at", "")[:10]

                # Build issue content with full metadata
                content = f"# {issue['title']}\n\n"
                content += f"**Repository:** {repo}\n"
                content += f"**Issue:** #{issue['number']}\n"
                content += f"**URL:** {issue_url}\n"
                content += f"**Labels:** {labels_str}\n"
                content += f"**State:** {issue['state']}\n"
                content += f"**Created:** {created_at}\n"
                content += f"**Updated:** {updated_at}\n\n"
                content += issue.get("body", "") or ""

                # Fetch and append comments
                if issue.get("comments", 0) > 0:
                    comments = fetch_comments(owner, name, issue["number"])
                    content += comments

                repo_issues.append({
                    "path": f"issues/{name}/{issue['number']}",
                    "content": content,
                    "file_name": f"issue-{name}-{issue['number']}.md",
                    "url": issue_url
                })

                if len(repo_issues) >= max_issues_per_repo:
                    break

            page += 1

        all_issues.extend(repo_issues)
        print(f"  Fetched {len(repo_issues)} issues from {repo}")

    print(f"Total issues fetched: {len(all_issues)}")

    with open(issues_data.path, 'w', encoding='utf-8') as f:
        for issue_data in all_issues:
            f.write(json.dumps(issue_data, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["requests", "langchain-text-splitters"],
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

    print(f"Using embeddings service: {embeddings_service_url}")
    embedding_batch_size = max(1, int(embedding_batch_size))

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

            print(f"File: {file_data['path']} -> {len(chunks)} chunks")

            for chunk_idx, chunk in enumerate(chunks):
                records.append({
                    'file_unique_id': file_unique_id,
                    'repo_name': repo_name,
                    'file_path': file_data['path'],
                    'file_name': file_data['file_name'],
                    'citation_url': citation_url[:1024],
                    'chunk_index': chunk_idx,
                    'content_text': chunk[:2000],
                })

    print(f"Created {len(records)} chunks; requesting embeddings from TEI service...")

    # TEI all-mpnet-base-v2 rejects any input >=384 tokens (~1000 chars).
    max_tei_chars = 1000
    for i in range(0, len(records), embedding_batch_size):
        batch = records[i:i + embedding_batch_size]
        texts = [r["content_text"][:max_tei_chars] for r in batch]
        response = requests.post(
            embeddings_service_url,
            json={"inputs": texts},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        vectors = response.json()
        for idx, vector in enumerate(vectors):
            batch[idx]["embedding"] = vector

    print(f"Embedded {len(records)} chunks")

    with open(embedded_data.path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


@dsl.component(
    base_image="docker.io/library/python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    repo_name: str,
    directory_path: str,
):
   
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    import json
    import os
    from datetime import datetime

    SCHEMA_VERSION = 1
    SCHEMA_DESCRIPTION = f"RAG collection for documentation (v={SCHEMA_VERSION})"
    DELETE_BATCH_SIZE = 100
    QUERY_BATCH_SIZE = 2000

    def _escape_milvus_str(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _normalize_directory_path(path: str) -> str:
        return path.strip().strip("/")

    def _build_scope_expr(repo_name: str, directory_path: str) -> str:
        normalized_dir = _normalize_directory_path(directory_path)
        scoped_prefix = f"{repo_name}:{normalized_dir}/"
        return f'file_unique_id like "{_escape_milvus_str(scoped_prefix)}%"'

    def _file_unique_id_in_scope(file_unique_id: str, repo_name: str, directory_path: str) -> bool:
        normalized_dir = _normalize_directory_path(directory_path)
        scope_prefix = f"{repo_name}:{normalized_dir}/"
        return file_unique_id.startswith(scope_prefix)

    def _delete_file_unique_ids(collection: Collection, file_ids: list[str], batch_size: int) -> int:
        deleted = 0
        current_batch_ids = None
        try:
            for i in range(0, len(file_ids), batch_size):
                current_batch_ids = file_ids[i:i + batch_size]
                quoted = ", ".join(f'"{_escape_milvus_str(uid)}"' for uid in current_batch_ids)
                expr = f"file_unique_id in [{quoted}]"
                old = collection.query(expr=expr, output_fields=["id"], limit=16384)
                if old:
                    collection.delete(expr)
                    deleted += len(old)
        except Exception as e:
            print(f"ERROR during delete phase: {e}")
            print(f"Failed batch unique_ids ({len(current_batch_ids) if current_batch_ids else 0} ids): {current_batch_ids}")
            raise
        return deleted

    def _collect_scoped_file_unique_ids(collection: Collection, repo_name: str, directory_path: str) -> set[str]:
        existing_file_ids: set[str] = set()
        scope_expr = _build_scope_expr(repo_name, directory_path)

        try:
            iterator = collection.query_iterator(
                batch_size=QUERY_BATCH_SIZE,
                expr=scope_expr,
                output_fields=["file_unique_id"],
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    for row in batch:
                        file_unique_id = row.get("file_unique_id")
                        if file_unique_id:
                            existing_file_ids.add(file_unique_id)
            finally:
                iterator.close()
            return existing_file_ids
        except Exception as e:
            print(
                "WARNING: Scoped Milvus query_iterator failed; "
                f"falling back to repo-level filtering in Python. Error: {e}"
            )

        repo_expr = f'repo_name == "{_escape_milvus_str(repo_name)}"'
        iterator = collection.query_iterator(
            batch_size=QUERY_BATCH_SIZE,
            expr=repo_expr,
            output_fields=["file_unique_id"],
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    file_unique_id = row.get("file_unique_id")
                    if file_unique_id and _file_unique_id_in_scope(file_unique_id, repo_name, directory_path):
                        existing_file_ids.add(file_unique_id)
        finally:
            iterator.close()

        return existing_file_ids

    milvus_user = os.environ.get("MILVUS_USER", "root")
    milvus_password = os.environ.get("MILVUS_PASSWORD", "")
    if not milvus_password:
        raise RuntimeError("MILVUS_PASSWORD must be set via pipeline secret (not in source code)")

    connections.connect(
        "default",
        host=milvus_host,
        port=milvus_port,
        user=milvus_user,
        password=milvus_password,
    )

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

    schema = CollectionSchema(fields, SCHEMA_DESCRIPTION)

    collection_existed = utility.has_collection(collection_name)
    if collection_existed:
        collection = Collection(collection_name)
        existing_desc = collection.description or ""
        if f"v={SCHEMA_VERSION}" not in existing_desc:
            raise RuntimeError(
                f"Schema version mismatch for {collection_name}. "
                f"Expected v={SCHEMA_VERSION}, found description: '{existing_desc}'. "
                f"Run a migration job to drop+recreate before re-indexing."
            )
        print(f"Using existing collection: {collection_name} (schema v={SCHEMA_VERSION})")
    else:
        collection = Collection(collection_name, schema)
        print(f"Created new collection: {collection_name} (schema v={SCHEMA_VERSION})")

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

    if not records:
        raise RuntimeError(
            "No records were produced during ingestion. "
            "Aborting before Milvus reconciliation to prevent "
            "accidental deletion of existing documents."
        )

    current_file_ids = {r["file_unique_id"] for r in records}

    # Full rebuild reconciliation: remove Milvus documents that no longer exist
    # in the GitHub directory being indexed. Without this, deleted or renamed
    # files leave orphan vectors because the per-run delete only targets IDs
    # present in the current ingest records.
    if collection_existed and len(collection.indexes) > 0:
        collection.load()
        existing_file_ids = _collect_scoped_file_unique_ids(
            collection=collection,
            repo_name=repo_name,
            directory_path=directory_path,
        )
        orphan_ids = sorted(existing_file_ids - current_file_ids)

        print(f"Found {len(existing_file_ids)} existing documents in Milvus")
        print(f"Found {len(current_file_ids)} documents from GitHub")
        print(f"Removing {len(orphan_ids)} orphan documents")

        if orphan_ids:
            orphan_deleted = _delete_file_unique_ids(collection, orphan_ids, DELETE_BATCH_SIZE)
            collection.flush()
            print(f"Deleted {orphan_deleted} orphan chunks for {len(orphan_ids)} files")

    if records:
        # load() before delete requires an existing index; new collections have none yet
        if collection_existed and len(collection.indexes) > 0:
            unique_ids = sorted(current_file_ids)
            deleted = _delete_file_unique_ids(collection, unique_ids, DELETE_BATCH_SIZE)
            if deleted:
                collection.flush()
                print(f"Deleted {deleted} old chunks for {len(unique_ids)} files")

        # Insert new chunks (failure-aware)
        batch_size = 1000
        inserted = 0
        current_batch = None
        current_batch_start = 0
        try:
            for i in range(0, len(records), batch_size):
                current_batch_start = i
                current_batch = records[i:i + batch_size]
                collection.insert(current_batch)
                inserted += len(current_batch)
            current_batch = None
            collection.flush()
        except Exception as e:
            print(f"ERROR during insert. Inserted={inserted}/{len(records)}. Error: {e}")
            if current_batch is not None:
                failed_ids = sorted(set(r["file_unique_id"] for r in current_batch))
                print(f"Failed batch starts at record {current_batch_start}, file_unique_ids in failing batch: {failed_ids}")
            else:
                print("Failure occurred during flush() after all batches were submitted")
            raise

        # Create index if not already present
        if not collection.has_index():
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": min(1024, len(records))}
            }
            collection.create_index("vector", index_params, timeout=120)
        collection.load()
        print(f"Inserted {len(records)} records. Total: {collection.num_entities}")


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
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embeddings_service_url: str = (
        "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed"
    ),
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    milvus_host: str = "milvus-milvus.ml-infra.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = DOCS_COLLECTION,
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings_service_url=embeddings_service_url,
        embedding_batch_size=embedding_batch_size,
    )
    
    # Store in Milvus
    store_task = store_milvus(
        embedded_data=chunk_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
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


if __name__ == "__main__":
    import os
    # Set environment variable to disable caching by default
    os.environ['KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT'] = 'true'
    
    # Compile the pipeline with caching disabled by default
    kfp.compiler.Compiler().compile(
        pipeline_func=github_rag_pipeline,
        package_path="github_rag_pipeline.yaml"
    )