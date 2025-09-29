# Kubeflow Documentation RAG Pipelines

This directory contains Kubeflow Pipelines for processing Kubeflow documentation and building a Retrieval-Augmented Generation (RAG) system.

## 📁 Files Overview

- **`kubeflow-pipeline.py`** - Full rebuild pipeline (processes entire documentation corpus)
- **`incremental-pipeline.py`** - Incremental pipeline (processes only changed files)
- **`github_rag_pipeline.yaml`** - Compiled full pipeline
- **`github_rag_incremental_pipeline.yaml`** - Compiled incremental pipeline

## 🔄 Pipeline Overview

The pipelines download documentation from GitHub repositories, process the content, generate embeddings, and store them in Milvus vector database for semantic search capabilities.

---

## 🚀 Full Pipeline (`kubeflow-pipeline.py`)

**Purpose**: Complete processing of all documentation files in a repository.

### Components

1. **Download GitHub Directory** - Recursively fetches all `.md` and `.html` files from a specified directory
2. **Chunk and Embed** - Splits content into chunks and generates embeddings using sentence-transformers
3. **Store in Milvus** - Creates/updates Milvus collection with vector embeddings

### Key Features

- Aggressive content cleaning (removes Hugo frontmatter, HTML tags, navigation artifacts)
- Configurable chunk size and overlap
- Automatic citation URL generation
- GPU support for embedding generation
- Drops and recreates collection for clean rebuild

### Usage

```bash
python kubeflow-pipeline.py
```

### Parameters

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Default</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>repo_owner</code></td>
<td>"kubeflow"</td>
<td>GitHub repository owner</td>
</tr>
<tr>
<td><code>repo_name</code></td>
<td>"website"</td>
<td>GitHub repository name</td>
</tr>
<tr>
<td><code>directory_path</code></td>
<td>"content/en"</td>
<td>Documentation directory</td>
</tr>
<tr>
<td><code>github_token</code></td>
<td>""</td>
<td>GitHub API token (optional)</td>
</tr>
<tr>
<td><code>base_url</code></td>
<td>"https://www.kubeflow.org/docs"</td>
<td>Base URL for citations</td>
</tr>
<tr>
<td><code>chunk_size</code></td>
<td>1000</td>
<td>Text chunk size for embeddings</td>
</tr>
<tr>
<td><code>chunk_overlap</code></td>
<td>100</td>
<td>Overlap between chunks</td>
</tr>
<tr>
<td><code>milvus_host</code></td>
<td>"milvus-standalone-final.santhosh.svc.cluster.local"</td>
<td>Milvus server host</td>
</tr>
<tr>
<td><code>milvus_port</code></td>
<td>"19530"</td>
<td>Milvus server port</td>
</tr>
<tr>
<td><code>collection_name</code></td>
<td>"docs_rag"</td>
<td>Milvus collection name</td>
</tr>
</tbody>
</table>

### When to Use

- Initial setup of the RAG system
- Major documentation restructuring
- Periodic full rebuilds for cleanup
- When incremental updates are not feasible

---

## ⚡ Incremental Pipeline (`incremental-pipeline.py`)

**Purpose**: Process only changed files to update existing vector database efficiently.

### How Incremental Updates Work

1. **Detect Changed Files** - Receive a list of changed file paths
2. **Delete Old Vectors** - Remove existing embeddings for changed files from Milvus
3. **Download Specific Files** - Fetch only the changed files from GitHub
4. **Process & Embed** - Chunk and create embeddings for the new content
5. **Insert New Vectors** - Store updated embeddings in Milvus

### Components

1. **Delete Old Vectors** - Removes existing vectors for changed files
2. **Download Specific Files** - Fetches only the changed files from GitHub
3. **Chunk and Embed Incremental** - Processes only the changed files
4. **Store Incremental** - Adds new vectors to existing collection

### Key Features

- Efficient updates without full reprocessing
- Maintains collection integrity
- Handles file deletions and modifications
- Preserves existing data
- Creates collection if it doesn't exist

### Usage

```python
import json
from incremental_pipeline import github_rag_incremental_pipeline

# List of changed files
changed_files = [
    "content/en/docs/started/getting-started.md",
    "content/en/docs/components/pipelines/overview.md"
]

# Convert to JSON string (required by pipeline)
changed_files_json = json.dumps(changed_files)

# Compile pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=github_rag_incremental_pipeline,
    package_path="github_rag_incremental_pipeline.yaml"
)
```

### Parameters

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Default</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>repo_owner</code></td>
<td>"kubeflow"</td>
<td>GitHub repository owner</td>
</tr>
<tr>
<td><code>repo_name</code></td>
<td>"website"</td>
<td>GitHub repository name</td>
</tr>
<tr>
<td><code>changed_files</code></td>
<td>"[]"</td>
<td>JSON string of changed file paths</td>
</tr>
<tr>
<td><code>github_token</code></td>
<td>""</td>
<td>GitHub API token (optional but recommended)</td>
</tr>
<tr>
<td><code>base_url</code></td>
<td>"https://www.kubeflow.org/docs"</td>
<td>Base URL for citations</td>
</tr>
<tr>
<td><code>chunk_size</code></td>
<td>1200</td>
<td>Text chunk size for embeddings</td>
</tr>
<tr>
<td><code>chunk_overlap</code></td>
<td>100</td>
<td>Overlap between chunks</td>
</tr>
<tr>
<td><code>milvus_host</code></td>
<td>"milvus-standalone-final.santhosh.svc.cluster.local"</td>
<td>Milvus server host</td>
</tr>
<tr>
<td><code>milvus_port</code></td>
<td>"19530"</td>
<td>Milvus server port</td>
</tr>
<tr>
<td><code>collection_name</code></td>
<td>"docs_rag"</td>
<td>Milvus collection name</td>
</tr>
</tbody>
</table>

### Integration Patterns

#### Git Integration

```python
import subprocess
import json

def get_changed_files_from_git(since_commit="HEAD~1"):
    cmd = ["git", "diff", "--name-only", since_commit, "HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        # Filter for documentation files
        doc_files = [f for f in files if f.endswith(('.md', '.html'))]
        return json.dumps(doc_files)
    return "[]"
```

#### Webhook Integration

```python
def process_github_webhook(webhook_payload):
    all_changed_files = []
    
    for commit in webhook_payload.get("commits", []):
        all_changed_files.extend(commit.get("modified", []))
        all_changed_files.extend(commit.get("added", []))
    
    # Filter for documentation files
    doc_files = [f for f in set(all_changed_files) if f.endswith(('.md', '.html'))]
    return json.dumps(doc_files)
```

### When to Use

- Regular documentation updates
- CI/CD integration
- Real-time updates from webhooks
- Efficient processing of small changes

---

## 📊 Performance Comparison

<table>
<thead>
<tr>
<th>Metric</th>
<th>Full Pipeline</th>
<th>Incremental Pipeline</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Typical Files Processed</strong></td>
<td>~500-1000 files</td>
<td>1-10 files</td>
</tr>
<tr>
<td><strong>Processing Time</strong></td>
<td>15-30 minutes</td>
<td>1-3 minutes</td>
</tr>
<tr>
<td><strong>Resource Usage</strong></td>
<td>High (full rebuild)</td>
<td>Low (targeted update)</td>
</tr>
<tr>
<td><strong>Milvus Operations</strong></td>
<td>Drop + Recreate</td>
<td>Delete + Insert</td>
</tr>
<tr>
<td><strong>Suitable For</strong></td>
<td>Initial setup, major changes</td>
<td>Regular updates, CI/CD</td>
</tr>
</tbody>
</table>

## 🔍 Monitoring & Debugging

### Check Pipeline Status

```python
# After submitting the pipeline
run = client.get_run(run_id)
print(f"Status: {run.run.status}")

# Get component logs
for component in run.run.pipeline_spec.components:
    logs = client.get_run_logs(run_id, component.name)
    print(f"{component.name}: {logs}")
```

### Verify Milvus Updates

```python
from pymilvus import connections, Collection

connections.connect("default", host="your-milvus-host", port="19530")
collection = Collection("docs_rag")

# Check total count
print(f"Total vectors: {collection.num_entities}")

# Check specific file
file_id = "website:content/en/docs/your-file.md"
results = collection.query(
    expr=f'file_unique_id == "{file_id}"',
    output_fields=["chunk_index", "last_updated"]
)
print(f"Chunks for {file_id}: {len(results)}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Collection doesn't exist"** - The incremental pipeline creates the collection if needed
2. **"No files to process"** - Check that file paths are correct and files exist
3. **"GitHub API rate limit"** - Use a GitHub token for higher limits
4. **"Milvus connection failed"** - Verify host, port, and network connectivity

### Debug Mode

```python
# In any component
import logging
logging.basicConfig(level=logging.DEBUG)
print(f"Debug: Processing {len(file_paths_list)} files")
```

## 🔄 Migration Strategy

1. **First Time Setup**: Run the full pipeline once to create the initial collection
2. **Switch to Incremental**: Use the incremental pipeline for subsequent updates
3. **Periodic Full Rebuilds**: Optionally run full pipeline monthly/quarterly for cleanup

## Requirements

- Kubeflow Pipelines
- Milvus vector database
- GPU nodes (for embedding generation)
- GitHub token (optional, for private repositories)
