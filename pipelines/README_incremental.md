# Incremental RAG Pipeline

This directory contains an incremental version of the RAG (Retrieval-Augmented Generation) pipeline that processes only changed files instead of rebuilding the entire documentation corpus.

## 📁 Files Overview

- **`kubeflow-pipeline.py`** - Original full rebuild pipeline
- **`incremental-pipeline.py`** - New incremental pipeline (processes only changed files)
- **`example_incremental_usage.py`** - Usage examples and integration patterns
- **`README_incremental.md`** - This documentation

## 🔄 How Incremental Updates Work

The incremental pipeline follows this process:

1. **Detect Changed Files** - Receive a list of changed file paths
2. **Delete Old Vectors** - Remove existing embeddings for changed files from Milvus
3. **Download Specific Files** - Fetch only the changed files from GitHub
4. **Process & Embed** - Chunk and create embeddings for the new content
5. **Insert New Vectors** - Store updated embeddings in Milvus

## 🚀 Quick Start

### 1. Basic Usage

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

### 2. Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `repo_owner` | "kubeflow" | GitHub repository owner |
| `repo_name` | "website" | GitHub repository name |
| `changed_files` | "[]" | JSON string of changed file paths |
| `github_token` | "" | GitHub API token (optional but recommended) |
| `base_url` | "https://www.kubeflow.org/docs" | Base URL for citations |
| `chunk_size` | 1000 | Text chunk size for embeddings |
| `chunk_overlap` | 100 | Overlap between chunks |
| `milvus_host` | "milvus-standalone-final.santhosh.svc.cluster.local" | Milvus server host |
| `milvus_port` | "19530" | Milvus server port |
| `collection_name` | "docs_rag" | Milvus collection name |

## 🔧 Integration Patterns

### Git Integration

Automatically detect changed files from git:

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

# Use in pipeline
changed_files_json = get_changed_files_from_git()
```

### Webhook Integration

Process GitHub webhook payloads:

```python
def process_github_webhook(webhook_payload):
    all_changed_files = []
    
    for commit in webhook_payload.get("commits", []):
        all_changed_files.extend(commit.get("modified", []))
        all_changed_files.extend(commit.get("added", []))
        # Handle removed files separately if needed
    
    # Filter for documentation files
    doc_files = [f for f in set(all_changed_files) if f.endswith(('.md', '.html'))]
    return json.dumps(doc_files)
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Incremental RAG Update
on:
  push:
    paths:
      - 'content/en/docs/**/*.md'
      - 'content/en/docs/**/*.html'

jobs:
  update-rag:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 2  # Need previous commit for diff
      
      - name: Get changed files
        id: changed-files
        run: |
          CHANGED=$(git diff --name-only HEAD~1 HEAD | grep -E '\.(md|html)$' | jq -R -s -c 'split("\n")[:-1]')
          echo "files=$CHANGED" >> $GITHUB_OUTPUT
      
      - name: Trigger Kubeflow Pipeline
        run: |
          # Submit pipeline with changed files
          # Implementation depends on your KFP setup
```

## 📊 Component Details

### 1. `download_specific_files`

Downloads only the specified files from GitHub instead of the entire directory tree.

**Key Features:**
- Accepts JSON array of file paths
- Skips non-documentation files
- Handles API rate limits gracefully
- Supports both public and private repositories

### 2. `delete_old_vectors`

Removes existing vectors for changed files from Milvus before inserting new ones.

**Key Features:**
- Deletes by `file_unique_id` (format: `{repo_name}:{file_path}`)
- Handles missing collections gracefully
- Provides detailed logging of deletion counts
- Ensures clean updates without duplicates

### 3. `chunk_and_embed_incremental`

Same processing logic as the original pipeline but optimized for smaller batches.

**Key Features:**
- Identical text cleaning and chunking logic
- GPU acceleration when available
- Efficient processing of small file sets
- Detailed chunk statistics

### 4. `store_milvus_incremental`

Inserts new vectors without dropping the entire collection.

**Key Features:**
- Creates collection if it doesn't exist
- Preserves existing data
- Handles indexing efficiently
- Provides collection size statistics

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

## ⚡ Performance Comparison

| Metric | Full Pipeline | Incremental Pipeline |
|--------|---------------|---------------------|
| **Typical Files Processed** | ~500-1000 files | 1-10 files |
| **Processing Time** | 15-30 minutes | 1-3 minutes |
| **Resource Usage** | High (full rebuild) | Low (targeted update) |
| **Milvus Operations** | Drop + Recreate | Delete + Insert |
| **Suitable For** | Initial setup, major changes | Regular updates, CI/CD |

## 🛠️ Troubleshooting

### Common Issues

1. **"Collection doesn't exist"** - The incremental pipeline creates the collection if needed
2. **"No files to process"** - Check that file paths are correct and files exist
3. **"GitHub API rate limit"** - Use a GitHub token for higher limits
4. **"Milvus connection failed"** - Verify host, port, and network connectivity

### Debug Mode

Add debug logging to components:

```python
# In any component
import logging
logging.basicConfig(level=logging.DEBUG)
print(f"Debug: Processing {len(file_paths_list)} files")
```

## 🔄 Migration from Full Pipeline

1. **First Time Setup**: Run the full pipeline once to create the initial collection
2. **Switch to Incremental**: Use the incremental pipeline for subsequent updates
3. **Periodic Full Rebuilds**: Optionally run full pipeline monthly/quarterly for cleanup

## 📈 Future Enhancements

- **Batch Processing**: Handle large sets of changed files efficiently
- **Retry Logic**: Automatic retry for failed file downloads
- **Metrics Collection**: Detailed performance and success metrics
- **File Deletion Handling**: Remove vectors when files are deleted from repo
- **Multi-Repository Support**: Process changes from multiple repositories

## 🤝 Contributing

To improve the incremental pipeline:

1. Test with your specific repository structure
2. Add error handling for edge cases
3. Optimize for your Milvus configuration
4. Share performance improvements

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review component logs in Kubeflow Pipelines UI
- Verify Milvus collection status and connectivity
