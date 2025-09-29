# Kubeflow Documentation RAG Pipelines

This directory contains Kubeflow Pipelines for processing Kubeflow documentation and building a Retrieval-Augmented Generation (RAG) system.

## Overview

The pipelines download documentation from GitHub repositories, process the content, generate embeddings, and store them in Milvus vector database for semantic search capabilities.

## Full Pipeline (`kubeflow-pipeline.py`)

**Purpose**: Complete processing of all documentation files in a repository.

**Components**:
1. **Download GitHub Directory** - Recursively fetches all `.md` and `.html` files from a specified directory
2. **Chunk and Embed** - Splits content into chunks and generates embeddings using sentence-transformers
3. **Store in Milvus** - Creates/updates Milvus collection with vector embeddings

**Key Features**:
- Aggressive content cleaning (removes Hugo frontmatter, HTML tags, navigation artifacts)
- Configurable chunk size and overlap
- Automatic citation URL generation
- GPU support for embedding generation

**Usage**:
```bash
python kubeflow-pipeline.py
```

**Parameters**:
- `repo_owner`: GitHub repository owner (default: "kubeflow")
- `repo_name`: Repository name (default: "website")
- `directory_path`: Documentation directory (default: "content/en")
- `chunk_size`: Text chunk size (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 100)
- `milvus_host`: Milvus server host
- `collection_name`: Milvus collection name (default: "docs_rag")

## Incremental Pipeline (`incremental-pipeline.py`)

**Purpose**: Process only changed files to update existing vector database efficiently.

**Components**:
1. **Delete Old Vectors** - Removes existing vectors for changed files
2. **Download Specific Files** - Fetches only the changed files from GitHub
3. **Chunk and Embed Incremental** - Processes only the changed files
4. **Store Incremental** - Adds new vectors to existing collection

**Key Features**:
- Efficient updates without full reprocessing
- Maintains collection integrity
- Handles file deletions and modifications
- Preserves existing data

**Usage**:
```bash
python incremental-pipeline.py
```

**Parameters**:
- `changed_files`: JSON string of file paths that changed
- All other parameters same as full pipeline

## Pipeline Output

Both pipelines generate a YAML file that can be deployed to Kubeflow Pipelines:
- Full pipeline: `github_rag_pipeline.yaml`
- Incremental pipeline: `github_rag_incremental_pipeline.yaml`

## Requirements

- Kubeflow Pipelines
- Milvus vector database
- GPU nodes (for embedding generation)
- GitHub token (optional, for private repositories)
