# Agent Guide: Kubeflow Documentation AI Assistant

## Purpose

- **Who this is for**: AI agents and developers working inside this repo.
- **What you get**: The minimum set of facts, files, and commands to navigate, modify, and run the docs-agent locally.

### Document metadata

- Last updated: 2026-02-14
- Scope: docs-agent (RAG-based documentation assistant), Python (server/Pipelines), YAML (manifests)

### Maintenance (agents and contributors)

- If you change commands, file paths, environment variables, or workflows in this repo, update this guide in the relevant sections.
- When you add new components or change the architecture, update the "Architecture" section.
- If you come across new common errors or fixes, extend "Troubleshooting and pitfalls".
- Always bump the "Last updated" date above when you make substantive changes.

## Architecture

The docs-agent is a Retrieval-Augmented Generation (RAG) system for Kubeflow documentation. It consists of three main components:

### High-Level Architecture

1. **API Servers** (`server/` and `server-https/`)
   - WebSocket API (`server/app.py`): Real-time chat with bidirectional communication
   - HTTPS API (`server-https/app.py`): RESTful endpoints with streaming (SSE) support

2. **Kubeflow Pipelines** (`pipelines/`)
   - ETL pipeline for fetching, chunking, embedding, and storing documentation in Milvus

3. **Infrastructure**
   - Milvus Vector Database: Stores document embeddings for semantic search
   - KServe: Serves the LLM (Llama 3.1-8B) for inference

### Data Flow

- User query → API Server → Embed query → Milvus similarity search → Return context → LLM generates response
- Documentation ingestion: GitHub → Kubeflow Pipeline → Chunk & Embed → Milvus storage

## Key Paths and Files

- WebSocket API: `server/app.py`
- HTTPS API: `server-https/app.py`
- RAG Pipeline: `pipelines/kubeflow-pipeline.py`
- Pipeline Requirements: `pipelines/requirements.txt`
- Server Requirements: `server/requirements.txt`
- HTTPS Server Requirements: `server-https/requirements.txt` (check if exists, or use server/requirements.txt)
- Milvus Manifests: `manifests/milvus-deployment.yaml`
- KServe Serving Runtime: `manifests/serving-runtime.yaml`
- Frontend Assets: `docs_scripts/chatbot.js`, `docs_styles/chatbot.css`

## Local Development Setup

### Prerequisites

- Python 3.9+
- Kubernetes cluster (for full deployment)
- Kubeflow Pipelines installed
- Milvus vector database
- KServe with LLM model

### Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel

# Install server dependencies
pip install -r server/requirements.txt

# Install pipeline dependencies
pip install -r pipelines/requirements.txt
```

### Required Environment Variables

#### For API Servers

```bash
# KServe (LLM endpoint)
export KSERVE_URL="http://llama.santhosh.svc.cluster.local/openai/v1/chat/completions"
export MODEL="llama3.1-8B"
export PORT="8000"

# Milvus
export MILVUS_HOST="my-release-milvus.santhosh.svc.cluster.local"
export MILVUS_PORT="19530"
export MILVUS_COLLECTION="docs_rag"
export MILVUS_VECTOR_FIELD="vector"
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
```

#### For Kubeflow Pipeline

```bash
# Pipeline parameters (set in KFP UI or CLI)
repo_owner="kubeflow"
repo_name="website"
directory_path="content/en"
github_token=""
base_url="https://www.kubeflow.org/docs"
chunk_size=1000
chunk_overlap=100
milvus_host="my-release-milvus.santhosh.svc.cluster.local"
milvus_port="19530"
collection_name="docs_rag"
```

## Local Execution

### Starting the API Servers

#### WebSocket API

```bash
# From project root
python server/app.py
```

The WebSocket server:
- Runs on port 8000 (configurable via PORT env var)
- Accepts WebSocket connections at `ws://localhost:8000`
- Provides health check at `http://localhost:8000/health`

#### HTTPS API

```bash
# From project root
python server-https/app.py
```

The HTTPS server:
- Runs on port 8000 (configurable via PORT env var)
- Main endpoint: `POST /chat`
- Health check: `GET /health`
- Supports both streaming (SSE) and non-streaming responses

### Running the Pipeline

```bash
# Compile the pipeline
python pipelines/kubeflow-pipeline.py

# This generates: github_rag_pipeline.yaml

# Upload and run via KFP CLI or UI
kfp pipeline upload -p "GitHub RAG Pipeline" github_rag_pipeline.yaml
```

## API Reference

### HTTPS API Endpoints

#### POST /chat

Main chat endpoint with RAG capabilities.

**Request:**
```json
{
    "message": "How do I create a Kubeflow pipeline?",
    "stream": true  // or false for non-streaming
}
```

**Streaming Response (SSE):**
```
data: {"type": "content", "content": "response text"}
data: {"type": "tool_result", "tool_name": "search_kubeflow_docs", "content": "search results"}
data: {"type": "citations", "citations": ["url1", "url2"]}
data: {"type": "done"}
```

**Non-streaming Response:**
```json
{
    "response": "Complete response text",
    "citations": ["url1", "url2"]
}
```

#### GET /health

Health check endpoint for Kubernetes probes.

**Response:**
```json
{
    "status": "healthy",
    "service": "https-api"
}
```

#### OPTIONS /chat, OPTIONS /, OPTIONS /health

Preflight CORS endpoints for browser requests.

**Response:**
```json
{
    "message": "OK"
}
```

### WebSocket API

Connect to `ws://localhost:8000` and send JSON messages:

```json
{
    "message": "What is KServe?"
}
```

**Response types:**
- `system`: Welcome message
- `content`: Streaming response chunks
- `tool_result`: Tool execution results
- `citations`: Source citations
- `done`: Response complete
- `error`: Error messages

## Key Components

### System Prompt

Both API servers use a shared system prompt defined in the code:

- Role: Kubeflow Documentation Assistant
- Tool: `search_kubeflow_docs` - searches official Kubeflow docs
- Routing logic: Determines when to use tools vs. direct answers
- Style: Concise, markdown-formatted responses

### Milvus Search

The `milvus_search()` function:
1. Connects to Milvus using environment variables
2. Encodes query using sentence-transformers
3. Performs cosine similarity search
4. Returns top-k results with citations

### Tool Calling

The system uses automatic tool calling:
1. LLM decides when to search documentation
2. Tool is executed against Milvus
3. Results are fed back to LLM
4. Final response includes citations

## Milvus Schema

The collection schema for storing documentation:

```python
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
```

- Vector dimension: 768 (for sentence-transformers/all-mpnet-base-v2)
- Index type: IVF_FLAT with COSINE metric

## Kubeflow Pipeline Components

### 1. download_github_directory

Fetches documentation files from GitHub repositories.

- Input: repo_owner, repo_name, directory_path, github_token
- Output: JSON Lines file with file paths and content

### 2. chunk_and_embed

Processes text and creates embeddings.

- Input: Raw documentation data, chunk_size, chunk_overlap
- Output: JSON Lines with embeddings and metadata
- Cleaning: Removes Hugo frontmatter, HTML, URLs, etc.

### 3. store_milvus

Stores vectors in Milvus.

- Input: Embedded data, milvus_host, milvus_port, collection_name
- Creates/drops collection, inserts in batches, creates index

## Troubleshooting and Pitfalls

### Common Issues

1. **Milvus Connection Errors**
   - Verify Milvus is running: `kubectl get pods -n <namespace> | grep milvus`
   - Check service: `kubectl get svc -n <namespace> | grep milvus`
   - Test connection: `python -c "from pymilvus import connections; connections.connect('default', host='<host>', port='19530')"`

2. **KServe Connection Errors**
   - Verify InferenceService: `kubectl get inferenceservice -n <namespace>`
   - Check logs: `kubectl logs -f deployment/<llm-deployment>`
   - Verify URL format: `http://<service>.<namespace>.svc.cluster.local/openai/v1/chat/completions`

3. **Pipeline Execution Errors**
   - RBAC issues: Ensure KFP service account has proper permissions
   - GPU availability: Verify GPU nodes are available for embedding step

4. **WebSocket/HTTPS Connection Issues**
   - SSL certificate errors: Ensure proper certificates for production
   - CORS issues: Check allow_origins configuration

### Quick Debug Commands

```bash
# Check Milvus status
kubectl get pods -n <namespace> | grep milvus

# Check KServe status
kubectl get inferenceservice -n <namespace>

# Check API server logs
kubectl logs -f deployment/docs-assistant-api

# Test Milvus connection
python -c "from pymilvus import connections; connections.connect('default', host='<host>', port='19530'); print('Connected!')"

# Test API health
curl http://localhost:8000/health
```

## Key Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| KSERVE_URL | http://llama.santhosh.svc.cluster.local/openai/v1/chat/completions | LLM endpoint |
| MODEL | llama3.1-8B | Model name |
| PORT | 8000 | Server port |
| MILVUS_HOST | my-release-milvus.santhosh.svc.cluster.local | Milvus host |
| MILVUS_PORT | 19530 | Milvus port |
| MILVUS_COLLECTION | docs_rag | Collection name |
| EMBEDDING_MODEL | sentence-transformers/all-mpnet-base-v2 | Embedding model |

## Quick Reference

### Essential Commands

- Start WebSocket server: `python server/app.py`
- Start HTTPS server: `python server-https/app.py`
- Compile pipeline: `python pipelines/kubeflow-pipeline.py`
- Test chat (streaming): `curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "What is KServe?", "stream": true}'`
- Test chat (non-streaming): `curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message": "What is KServe?", "stream": false}'`

### Key Files

- WebSocket API: `server/app.py`
- HTTPS API: `server-https/app.py`
- RAG Pipeline: `pipelines/kubeflow-pipeline.py`
- Incremental Pipeline: `pipelines/incremental-pipeline.py`

**Dependencies**

**WebSocket Server (server/):**
- websockets, httpx, pymilvus, sentence-transformers, torch, numpy

**HTTPS Server (server-https/):**
- fastapi, uvicorn[standard], pydantic, httpx, sentence-transformers, pymilvus, torch, numpypipelines/):

**Pipeline (**
- kfp, requests, beautifulsoup4, sentence-transformers, langchain, torch, pymilvus, numpy
