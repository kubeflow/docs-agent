import os
import json
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# We will need config here. 
# Ideally, we should have a config.py, but for now I'll duplicate or import if possible.
# To avoid circular imports with app.py, let's just read env vars here too or create a config module.
# Let's create a simple config approach.

MILVUS_HOST = os.getenv("MILVUS_HOST", "my-release-milvus.santhosh.svc.cluster.local")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "docs_rag")
MILVUS_VECTOR_FIELD = os.getenv("MILVUS_VECTOR_FIELD", "vector")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

def milvus_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Execute a semantic search in Milvus and return structured JSON serializable results."""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        # Encoder (same model as pipeline)
        # TODO: optimization - load model once globally if possible, or use a service
        encoder = SentenceTransformer(EMBEDDING_MODEL)
        query_vec = encoder.encode(query).tolist()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
        results = collection.search(
            data=[query_vec],
            anns_field=MILVUS_VECTOR_FIELD,
            param=search_params,
            limit=int(top_k),
            output_fields=["file_path", "content_text", "citation_url"],
        )

        hits = []
        for hit in results[0]:
            # similarity = 1 - distance for COSINE in Milvus
            similarity = 1.0 - float(hit.distance)
            entity = hit.entity
            content_text = entity.get("content_text") or ""
            if isinstance(content_text, str) and len(content_text) > 400:
                content_text = content_text[:400] + "..."
            hits.append({
                "similarity": similarity,
                "file_path": entity.get("file_path"),
                "citation_url": entity.get("citation_url"),
                "content_text": content_text,
            })
        return {"results": hits}
    except Exception as e:
        print(f"[ERROR] Milvus search failed: {e}")
        return {"results": []}
    finally:
        try:
            connections.disconnect(alias="default")
        except Exception:
            pass

# Tool definition for LangChain/LangGraph
from langchain_core.tools import tool

@tool
def search_kubeflow_docs(query: str) -> str:
    """
    Search the official Kubeflow documentation.
    Use this tool to find information about Kubeflow components, installation, configuration,
    APIs, SDKs, and troubleshooting.
    """
    # Default top_k to 5 inside
    try:
        print(f"[TOOL] Searching docs for: {query}")
        result = milvus_search(query, top_k=5)
        hits = result.get("results", [])
        
        if not hits:
            return "No relevant documentation found."
            
        formatted_results = []
        for hit in hits:
            formatted_results.append(
                f"Source: {hit.get('citation_url', 'Unknown')}\n"
                f"Content: {hit.get('content_text', '')}\n"
            )
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching docs: {str(e)}"

# GitHub Tool
import requests

@tool
def search_github_issues(query: str) -> str:
    """
    Search Kubeflow GitHub issues and pull requests.
    Use this tool when the user asks about bugs, feature requests, specific errors 
    that might be discussed in issues, or recent changes.
    """
    try:
        # Use GitHub Search API
        url = "https://api.github.com/search/issues"
        params = {
            "q": f"org:kubeflow {query}",
            "sort": "updated",
            "per_page": 5
        }
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Add token if available
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"
            
        print(f"[TOOL] Searching GitHub issues for: {query}")
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            return "No relevant GitHub issues found."
            
        formatted_results = []
        for item in items:
            formatted_results.append(
                f"Title: {item.get('title')}\n"
                f"URL: {item.get('html_url')}\n"
                f"State: {item.get('state')}\n"
                f"Summary: {item.get('body', '')[:200]}...\n"
            )
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching GitHub issues: {str(e)}"
