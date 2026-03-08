"""RAG utilities for Milvus search and tool execution."""

import json
from typing import Dict, Any, List, Tuple

from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

from .config import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    MILVUS_VECTOR_FIELD,
    EMBEDDING_MODEL,
)


def milvus_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Execute a semantic search in Milvus and return structured JSON serializable results."""
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

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


async def execute_tool(tool_call: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Execute a tool call and return the result and citations."""
    try:
        function_name = tool_call.get("function", {}).get("name")
        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

        if function_name == "search_kubeflow_docs":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)

            print(f"[TOOL] Executing Milvus search for: '{query}' (top_k={top_k})")
            result = milvus_search(query, 15)

            citations = []
            formatted_results = []

            for hit in result.get("results", []):
                citation_url = hit.get("citation_url", "")
                if citation_url and citation_url not in citations:
                    citations.append(citation_url)

                formatted_results.append(
                    f"File: {hit.get('file_path', 'Unknown')}\n"
                    f"Content: {hit.get('content_text', '')}\n"
                    f"URL: {citation_url}\n"
                    f"Similarity: {hit.get('similarity', 0):.3f}\n"
                )

            formatted_text = "\n".join(formatted_results) if formatted_results else "No relevant results found."
            return formatted_text, citations

        return f"Unknown tool: {function_name}", []

    except Exception as e:
        print(f"[ERROR] Tool execution failed: {e}")
        return f"Tool execution failed: {e}", []
