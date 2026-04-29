import argparse
import os
from typing import Any, Dict, List

from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

from shared.reranking import candidate_pool_limit, load_rerank_config_from_env, rerank_documents


DEFAULT_QUERIES = [
    "How do I create a Kubeflow Pipeline?",
    "How to deploy an InferenceService in KServe?",
    "Kubeflow Notebook setup requirements",
]


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval results before/after reranking.")
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES, help="Queries to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Final top-k results to keep")
    parser.add_argument(
        "--show-content-chars",
        type=int,
        default=180,
        help="Number of content characters to print per result",
    )
    return parser.parse_args()


def _print_docs(title: str, docs: List[Dict[str, Any]], show_content_chars: int) -> None:
    print(f"\n{title}")
    if not docs:
        print("  (no results)")
        return

    for idx, doc in enumerate(docs, start=1):
        content = (doc.get("content_text") or "").replace("\n", " ").strip()
        if len(content) > show_content_chars:
            content = content[:show_content_chars] + "..."

        print(
            f"  {idx}. score={doc.get('rerank_score', doc.get('similarity', 0.0)):.4f} "
            f"sim={doc.get('similarity', 0.0):.4f} "
            f"keyword={doc.get('keyword_score', 0.0):.4f} "
            f"metadata={doc.get('metadata_score', 0.0):.4f}"
        )
        print(f"     file={doc.get('file_path', '')}")
        print(f"     url={doc.get('citation_url', '')}")
        print(f"     text={content}")


def retrieve_candidates(
    query: str,
    model: SentenceTransformer,
    collection: Collection,
    top_k: int,
    candidate_limit: int,
    vector_field: str,
) -> List[Dict[str, Any]]:
    query_vec = model.encode(query).tolist()
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}

    results = collection.search(
        data=[query_vec],
        anns_field=vector_field,
        param=search_params,
        limit=candidate_limit,
        output_fields=["file_path", "content_text", "citation_url"],
    )

    docs: List[Dict[str, Any]] = []
    for hit in results[0]:
        entity = hit.entity
        docs.append(
            {
                "similarity": 1.0 - float(hit.distance),
                "file_path": entity.get("file_path"),
                "citation_url": entity.get("citation_url"),
                "content_text": entity.get("content_text") or "",
            }
        )

    return docs


def main() -> None:
    args = build_args()

    milvus_host = os.getenv("MILVUS_HOST", "my-release-milvus.docs-agent.svc.cluster.local")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    milvus_collection = os.getenv("MILVUS_COLLECTION", "docs_rag")
    milvus_vector_field = os.getenv("MILVUS_VECTOR_FIELD", "vector")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

    rerank_config = load_rerank_config_from_env()
    requested_top_k = max(1, int(args.top_k))
    candidate_limit = candidate_pool_limit(requested_top_k, rerank_config)

    print("Retrieval evaluation configuration")
    print(f"- collection: {milvus_collection}")
    print(f"- top_k: {requested_top_k}")
    print(f"- candidate_limit: {candidate_limit}")
    print(f"- rerank_enabled: {rerank_config.enabled}")

    connections.connect(alias="default", host=milvus_host, port=milvus_port)
    try:
        collection = Collection(milvus_collection)
        collection.load()
        model = SentenceTransformer(embedding_model_name)

        for query in args.queries:
            print("\n" + "=" * 100)
            print(f"Query: {query}")

            candidates = retrieve_candidates(
                query=query,
                model=model,
                collection=collection,
                top_k=requested_top_k,
                candidate_limit=candidate_limit,
                vector_field=milvus_vector_field,
            )

            before_docs = candidates[:requested_top_k]
            after_docs = rerank_documents(
                query=query,
                docs=candidates,
                config=rerank_config,
                top_k=requested_top_k,
            )

            _print_docs("Before reranking", before_docs, args.show_content_chars)
            _print_docs("After reranking", after_docs, args.show_content_chars)
    finally:
        connections.disconnect(alias="default")


if __name__ == "__main__":
    main()
