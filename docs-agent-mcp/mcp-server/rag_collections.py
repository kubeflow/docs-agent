"""Milvus collection and field names.

Milvus allows only letters, numbers, and underscores (no hyphens).
"""

DOCS_COLLECTION = "kubeflow_docs"
ISSUES_COLLECTION = "issues_rag"
CODE_COLLECTION = "code_rag"

DENSE_FIELD = "vector"
SPARSE_FIELD = "sparse_vector"
BM25_INPUT_FIELD = "content_text"
DENSE_DIM = 768

# Older names some imports still use.
HYBRID_DENSE_FIELD = DENSE_FIELD
HYBRID_SPARSE_FIELD = SPARSE_FIELD
HYBRID_BM25_INPUT_FIELD = BM25_INPUT_FIELD
HYBRID_DENSE_DIM = DENSE_DIM
