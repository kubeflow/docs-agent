import os
from dotenv import load_dotenv

load_dotenv()

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "docs_rag")
MILVUS_VECTOR_FIELD = "vector"

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# LLM Configuration (for Ragas and Synthetic Dataset Generation)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
EVAL_MODEL = os.getenv("EVAL_MODEL", "llama3.1-8B-instant")

# Evaluation Settings
TOP_K = 10
TEMPERATURE = 0.0
TOP_P = 0.95
MAX_TOKENS = 1024
DATASET_PATH = "evaluation/dataset.jsonl"
RESULTS_PATH = "evaluation/results.csv"
