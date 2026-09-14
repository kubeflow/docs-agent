"""
Shared embedding utilities for docs-agent ingestion pipelines.

Supports multiple embedding backends:
  - sentence-transformers (local, default for development)
  - openai (API-based, for production)

Configure via environment variables:
  EMBEDDING_MODEL: Model name/path (default: sentence-transformers/all-MiniLM-L6-v2)
  OPENAI_API_KEY: Required only when EMBEDDING_MODEL=openai
"""

import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_embedding_model_name() -> str:
    """Get the configured embedding model name from environment."""
    return os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


def get_embedding_dimension() -> int:
    """Return the embedding dimension for the configured model.

    Returns:
        int: Vector dimension size.
    """
    model_name = get_embedding_model_name()
    dimension_map = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "nomic-embed-text": 768,
        "openai": 1536,
        "text-embedding-3-small": 1536,
    }
    for key, dim in dimension_map.items():
        if key in model_name:
            return dim
    # Default fallback
    logger.warning(
        "Unknown model '%s', defaulting to 384 dimensions.", model_name
    )
    return 384


class EmbeddingClient:
    """Unified embedding client supporting local and API-based models.

    Usage:
        client = EmbeddingClient()
        vectors = client.embed_batch(["hello world", "kubeflow pipelines"])
    """

    def __init__(self, model_name: Optional[str] = None, batch_size: int = 32):
        """Initialize the embedding client.

        Args:
            model_name: Override for EMBEDDING_MODEL env var.
            batch_size: Number of texts to embed per batch.
        """
        self.model_name = model_name or get_embedding_model_name()
        self.batch_size = batch_size
        self._model = None
        self._client = None

        logger.info("Embedding client initialized with model: %s", self.model_name)

    def _is_openai(self) -> bool:
        """Check if using OpenAI API backend."""
        return "openai" in self.model_name or "text-embedding" in self.model_name

    def _load_local_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_path = self.model_name
            # Strip the prefix if it's a sentence-transformers model
            if "/" in model_path and not model_path.startswith("/"):
                pass  # Use full HuggingFace path
            logger.info("Loading local model: %s", model_path)
            self._model = SentenceTransformer(model_path)
            logger.info("Model loaded successfully.")
        return self._model

    def _get_openai_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required "
                    "when using OpenAI embeddings."
                )
            self._client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized.")
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts with automatic batching and retry.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)

            logger.info(
                "Embedded batch %d/%d (%d texts)",
                batch_num,
                total_batches,
                len(batch),
            )

        return all_embeddings

    def _embed_batch_with_retry(
        self, texts: List[str], max_retries: int = 3
    ) -> List[List[float]]:
        """Embed a single batch with exponential backoff retry.

        Args:
            texts: Batch of texts to embed.
            max_retries: Maximum number of retry attempts.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        for attempt in range(max_retries):
            try:
                if self._is_openai():
                    return self._embed_openai(texts)
                else:
                    return self._embed_local(texts)
            except Exception as e:
                wait_time = (2 ** attempt) + (0.1 * attempt)
                logger.warning(
                    "Embedding failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    str(e),
                    wait_time,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Embedding failed after {max_retries} attempts: {e}"
                    ) from e
        return []  # unreachable, but satisfies type checker

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Embed using local sentence-transformers model.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors.
        """
        model = self._load_local_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed using OpenAI API.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors.
        """
        client = self._get_openai_client()
        model_name = self.model_name
        if "openai" in model_name and "text-embedding" not in model_name:
            model_name = "text-embedding-3-small"

        response = client.embeddings.create(input=texts, model=model_name)
        return [item.embedding for item in response.data]


# Convenience function
def embed_texts(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """Convenience function to embed texts with default settings.

    Args:
        texts: List of texts to embed.
        model_name: Optional model override.

    Returns:
        List of embedding vectors.
    """
    client = EmbeddingClient(model_name=model_name)
    return client.embed_texts(texts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke test
    test_texts = [
        "How to install Kubeflow on Kubernetes",
        "KFP pipeline component decorator",
        "Milvus vector database schema design",
    ]
    logger.info("Testing embedding with model: %s", get_embedding_model_name())
    logger.info("Expected dimensions: %d", get_embedding_dimension())

    client = EmbeddingClient()
    vectors = client.embed_texts(test_texts)
    for i, (text, vec) in enumerate(zip(test_texts, vectors)):
        logger.info("Text %d: '%s...' -> dim=%d", i, text[:40], len(vec))
