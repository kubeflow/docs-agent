# server/vector_services/embedding.py

from functools import lru_cache
import os
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np


EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


@lru_cache(maxsize=1)
def get_sentence_transformer() -> SentenceTransformer:
    """
    Return a shared SentenceTransformer instance.

    Uses an LRU cache to ensure a single model instance per process.
    """
    model_name = os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)
    model = SentenceTransformer(model_name)
    return model


@lru_cache(maxsize=8192)
def _embed_text_cached_single(text: str) -> np.ndarray:
    """
    Embed a single text with caching.

    Internal helper; use embed_text or embed_texts from callers.
    """
    model = get_sentence_transformer()
    emb = model.encode([text], convert_to_numpy=True)[0]
    return emb


def embed_text(text: str) -> List[float]:
    """
    Public helper for embedding a single text as a Python list[float].
    """
    emb = _embed_text_cached_single(text)
    return emb.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts.

    Uses the shared model instance; for repeated single texts, the
    internal cache will avoid recomputation.
    """
    if not texts:
        return []

    model = get_sentence_transformer()
    embs = model.encode(texts, convert_to_numpy=True)
    if isinstance(embs, np.ndarray):
        return [row.tolist() for row in embs]

    # Fallback if encode returns a list-like object
    return [np.asarray(row).tolist() for row in embs]