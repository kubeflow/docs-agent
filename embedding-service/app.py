"""Dedicated embedding service for docs-agent (ADR-004).

Loads the embedding model ONCE at startup and exposes an HTTP endpoint.
All components (pipelines, servers) call this service instead of loading
the 2 GB model locally.
"""

import os
import sys

# Force UTF-8 encoding internally for Windows environments
sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(
    title="Embedding Service",
    description="Centralised embedding endpoint for docs-agent (ADR-004)",
    version="1.0.0",
)

# ── Load model ONCE at startup ──────────────────────────────────────────
model = SentenceTransformer(MODEL_NAME)
print(
    f"Model loaded: {MODEL_NAME} "
    f"(dim={model.get_sentence_embedding_dimension()})"
)


# ── Request / Response schemas ──────────────────────────────────────────
class EmbedRequest(BaseModel):
    texts: list[str]  # batch of texts to embed


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int


# ── Endpoints ───────────────────────────────────────────────────────────
@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """Return embeddings for a batch of texts.

    Accepts up to ~512 texts per call.  The underlying SentenceTransformer
    `.encode()` is CPU-bound, so FastAPI runs it in its default threadpool
    (because this is a plain ``def``, not ``async def``).
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty")

    vectors = model.encode(request.texts).tolist()
    return EmbedResponse(
        embeddings=vectors,
        model=MODEL_NAME,
        dimension=model.get_sentence_embedding_dimension(),
    )


@app.get("/health")
def health():
    """Liveness / readiness probe."""
    return {"status": "healthy", "model": MODEL_NAME}


# ── Entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
