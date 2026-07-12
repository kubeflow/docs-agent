"""Anonymous session-token issuer for the Kubeflow docs chatbot.

POST /api/session           -> short-lived RS256 JWT (no user accounts)
GET  /.well-known/jwks.json -> public JWKS (fetched by Istio RequestAuthentication)
GET  /healthz               -> liveness

The chat widget fetches a token on page load and attaches it as
`Authorization: Bearer <jwt>` to every A2A request. Istio validates
signature + expiry at the ingress gateway; the widget re-fetches a token
when it receives a 401.
"""

import os

from fastapi import FastAPI

from issuer_core import (
    DEFAULT_ISSUER,
    DEFAULT_TTL_SECONDS,
    build_jwks,
    compute_kid,
    load_or_generate_private_key,
    mint_token,
)

PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH", "/keys/private.pem")
ISSUER = os.getenv("SESSION_ISSUER", DEFAULT_ISSUER)
TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
PORT = int(os.getenv("PORT", "8000"))

_private_key = load_or_generate_private_key(PRIVATE_KEY_PATH)
_kid = compute_kid(_private_key)
_jwks = build_jwks(_private_key, _kid)

app = FastAPI(title="Kubeflow Docs Session Issuer", docs_url=None, redoc_url=None)


@app.post("/api/session")
def create_session():
    return mint_token(_private_key, _kid, issuer=ISSUER, ttl_seconds=TTL_SECONDS)


@app.get("/.well-known/jwks.json")
def jwks():
    return _jwks


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
