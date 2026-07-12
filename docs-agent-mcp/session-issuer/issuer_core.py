"""Core logic for the anonymous session-token issuer.

Framework-free (no FastAPI imports) so it can be unit-tested without web
dependencies. Mints short-lived RS256 JWTs whose signature and expiry are
validated by Istio's RequestAuthentication at the ingress gateway — expired
or missing sessions are rejected at the edge, before reaching the LLM.

No user accounts: `sub` is a random session id. "Logout"/invalidation is
simply token expiry (`exp`), so no revocation store is needed.
"""

import base64
import hashlib
import os
import time
import uuid

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

DEFAULT_ISSUER = "kubeflow-docs-session-issuer"
DEFAULT_TTL_SECONDS = 1800  # 30 minutes


def _b64url_uint(n: int) -> str:
    """Base64url-encode an unsigned integer without padding (RFC 7518)."""
    data = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def load_or_generate_private_key(path=None):
    """Load the RSA private key PEM from `path`, or generate an ephemeral one.

    In-cluster the key comes from a Secret mount so tokens stay valid across
    pod restarts; the ephemeral fallback is for local development only.
    """
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def compute_kid(private_key) -> str:
    """Stable key id: SHA-256 of the public key DER, first 16 hex chars."""
    der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()[:16]


def build_jwks(private_key, kid: str) -> dict:
    """Public JWKS document consumed by Istio via the jwksUri."""
    numbers = private_key.public_key().public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "alg": "RS256",
                "kid": kid,
                "n": _b64url_uint(numbers.n),
                "e": _b64url_uint(numbers.e),
            }
        ]
    }


def mint_token(private_key, kid: str, issuer: str = DEFAULT_ISSUER, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> dict:
    """Mint an anonymous session JWT. Returns an OAuth-style token response."""
    now = int(time.time())
    claims = {
        "iss": issuer,
        "sub": f"session-{uuid.uuid4()}",
        "iat": now,
        "exp": now + ttl_seconds,
    }
    token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})
    return {"access_token": token, "token_type": "Bearer", "expires_in": ttl_seconds}
