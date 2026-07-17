"""Tests for the session issuer core (docs-agent-mcp/session-issuer/issuer_core.py).

Pure-unit: no FastAPI/uvicorn imports, no network. Verifies the JWT
mint -> JWKS -> verify roundtrip that Istio RequestAuthentication performs
at the ingress gateway.
"""

import base64
import importlib.util
import time
from pathlib import Path

import jwt
import pytest

ISSUER_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "session-issuer"
ISSUER_CORE_PATH = ISSUER_DIR / "issuer_core.py"

spec = importlib.util.spec_from_file_location("issuer_core", ISSUER_CORE_PATH)
issuer_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(issuer_core)


@pytest.fixture(scope="module")
def private_key():
    return issuer_core.load_or_generate_private_key(path=None)


@pytest.fixture(scope="module")
def kid(private_key):
    return issuer_core.compute_kid(private_key)


@pytest.mark.unit
class TestMintAndVerify:
    def test_token_verifies_against_public_key(self, private_key, kid):
        """The roundtrip Istio performs: verify RS256 signature + expiry."""
        resp = issuer_core.mint_token(private_key, kid)
        claims = jwt.decode(
            resp["access_token"],
            private_key.public_key(),
            algorithms=["RS256"],
            options={"require": ["iss", "sub", "iat", "exp"]},
        )
        assert claims["iss"] == issuer_core.DEFAULT_ISSUER
        assert claims["sub"].startswith("session-")

    def test_token_response_shape(self, private_key, kid):
        resp = issuer_core.mint_token(private_key, kid, ttl_seconds=60)
        assert resp["token_type"] == "Bearer"
        assert resp["expires_in"] == 60

    def test_expiry_honors_ttl(self, private_key, kid):
        resp = issuer_core.mint_token(private_key, kid, ttl_seconds=120)
        claims = jwt.decode(resp["access_token"], private_key.public_key(), algorithms=["RS256"])
        assert claims["exp"] - claims["iat"] == 120
        assert abs(claims["iat"] - time.time()) < 10

    def test_expired_token_rejected(self, private_key, kid):
        resp = issuer_core.mint_token(private_key, kid, ttl_seconds=-10)
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt.decode(resp["access_token"], private_key.public_key(), algorithms=["RS256"])

    def test_sessions_are_unique(self, private_key, kid):
        subs = set()
        for _ in range(5):
            resp = issuer_core.mint_token(private_key, kid)
            claims = jwt.decode(resp["access_token"], private_key.public_key(), algorithms=["RS256"])
            subs.add(claims["sub"])
        assert len(subs) == 5

    def test_wrong_key_rejected(self, private_key, kid):
        """A token signed by another key must not verify (forged tokens)."""
        other_key = issuer_core.load_or_generate_private_key(path=None)
        resp = issuer_core.mint_token(other_key, kid)
        with pytest.raises(jwt.InvalidSignatureError):
            jwt.decode(resp["access_token"], private_key.public_key(), algorithms=["RS256"])

    def test_kid_in_header(self, private_key, kid):
        resp = issuer_core.mint_token(private_key, kid)
        header = jwt.get_unverified_header(resp["access_token"])
        assert header["kid"] == kid
        assert header["alg"] == "RS256"


@pytest.mark.unit
class TestJwks:
    def test_jwks_shape(self, private_key, kid):
        jwks = issuer_core.build_jwks(private_key, kid)
        assert len(jwks["keys"]) == 1
        key = jwks["keys"][0]
        assert key["kty"] == "RSA"
        assert key["alg"] == "RS256"
        assert key["use"] == "sig"
        assert key["kid"] == kid

    def test_jwks_b64url_no_padding(self, private_key, kid):
        """RFC 7518: base64url without padding; Envoy's JWKS parser is strict."""
        key = issuer_core.build_jwks(private_key, kid)["keys"][0]
        for field in ("n", "e"):
            assert "=" not in key[field]
            assert "+" not in key[field]
            assert "/" not in key[field]

    def test_jwks_verifies_token(self, private_key, kid):
        """Full path: reconstruct the public key from JWKS and verify a token."""
        jwks = issuer_core.build_jwks(private_key, kid)
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(jwks["keys"][0])
        resp = issuer_core.mint_token(private_key, kid)
        claims = jwt.decode(resp["access_token"], public_key, algorithms=["RS256"])
        assert claims["iss"] == issuer_core.DEFAULT_ISSUER

    def test_e_is_65537(self, private_key, kid):
        key = issuer_core.build_jwks(private_key, kid)["keys"][0]
        e = int.from_bytes(base64.urlsafe_b64decode(key["e"] + "=="), "big")
        assert e == 65537


@pytest.mark.unit
class TestKeyLoading:
    def test_stable_kid(self, private_key):
        assert issuer_core.compute_kid(private_key) == issuer_core.compute_kid(private_key)

    def test_load_from_pem(self, private_key, tmp_path):
        """Key round-trips through PEM (the Secret mount path in-cluster)."""
        from cryptography.hazmat.primitives import serialization

        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        key_file = tmp_path / "private.pem"
        key_file.write_bytes(pem)
        loaded = issuer_core.load_or_generate_private_key(str(key_file))
        assert issuer_core.compute_kid(loaded) == issuer_core.compute_kid(private_key)

    def test_missing_path_generates_ephemeral(self, tmp_path):
        key = issuer_core.load_or_generate_private_key(str(tmp_path / "nope.pem"))
        assert key.key_size == 2048
