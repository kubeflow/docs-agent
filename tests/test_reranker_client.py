"""Regression tests for reranker_client error handling and partial-response ordering."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"))

import reranker_client  # noqa: E402

RERANKER_URL = "http://reranker.test/rerank"


def _make_response(
    *,
    status_code: int = 200,
    body: bytes = b"[]",
    content_type: str = "application/json",
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    response.headers["Content-Type"] = content_type
    response.url = RERANKER_URL
    return response


def _json_response(payload: object, *, status_code: int = 200) -> requests.Response:
    return _make_response(status_code=status_code, body=json.dumps(payload).encode("utf-8"))


def _sample_hits() -> list[dict]:
    return [
        {"id": 1, "distance": 14.2, "entity": {"content_text": "doc0"}},
        {"id": 2, "distance": 11.8, "entity": {"content_text": "doc1"}},
        {"id": 3, "distance": 9.5, "entity": {"content_text": "doc2"}},
    ]


def test_malformed_json_returns_original_hits_unchanged(monkeypatch):
    hits = _sample_hits()
    original = copy.deepcopy(hits)
    response = _make_response(status_code=200, body=b"<html>502 Bad Gateway</html>", content_type="text/html")

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert result == original
    assert all("reranked" not in hit for hit in result)


def test_empty_response_body_returns_original_hits_unchanged(monkeypatch):
    hits = _sample_hits()
    original = copy.deepcopy(hits)
    response = _make_response(status_code=200, body=b"")

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert result == original
    assert all("reranked" not in hit for hit in result)


@pytest.mark.parametrize(
    "side_effect",
    [
        requests.Timeout("timed out"),
        requests.ConnectionError("connection refused"),
        _make_response(status_code=500, body=b"internal server error", content_type="text/plain"),
    ],
)
def test_request_and_http_errors_preserve_original_hits(monkeypatch, side_effect):
    hits = _sample_hits()
    original = copy.deepcopy(hits)

    def fake_post(*args, **kwargs):
        if isinstance(side_effect, Exception):
            raise side_effect
        return side_effect

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", fake_post)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert result == original
    assert all("reranked" not in hit for hit in result)


@pytest.mark.parametrize("body", [b'{"error": "unexpected"}', b'"ok"', b"null", b"42"])
def test_unexpected_top_level_json_preserves_original_hits(monkeypatch, body):
    hits = _sample_hits()
    original = copy.deepcopy(hits)
    response = _make_response(status_code=200, body=body)

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert result == original


def test_non_dict_payload_entries_do_not_crash(monkeypatch):
    hits = _sample_hits()[:2]
    response = _json_response(["bad-entry", None, 123, {"index": 1, "score": 0.85}])

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert len(result) == 2
    assert [hit["entity"]["content_text"] for hit in result] == ["doc1", "doc0"]
    assert result[0]["distance"] == pytest.approx(0.85)
    assert result[0]["reranked"] is True
    assert result[1]["distance"] == 14.2
    assert "reranked" not in result[1]


def test_complete_reranker_response_sorts_by_score_descending(monkeypatch):
    hits = _sample_hits()
    response = _json_response(
        [
            {"index": 1, "score": 0.25},
            {"index": 2, "score": 0.95},
            {"index": 0, "score": 0.6},
        ]
    )

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert len(result) == 3
    assert [hit["entity"]["content_text"] for hit in result] == ["doc2", "doc0", "doc1"]
    assert [hit["distance"] for hit in result] == [0.95, 0.6, 0.25]
    assert all(hit["reranked"] is True for hit in result)


def test_equal_scores_use_deterministic_original_index_tiebreaker(monkeypatch):
    hits = _sample_hits()
    response = _json_response(
        [
            {"index": 2, "score": 0.8},
            {"index": 0, "score": 0.8},
            {"index": 1, "score": 0.8},
        ]
    )

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert [hit["entity"]["content_text"] for hit in result] == ["doc0", "doc1", "doc2"]
    assert all(hit["distance"] == 0.8 and hit["reranked"] is True for hit in result)


@pytest.mark.parametrize(
    "hits, payload, expected_order, expected_distances",
    [
        # BM25 scores (14.2, 11.8, 9.5) are numerically larger than cross-encoder scores (0.95).
        (
            [
                {"id": 1, "distance": 14.2, "entity": {"content_text": "doc0"}},
                {"id": 2, "distance": 11.8, "entity": {"content_text": "doc1"}},
                {"id": 3, "distance": 9.5, "entity": {"content_text": "doc2"}},
            ],
            [{"index": 2, "score": 0.95}],
            ["doc2", "doc0", "doc1"],
            [0.95, 14.2, 11.8],
        ),
        # Cosine similarity (0.89) exceeds cross-encoder scores (0.65, 0.30).
        (
            [
                {"id": 1, "distance": 0.89, "entity": {"content_text": "doc0"}},
                {"id": 2, "distance": 0.84, "entity": {"content_text": "doc1"}},
                {"id": 3, "distance": 0.78, "entity": {"content_text": "doc2"}},
            ],
            [{"index": 1, "score": 0.30}, {"index": 2, "score": 0.65}],
            ["doc2", "doc1", "doc0"],
            [0.65, 0.30, 0.89],
        ),
    ],
)
def test_partial_response_ranks_reranked_first_and_preserves_unreranked_order(
    monkeypatch,
    hits,
    payload,
    expected_order,
    expected_distances,
):
    response = _json_response(payload)

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert len(result) == len(hits)
    assert [hit["entity"]["content_text"] for hit in result] == expected_order
    assert [hit["distance"] for hit in result] == expected_distances
    reranked_count = len(payload)
    assert all(hit.get("reranked") is True for hit in result[:reranked_count])
    assert all("reranked" not in hit for hit in result[reranked_count:])


def test_empty_reranker_list_preserves_original_non_distance_order(monkeypatch):
    # Intentionally not sorted by distance (e.g., after intent_router version matching).
    hits = [
        {"id": 1, "distance": 1.0, "entity": {"content_text": "docA"}},
        {"id": 2, "distance": 9.0, "entity": {"content_text": "docB"}},
        {"id": 3, "distance": 5.0, "entity": {"content_text": "docC"}},
    ]
    original = copy.deepcopy(hits)
    response = _json_response([])

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow 1.9 release date", hits)

    assert result == original
    assert [hit["entity"]["content_text"] for hit in result] == ["docA", "docB", "docC"]
    assert all("reranked" not in hit for hit in result)


@pytest.mark.parametrize(
    "invalid_item",
    [
        {"index": 0},  # missing score
        {"index": 0, "score": None},  # null score
        {"index": 0, "score": "0.9"},  # string score
        {"index": 0, "score": True},  # boolean score
        {"index": 0, "score": False},  # boolean score
        {"index": 0, "score": float("nan")},  # NaN score
        {"index": 0, "score": float("inf")},  # +inf score
        {"index": 0, "score": float("-inf")},  # -inf score
        {"index": 0, "score": 10**10000},  # integer too large for float
        {"index": 0, "score": -(10**10000)},  # negative integer too large for float
    ],
)
def test_invalid_scores_are_rejected(monkeypatch, invalid_item):
    hits = _sample_hits()[:2]
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [invalid_item, {"index": 1, "score": 0.7}]

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: mock_response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert len(result) == 2
    assert [hit["entity"]["content_text"] for hit in result] == ["doc1", "doc0"]
    assert result[0]["distance"] == pytest.approx(0.7)
    assert result[0]["reranked"] is True
    assert result[1]["distance"] == 14.2
    assert "reranked" not in result[1]


def test_boolean_indices_are_rejected(monkeypatch):
    hits = _sample_hits()
    response = _json_response(
        [
            {"index": True, "score": 0.99},
            {"index": False, "score": 0.98},
            {"index": 2, "score": 0.5},
        ]
    )

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert len(result) == 3
    assert [hit["entity"]["content_text"] for hit in result] == ["doc2", "doc0", "doc1"]
    assert result[0]["reranked"] is True
    assert result[0]["distance"] == 0.5
    assert "reranked" not in result[1]
    assert "reranked" not in result[2]


def test_duplicate_and_out_of_range_indices_do_not_duplicate_hits(monkeypatch):
    hits = _sample_hits()[:2]
    response = _json_response(
        [
            {"index": 0, "score": 0.9},
            {"index": 0, "score": 0.1},
            {"index": -1, "score": 0.99},
            {"index": 2, "score": 0.99},
            {"index": 99, "score": 0.99},
            {"index": 1, "score": 0.5},
        ]
    )

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert len(result) == 2
    assert [hit["entity"]["content_text"] for hit in result] == ["doc0", "doc1"]
    assert [hit["distance"] for hit in result] == [0.9, 0.5]
    assert all(hit["reranked"] is True for hit in result)


def test_original_input_hit_dicts_are_not_mutated(monkeypatch):
    hits = _sample_hits()
    snapshot = copy.deepcopy(hits)
    response = _json_response([{"index": 2, "score": 1}, {"index": 0, "score": 0.4}])

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert hits == snapshot
    assert result[0]["distance"] == 1.0
    assert isinstance(result[0]["distance"], float)
    assert all("reranked" not in hit for hit in hits)


def test_all_invalid_entries_return_original_hits_unchanged(monkeypatch):
    hits = [
        {"id": 1, "distance": 2.0, "entity": {"content_text": "docA"}},
        {"id": 2, "distance": 8.0, "entity": {"content_text": "docB"}},
    ]
    original = copy.deepcopy(hits)
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        "not-a-dict",
        {"index": True, "score": 0.9},
        {"index": 99, "score": 0.8},
        {"index": 0},
        {"index": 1, "score": None},
        {"index": 0, "score": "0.5"},
        {"index": 1, "score": float("nan")},
    ]

    monkeypatch.setenv("RERANKER_URL", RERANKER_URL)
    monkeypatch.setattr(reranker_client.requests, "post", lambda *args, **kwargs: mock_response)

    result = reranker_client.rerank_hits("kubeflow pipelines", hits)

    assert result == original
    assert [hit["entity"]["content_text"] for hit in result] == ["docA", "docB"]
    assert all("reranked" not in hit for hit in result)
