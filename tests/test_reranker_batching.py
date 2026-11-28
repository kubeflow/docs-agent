"""Rerank requests must fit TEI's --max-client-batch-size (issue #308)."""

import logging
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs-agent-mcp" / "mcp-server"))

import reranker_client  # noqa: E402

SERVER_LIMIT = 32


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} Client Error")

    def json(self):
        return self.payload


class FakeTEI:
    """Rejects oversized batches like TEI; scores "doc <n>" as n/1000, best first."""

    def __init__(self, fail_on_call=None):
        self.batches = []
        self.timeouts = []
        self.fail_on_call = fail_on_call

    def post(self, url, json, headers, timeout):
        texts = json["texts"]
        self.batches.append(texts)
        self.timeouts.append(timeout)
        if len(self.batches) == self.fail_on_call:
            raise requests.ConnectionError("connection reset")
        if len(texts) > SERVER_LIMIT:
            return FakeResponse({"error": "batch size"}, status=413)
        ranked = [{"index": i, "score": int(t.split()[1]) / 1000} for i, t in enumerate(texts)]
        return FakeResponse(sorted(ranked, key=lambda r: -r["score"]))


def make_hits(count):
    # BM25-like distances in the opposite order to the cross-encoder scores.
    return [{"distance": 100.0 - i, "entity": {"content_text": f"doc {i}"}} for i in range(count)]


def install(monkeypatch, tei):
    monkeypatch.setenv("RERANKER_URL", "http://reranker.test/rerank")
    monkeypatch.setattr(reranker_client.requests, "post", tei.post)
    return tei


@pytest.mark.parametrize(
    "count, sizes",
    [(1, [1]), (32, [32]), (33, [32, 1]), (50, [32, 18]), (64, [32, 32]), (65, [32, 32, 1])],
)
def test_requests_stay_within_the_batch_limit(monkeypatch, count, sizes):
    tei = install(monkeypatch, FakeTEI())

    reranker_client.rerank_hits("query", make_hits(count))

    assert [len(batch) for batch in tei.batches] == sizes


def test_scores_map_back_to_the_right_hit_across_batches(monkeypatch):
    install(monkeypatch, FakeTEI())
    hits = make_hits(50)

    result = reranker_client.rerank_hits("query", hits)

    # Every hit is scored, and the best ones come from the second batch.
    assert [h["entity"]["content_text"] for h in result] == [f"doc {i}" for i in reversed(range(50))]
    assert all(h["distance"] == int(h["entity"]["content_text"].split()[1]) / 1000 for h in result)
    assert all(h["reranked"] is True for h in result)
    assert hits == make_hits(50), "input hits must not be modified"


def test_batch_size_is_configurable(monkeypatch):
    tei = install(monkeypatch, FakeTEI())
    monkeypatch.setattr(reranker_client, "MAX_BATCH", 10)

    result = reranker_client.rerank_hits("query", make_hits(25))

    assert [len(batch) for batch in tei.batches] == [10, 10, 5]
    assert len(result) == 25


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_any_failed_batch_keeps_the_original_order_and_warns(monkeypatch, caplog, fail_on_call):
    install(monkeypatch, FakeTEI(fail_on_call=fail_on_call))
    hits = make_hits(50)

    with caplog.at_level(logging.WARNING):
        result = reranker_client.rerank_hits("query", hits)

    # Half-reranked hits would mix cross-encoder scores with BM25 distances.
    assert result == make_hits(50)
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "connection reset" in warning.getMessage()
    assert "50" in warning.getMessage()


def test_unexpected_payload_keeps_the_original_order_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("RERANKER_URL", "http://reranker.test/rerank")
    monkeypatch.setattr(reranker_client.requests, "post", lambda *a, **k: FakeResponse({"error": "nope"}))

    with caplog.at_level(logging.WARNING):
        result = reranker_client.rerank_hits("query", make_hits(3))

    assert result == make_hits(3)
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_out_of_range_index_does_not_score_a_hit_in_another_batch(monkeypatch):
    def post(url, json, headers, timeout):
        size = len(json["texts"])
        scores = [{"index": i, "score": 0.5} for i in range(size)]
        # Index 32 is outside a 32-text batch, but is a valid position in the full list.
        return FakeResponse(scores + [{"index": size, "score": 0.99}])

    monkeypatch.setenv("RERANKER_URL", "http://reranker.test/rerank")
    monkeypatch.setattr(reranker_client.requests, "post", post)

    result = reranker_client.rerank_hits("query", make_hits(50))

    assert len(result) == 50
    assert {h["distance"] for h in result} == {0.5}


def test_timeout_is_a_budget_for_all_batches(monkeypatch, caplog):
    tei = install(monkeypatch, FakeTEI())
    clock = {"now": 0.0}

    def slow_post(*args, **kwargs):
        clock["now"] += 20
        return FakeTEI.post(tei, *args, **kwargs)

    monkeypatch.setattr(reranker_client.requests, "post", slow_post)
    monkeypatch.setattr(reranker_client.time, "monotonic", lambda: clock["now"])

    with caplog.at_level(logging.WARNING):
        result = reranker_client.rerank_hits("query", make_hits(96), timeout=30)

    # 20s + 20s uses up the budget, so the third batch is never sent.
    assert tei.timeouts == [30, 10]
    assert result == make_hits(96)
    assert any("budget" in r.getMessage() for r in caplog.records)


def test_candidate_33_wins_from_the_second_batch(monkeypatch):
    # TEI numbers each request from 0, so candidate 33 comes back as index 0 of batch 2.
    def post(url, json, headers, timeout):
        return FakeResponse(
            [{"index": i, "score": 0.99 if t == "doc 32" else 0.1} for i, t in enumerate(json["texts"])]
        )

    monkeypatch.setenv("RERANKER_URL", "http://reranker.test/rerank")
    monkeypatch.setattr(reranker_client.requests, "post", post)

    result = reranker_client.rerank_hits("query", make_hits(50))

    assert result[0]["entity"]["content_text"] == "doc 32"
    assert result[0]["distance"] == 0.99
