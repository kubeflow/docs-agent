"""Unit tests for the chat-widget feedback service (gateway-guardrails/files/feedback.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "docs-agent-mcp" / "charts" / "gateway-guardrails" / "files" / "feedback.py"
_spec = importlib.util.spec_from_file_location("feedback", _PATH)
feedback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feedback)


def _body(**kw):
    return json.dumps({"contextId": "ctx-1", "value": "up", **kw}).encode()


class TestParse:
    def test_minimal(self):
        assert feedback.parse(_body()) == {"contextId": "ctx-1", "value": "up", "turn": None, "comment": ""}

    def test_turn_and_comment(self):
        fb = feedback.parse(_body(value="down", turn=2, comment="  wrong version  "))
        assert fb["turn"] == 2 and fb["comment"] == "wrong version"

    def test_comment_truncated(self):
        assert len(feedback.parse(_body(comment="x" * 1500))["comment"]) == feedback.MAX_COMMENT

    @pytest.mark.parametrize("kw", [
        {"value": "meh"},
        {"contextId": ""},
        {"contextId": "a b"},
        {"contextId": "x" * 200},
        {"turn": -1},
        {"turn": "1"},
        {"turn": True},
        {"comment": 5},
    ])
    def test_rejects(self, kw):
        with pytest.raises(feedback.BadRequest):
            feedback.parse(_body(**kw))

    def test_rejects_non_json_and_oversize(self):
        with pytest.raises(feedback.BadRequest):
            feedback.parse(b"not json")
        with pytest.raises(feedback.BadRequest):
            feedback.parse(b"{" + b" " * feedback.MAX_BODY + b"}")


class TestPickTrace:
    OBS = [
        {"traceId": "t2", "startTime": "2026-10-14T10:05:00Z"},
        {"traceId": "t1", "startTime": "2026-10-14T10:00:01Z"},
        {"traceId": "t1", "startTime": "2026-10-14T10:00:00Z"},
        {"traceId": "t3", "startTime": "2026-10-14T10:09:00Z"},
    ]

    def test_turn_indexes_in_chat_order(self):
        assert [feedback.pick_trace(self.OBS, i) for i in range(3)] == ["t1", "t2", "t3"]

    def test_latest_when_turn_missing_or_out_of_range(self):
        assert feedback.pick_trace(self.OBS, None) == "t3"
        assert feedback.pick_trace(self.OBS, 9) == "t3"

    def test_no_traces(self):
        assert feedback.pick_trace([], 0) is None


class TestScoreBody:
    def test_trace_score_is_idempotent_per_trace(self):
        fb = feedback.parse(_body(value="down", turn=1, comment="missing step"))
        body = feedback.score_body(fb, "abc")
        assert body["id"] == "abc-user_feedback" and body["traceId"] == "abc"
        assert body["value"] == 0 and body["dataType"] == "BOOLEAN" and body["comment"] == "missing step"

    def test_session_fallback_when_trace_not_ingested(self):
        body = feedback.score_body(feedback.parse(_body(turn=0)), None)
        assert body["sessionId"] == "ctx-1" and "traceId" not in body and body["value"] == 1
        assert body["id"] == "ctx-1-turn-0-user_feedback"
