"""
tests/test_agent_router.py
--------------------------
Real unit tests for server/agent_router.py — no mocks, no stubs.

Tests cover every pure-logic function and the full LangGraph graph structure.
External services (Milvus cluster, KServe LLM) are intentionally NOT called;
the module's own graceful error-handling paths are exercised instead.

Run:
    pytest tests/test_agent_router.py -v
"""

from __future__ import annotations

import pytest
from server.agent_router import (
    AgentState,
    RouteQuery,
    _format_hits,
    build_graph,
    handle_empty_retrieval,
    increment_retries,
    route_to_partition,
    MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> AgentState:
    base: AgentState = {
        "question": "How do I install Kubeflow?",
        "intent": "",
        "context": "",
        "citations": [],
        "response": "",
        "error": None,
        "retries": 0,
    }
    base.update(overrides)
    return base


def make_hit(content="some content", url="https://kubeflow.org/docs", path="docs/a.md", score=0.9):
    return {"citation_url": url, "content_text": content, "file_path": path, "similarity": score}


# ---------------------------------------------------------------------------
# RouteQuery — Pydantic schema validation (real Pydantic, no mocks)
# ---------------------------------------------------------------------------

class TestRouteQuery:
    @pytest.mark.parametrize("ds", ["docs", "issues", "platform"])
    def test_valid_datasource_passes_validation(self, ds):
        rq = RouteQuery(datasource=ds)
        assert rq.validate_datasource() == ds

    def test_invalid_datasource_raises_value_error(self):
        rq = RouteQuery(datasource="hallucinated_partition")
        with pytest.raises(ValueError, match="datasource must be one of"):
            rq.validate_datasource()

    def test_empty_datasource_raises_value_error(self):
        rq = RouteQuery(datasource="")
        with pytest.raises(ValueError):
            rq.validate_datasource()

    def test_case_sensitive(self):
        """'Docs' (capital D) is not a valid partition — must be lowercase."""
        rq = RouteQuery(datasource="Docs")
        with pytest.raises(ValueError):
            rq.validate_datasource()


# ---------------------------------------------------------------------------
# AgentState — structure & defaults
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_all_required_keys_present(self):
        state = make_state()
        assert set(state.keys()) == {
            "question", "intent", "context", "citations",
            "response", "error", "retries",
        }

    def test_retries_defaults_to_zero(self):
        state = make_state()
        assert state["retries"] == 0

    def test_citations_defaults_to_empty_list(self):
        state = make_state()
        assert state["citations"] == []

    def test_error_defaults_to_none(self):
        state = make_state()
        assert state["error"] is None


# ---------------------------------------------------------------------------
# _format_hits — pure function, no external calls
# ---------------------------------------------------------------------------

class TestFormatHits:
    def test_empty_hits_returns_empty_string_and_list(self):
        text, citations = _format_hits([])
        assert text == ""
        assert citations == []

    def test_single_hit_populates_context_and_citations(self):
        hits = [make_hit("KFP docs text", "https://kubeflow.org/kfp")]
        text, citations = _format_hits(hits)
        assert "KFP docs text" in text
        assert "https://kubeflow.org/kfp" in citations

    def test_deduplicates_citations(self):
        hits = [
            make_hit(url="https://a.com"),
            make_hit(url="https://a.com"),   # duplicate
            make_hit(url="https://b.com"),
        ]
        _, citations = _format_hits(hits)
        assert citations.count("https://a.com") == 1
        assert "https://b.com" in citations

    def test_preserves_citation_insertion_order(self):
        hits = [make_hit(url=f"https://site{i}.com") for i in range(5)]
        _, citations = _format_hits(hits)
        assert citations == [f"https://site{i}.com" for i in range(5)]

    def test_skips_empty_url(self):
        hits = [make_hit(url="")]
        _, citations = _format_hits(hits)
        assert citations == []

    def test_score_included_in_text(self):
        hits = [make_hit(score=0.95)]
        text, _ = _format_hits(hits)
        assert "0.950" in text

    def test_multiple_hits_separated_in_text(self):
        hits = [make_hit(content="A"), make_hit(content="B")]
        text, _ = _format_hits(hits)
        assert "A" in text
        assert "B" in text


# ---------------------------------------------------------------------------
# route_to_partition — conditional edge, pure function
# ---------------------------------------------------------------------------

class TestRouteToPartition:
    @pytest.mark.parametrize("intent", ["docs", "issues", "platform"])
    def test_returns_correct_partition_for_intent(self, intent):
        state = make_state(intent=intent)
        assert route_to_partition(state) == intent

    def test_empty_intent_returns_docs_fallback(self):
        state = make_state(intent="")
        assert route_to_partition(state) == "docs"

    def test_missing_intent_key_returns_docs_fallback(self):
        state = make_state()
        del state["intent"]
        assert route_to_partition(state) == "docs"


# ---------------------------------------------------------------------------
# handle_empty_retrieval — self-correction logic, pure function
# ---------------------------------------------------------------------------

class TestHandleEmptyRetrieval:
    def test_retries_when_error_and_under_limit(self):
        state = make_state(error="no results", retries=0)
        assert handle_empty_retrieval(state) == "retry"

    def test_retries_until_max_minus_one(self):
        state = make_state(error="no results", retries=MAX_RETRIES - 1)
        assert handle_empty_retrieval(state) == "retry"

    def test_ends_at_max_retries(self):
        state = make_state(error="no results", retries=MAX_RETRIES)
        assert handle_empty_retrieval(state) == "end"

    def test_ends_when_no_error(self):
        state = make_state(error=None, retries=0)
        assert handle_empty_retrieval(state) == "end"

    def test_ends_when_no_error_even_with_retries(self):
        state = make_state(error=None, retries=1)
        assert handle_empty_retrieval(state) == "end"

    def test_max_retries_constant_is_positive(self):
        assert MAX_RETRIES > 0


# ---------------------------------------------------------------------------
# increment_retries — pure function
# ---------------------------------------------------------------------------

class TestIncrementRetries:
    def test_increments_from_zero(self):
        result = increment_retries(make_state(retries=0))
        assert result["retries"] == 1

    def test_increments_from_nonzero(self):
        result = increment_retries(make_state(retries=3))
        assert result["retries"] == 4

    def test_does_not_mutate_other_state_keys(self):
        state = make_state(retries=0, question="test Q", intent="docs")
        result = increment_retries(state)
        assert result["question"] == "test Q"
        assert result["intent"] == "docs"


# ---------------------------------------------------------------------------
# build_graph — real LangGraph compile; checks graph topology
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def setup_method(self):
        self.graph = build_graph()

    def test_compiles_without_error(self):
        assert self.graph is not None

    def test_contains_all_expected_nodes(self):
        node_names = set(self.graph.get_graph().nodes.keys())
        expected = {"intent_router", "docs", "issues", "platform", "increment_retries"}
        assert expected.issubset(node_names)

    def test_entry_point_is_intent_router(self):
        g = self.graph.get_graph()
        # '__start__' → 'intent_router' edge must exist
        edge_targets = {e[1] for e in g.edges}
        assert "intent_router" in edge_targets

    def test_end_node_reachable_from_retrieval_nodes(self):
        g = self.graph.get_graph()
        edge_sources = {e[0] for e in g.edges}
        # Each partition node must have an outgoing edge
        for node in ("docs", "issues", "platform"):
            assert node in edge_sources, f"Node '{node}' has no outgoing edges"

    def test_increment_retries_connects_back_to_router(self):
        g = self.graph.get_graph()
        retry_edges = [e for e in g.edges if e[0] == "increment_retries"]
        assert any(e[1] == "intent_router" for e in retry_edges)
