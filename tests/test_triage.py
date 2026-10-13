"""Unit tests for the routing logic of tests/feedback/triage.py (no network)."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "feedback" / "triage.py"
_spec = importlib.util.spec_from_file_location("triage", _PATH)
triage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(triage)


def test_component_from_question_outweighs_sources():
    assert triage.guess_component("How do I set parallelTrialCount in Katib?", ["components/pipelines/x"]) == "katib"
    assert triage.guess_component("hello", []) == "general"


def test_owner_pipeline_layers_go_to_docs_agent():
    for layer in ("generation", "retrieval", "tool_routing", "agent_skipped_retrieval"):
        assert triage.owner({"layer": layer, "suggested_repo": "kubeflow/katib"}) == "kubeflow/docs-agent"


def test_owner_content_gaps_go_to_component_repo():
    assert triage.owner({"layer": "content_gap_hallucination", "suggested_repo": "kubeflow/katib"}) == "kubeflow/katib"
    assert triage.owner({"layer": "ok", "suggested_repo": "kubeflow/katib"}) == "-"


def test_split_chunks_and_sources():
    text = "### Result 1 [c1]\n**Source:** a.md\nbody\n### Result 2 [c2]\n**File:** b.yaml\n"
    chunks = triage.split_chunks(text)
    assert len(chunks) == 2 and triage.sources_of(chunks) == ["a.md", "b.yaml"]


def test_report_groups_by_owner():
    md = triage.report([
        {"layer": "retrieval", "component": "katib", "suggested_repo": "kubeflow/katib", "question": "q1",
         "reason": "r", "trace_id": "t1", "comment": ""},
        {"layer": "content_gap_handled", "component": "kserve", "suggested_repo": "kserve/kserve", "question": "q2",
         "reason": "r", "trace_id": "t2", "comment": "missing"},
    ])
    assert "## kubeflow/docs-agent" in md and "## kserve/kserve" in md and 'User said: "missing"' in md
