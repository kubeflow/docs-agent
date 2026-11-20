"""Tests for the temporal (latest/current/supported) rule in intent_router.pick_search_plan."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"))

from intent_router import pick_search_plan


@pytest.mark.parametrize(
    "query",
    [
        # tests/eval/flo_eval_dataset.json: fid-latest-release
        "What is the latest Kubeflow release?",
        "newest Katib release",
        # tests/eval/flo_eval_dataset.json: fid-current-kfp
        "What is the current version of Kubeflow Pipelines?",
        # keyword form the system prompt asks the model to send
        "Kubeflow Pipelines current version",
        "supported Kubernetes versions for Kubeflow",
        "Which release is current?",
    ],
)
def test_release_questions_search_release_notes(query):
    plan = pick_search_plan(query)
    assert plan.intent == "temporal"
    assert plan.filter_expr == 'doc_type == "release"'
    assert plan.rerank_by_release_date


@pytest.mark.parametrize(
    "query",
    [
        # tests/eval/flo_eval_dataset.json: fid-parallelfor-v2
        "Is dsl.ParallelFor supported in KFP v2?",
        # tests/eval/flo_eval_dataset.json: dbg-external-mysql
        "Is running Kubeflow Pipelines with an external MySQL supported, and how is it configured?",
        "How do I get the current run ID inside a KFP component?",
        "Which hyperparameter search algorithms are supported by Katib?",
        # keyword forms of the two questions above
        "KFP current run ID component",
        "Katib supported search algorithms",
    ],
)
def test_current_or_supported_alone_does_not_force_release_notes(query):
    plan = pick_search_plan(query)
    assert plan.intent != "temporal"
    assert plan.filter_expr == ""
    assert not plan.rerank_by_release_date
