"""Deterministic tests for the two-gate showcase evaluator."""

import importlib.util
import io
import sys
from pathlib import Path


EVAL_PATH = Path(__file__).parent / "eval" / "run_showcase_eval.py"
SPEC = importlib.util.spec_from_file_location("run_showcase_eval", EVAL_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVAL
SPEC.loader.exec_module(EVAL)


def golden_row():
    return {
        "id": "golden",
        "tool": "search_kubeflow_code",
        "query": "Show me an example",
        "retrieval_query": "katib random yaml",
        "expected_source_urls": ["https://github.com/kubeflow/katib/blob/master/random.yaml"],
        "must_appear_in_answer": ["kubeflow.org/v1beta1", "kind: Experiment"],
        "forbidden_in_answer": ["katib.mlr-org"],
    }


def test_retrieval_gate_and_evidence_are_scored_separately():
    output = """**Source:** https://github.com/kubeflow/katib/blob/master/random.yaml

apiVersion: kubeflow.org/v1beta1
"""
    result = EVAL.score_retrieval(golden_row(), output)

    assert result.passed
    assert result.missing_sources == []
    assert result.missing_evidence == ["kind: Experiment"]


def test_retrieval_gate_blocks_absent_source_even_when_answer_terms_exist():
    output = "apiVersion: kubeflow.org/v1beta1\nkind: Experiment"
    result = EVAL.score_retrieval(golden_row(), output)

    assert not result.passed
    assert result.missing_sources == golden_row()["expected_source_urls"]
    assert result.missing_evidence == []


def test_extract_answer_urls_removes_markdown_delimiter():
    answer = (
        "Sources:\n- [Random](https://github.com/kubeflow/katib/blob/master/random.yaml)\n"
        "See https://www.kubeflow.org/docs."
    )
    assert EVAL.extract_answer_urls(answer) == {
        "https://github.com/kubeflow/katib/blob/master/random.yaml",
        "https://www.kubeflow.org/docs",
    }


def test_tool_detection_requires_a_tool_name_field():
    event = {"result": {"message": {"parts": [{"kind": "function_call", "name": "search_kubeflow_code"}]}}}
    assert EVAL.event_names_tool(event, "search_kubeflow_code")
    assert not EVAL.event_names_tool(
        {"result": {"message": {"parts": [{"kind": "text", "text": "search_kubeflow_code"}]}}},
        "search_kubeflow_code",
    )


def test_strings_in_event_finds_embedded_tool_sources():
    event = {
        "result": {
            "parts": [
                {
                    "kind": "function_response",
                    "response": "**Source:** https://example.test/source\n\nEvidence",
                }
            ]
        }
    }
    values = list(EVAL.strings_in_event(event))

    assert EVAL.SOURCE_RE.findall("\n".join(values)) == ["https://example.test/source"]


def test_tool_calls_in_event_extracts_named_tool_args_once_per_event_shape():
    event = {
        "result": {
            "message": {
                "parts": [{
                    "kind": "function_call",
                    "name": "search_github_issues",
                    "args": {"query": "deploymentMode Serverless", "repo": "kserve/kserve"},
                }]
            }
        }
    }

    assert EVAL.tool_calls_in_event(event, "search_github_issues") == [
        '{"query": "deploymentMode Serverless", "repo": "kserve/kserve"}'
    ]


def test_final_message_wins_over_partial_stream_fragments():
    assert EVAL.select_answer("authoritative final", ["partial ", "draft"]) == (
        "authoritative final"
    )
    assert EVAL.select_answer("", ["partial ", "fallback"]) == "partial fallback"


def test_sse_parser_joins_multiline_data_and_skips_comments():
    stream = io.BytesIO(
        b': keepalive\n\n'
        b'data: {"result":\n'
        b'data: {"final": true}}\n\n'
        b'data: [DONE]\n\n'
    )
    assert list(EVAL.iter_sse_events(stream)) == [{"result": {"final": True}}]
