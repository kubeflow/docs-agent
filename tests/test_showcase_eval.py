"""Deterministic tests for the two-gate showcase evaluator."""

import importlib.util
import io
import json
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


def test_retrieval_gate_reads_source_lines_from_structured_mcp_payload():
    output = json.dumps(
        {
            "markdown_summary": (
                "**Source:** https://github.com/kubeflow/katib/blob/master/random.yaml\n\n"
                "apiVersion: kubeflow.org/v1beta1\nkind: Experiment"
            ),
            "citations": [
                {
                    "url": "https://github.com/kubeflow/katib/blob/master/random.yaml",
                    "file": "random.yaml",
                }
            ],
        }
    )

    result = EVAL.score_retrieval(golden_row(), output)

    assert result.passed
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


def test_extracts_source_and_structured_citation_from_json_encoded_tool_result():
    url = "https://github.com/kubeflow/katib/blob/master/random.yaml"
    event = {
        "result": {
            "message": {
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "response": json.dumps(
                                {
                                    "markdown_summary": f"**Source:** {url}\n\nEvidence",
                                    "citations": [{"url": url, "file": "random.yaml"}],
                                }
                            )
                        },
                    }
                ]
            }
        }
    }

    payloads = list(EVAL.widget_tool_result_payloads(event))

    assert len(payloads) == 1
    assert EVAL.source_urls_in_event(payloads[0]) == {url}
    assert EVAL.structured_citation_urls(payloads[0]) == {url}


def test_widget_citation_gate_ignores_json_in_answer_text():
    url = "https://github.com/kubeflow/katib/blob/master/random.yaml"
    event = {
        "result": {
            "message": {
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": json.dumps(
                            {
                                "markdown_summary": f"**Source:** {url}",
                                "citations": [{"url": url}],
                            }
                        ),
                    }
                ],
            }
        }
    }

    assert list(EVAL.widget_tool_result_payloads(event)) == []


def test_widget_tool_payloads_include_legacy_function_results():
    payload = json.dumps(
        {
            "markdown_summary": "**Source:** https://www.kubeflow.org/docs/pipelines/",
            "citations": [{"url": "https://www.kubeflow.org/docs/pipelines/"}],
        }
    )
    event = {
        "result": {
            "message": {
                "metadata": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_kubeflow_docs",
                                "result": payload,
                            }
                        }
                    ]
                }
            }
        }
    }

    assert list(EVAL.widget_tool_result_payloads(event)) == [payload]


def test_structured_citation_extractor_ignores_unrelated_url_fields():
    event = {
        "result": {
            "url": "https://example.test/not-a-citation",
            "structuredContent": {"citations": [{"url": "https://www.kubeflow.org/docs/pipelines/"}]},
        }
    }

    assert EVAL.structured_citation_urls(event) == {"https://www.kubeflow.org/docs/pipelines/"}


def test_agent_gate_requires_expected_structured_citation():
    row = golden_row()
    result = EVAL.AgentResult(
        row=row,
        answer="apiVersion: kubeflow.org/v1beta1\nkind: Experiment",
        tool_fired=True,
        tool_calls=[],
        tool_source_urls=set(row["expected_source_urls"]),
        structured_citation_urls=set(),
        answer_urls=set(),
        invented_urls=[],
        unbacked_citations=[],
        missing_expected_citations=list(row["expected_source_urls"]),
        missing_answer_strings=[],
        present_forbidden_strings=[],
    )

    assert not result.passed


def test_tool_calls_in_event_extracts_named_tool_args_once_per_event_shape():
    event = {
        "result": {
            "message": {
                "parts": [
                    {
                        "kind": "function_call",
                        "name": "search_github_issues",
                        "args": {"query": "deploymentMode Serverless", "repo": "kserve/kserve"},
                    }
                ]
            }
        }
    }

    assert EVAL.tool_calls_in_event(event, "search_github_issues") == [
        '{"query": "deploymentMode Serverless", "repo": "kserve/kserve"}'
    ]


def test_final_message_wins_over_partial_stream_fragments():
    assert EVAL.select_answer("authoritative final", ["partial ", "draft"]) == ("authoritative final")
    assert EVAL.select_answer("", ["partial ", "fallback"]) == "partial fallback"


def test_sse_parser_joins_multiline_data_and_skips_comments():
    stream = io.BytesIO(b': keepalive\n\ndata: {"result":\ndata: {"final": true}}\n\ndata: [DONE]\n\n')
    assert list(EVAL.iter_sse_events(stream)) == [{"result": {"final": True}}]
