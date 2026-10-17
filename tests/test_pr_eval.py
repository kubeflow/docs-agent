"""Deterministic tests for the pre-merge PR eval (comparison, readiness, report)."""

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

import pr_eval  # noqa: E402


def row(cid, routing=True, grounded="yes", error=None, tools=("search_kubeflow_docs",), reason="", fidelity=None):
    return {
        "case": {"id": cid},
        "routing": routing,
        "fidelity": fidelity,
        "judge": {"grounded": grounded, "answers": "yes", "reason": reason} if grounded else None,
        "out": {"error": error, "calls": [{"name": t, "args": {}} for t in tools]},
    }


def summary(**overrides):
    base = {
        "run": "abc1234 20261017",
        "cases": 3,
        "errors": 0,
        "routing": 1.0,
        "query_fidelity": 1.0,
        "grounded": 1.0,
        "answers": 1.0,
        "latency_p50_s": 5.0,
        "latency_p95_s": 9.0,
        "passed": True,
        "gates": {"min_routing": 0.85, "min_grounded": 0.65},
    }
    return base | overrides


def deployment(name, tag="pr-7-abcdef12", replicas=1, available=1, updated=1, generation=2, observed=2):
    return {
        "metadata": {"name": name, "generation": generation},
        "spec": {
            "replicas": replicas,
            "template": {"spec": {"containers": [{"image": f"ghcr.io/kubeflow/{name}:{tag}"}]}},
        },
        "status": {
            "replicas": replicas,
            "availableReplicas": available,
            "updatedReplicas": updated,
            "observedGeneration": observed,
        },
    }


def ready_preview():
    return [
        deployment("mcp-kubeflow-docs"),
        deployment("kubeflow-docs-agent", tag="0.7"),
        deployment("kubeflow-debug-agent", tag="0.7"),
    ]


def test_compare_finds_regressions_and_improvements():
    base = [row("a"), row("b", grounded="partial"), row("c"), row("d")]
    pr = [row("a", routing=False), row("b"), row("c"), row("d", error="timeout")]
    diff = pr_eval.compare(base, pr)
    assert diff["worse"] == ["a", "d"]
    assert diff["better"] == ["b"]


def test_cases_missing_from_main_are_not_compared():
    diff = pr_eval.compare([row("a")], [row("a"), row("new", routing=False)])
    assert diff["worse"] == [] and diff["better"] == []


def test_preview_waits_for_the_pr_image(monkeypatch):
    monkeypatch.setattr(pr_eval, "EXPECTED_TAG", "pr-7-abcdef12")
    deps = ready_preview()
    assert pr_eval.preview_problem(deps) is None
    deps[0] = deployment("mcp-kubeflow-docs", tag="pr-6-00000000")
    assert "waiting for pr-7-abcdef12" in pr_eval.preview_problem(deps)


def test_preview_waits_for_rollout_and_agents(monkeypatch):
    monkeypatch.setattr(pr_eval, "EXPECTED_TAG", "pr-7-abcdef12")
    rolling = ready_preview()
    rolling[1] = deployment("kubeflow-docs-agent", tag="0.7", observed=1)
    assert "rolling out" in pr_eval.preview_problem(rolling)
    assert "kubeflow-debug-agent" in pr_eval.preview_problem(ready_preview()[:2])


def test_report_shows_deltas_gates_and_changed_cases(monkeypatch):
    monkeypatch.setattr(pr_eval, "LANGFUSE_UI", "https://langfuse.example.org")
    base = {"summary": summary(), "rows": [row("a"), row("b")], "recorded": {}}
    cand = {
        "summary": summary(routing=0.5, grounded=0.5, passed=False),
        "rows": [row("a", routing=False, tools=(), reason="answered without searching"), row("b")],
        "recorded": {"dataset_id": "ds1", "items": {"a": {"trace": "t1", "agent_trace": "at1"}}},
    }
    text = pr_eval.render("7", "abcdef1234567", "1234567", base, cand, "proj1")
    assert text.startswith(pr_eval.MARKER)
    assert "| Tool routing | 1.00 | 0.50 | -0.50 |" in text
    assert "❌ fail" in text
    assert "1 case(s) got worse, 0 got better" in text
    assert "routing ✗ (no tool)" in text
    assert "https://langfuse.example.org/project/proj1/traces/at1" in text
    assert "answered without searching" in text


def test_posting_is_best_effort(monkeypatch, capsys):
    monkeypatch.setattr(pr_eval, "GITHUB_TOKEN", "")
    pr_eval.comment("report")
    assert "report" in capsys.readouterr().out

    def rejected(*_args, **_kwargs):
        raise pr_eval.urllib.error.HTTPError("u", 403, "Forbidden", {}, None)

    monkeypatch.setattr(pr_eval, "GITHUB_TOKEN", "bad")
    monkeypatch.setattr(pr_eval, "github", rejected)
    pr_eval.comment("report")  # must not raise
    pr_eval.release_preview()  # must not raise
    assert "report not posted" in capsys.readouterr().err


def test_image_tag_ignores_registry_ports():
    assert pr_eval.image_tag(deployment("x", tag="pr-1-aaaa")) == "pr-1-aaaa"
    d = deployment("x")
    d["spec"]["template"]["spec"]["containers"][0]["image"] = "registry:5000/x"
    assert pr_eval.image_tag(d) == ""


def test_judge_reasons_cannot_ping_or_embed():
    text = pr_eval.plain("Fine <img src=x> ![p](http://e/x.png) cc @maintainer | `x`")
    assert "<" not in text and "![" not in text and "](" not in text
    assert "@maintainer" not in text and "@​maintainer" in text
    assert "\\|" in text and "`" not in text


def test_lost_query_terms_count_as_worse():
    diff = pr_eval.compare([row("a", fidelity=True)], [row("a", fidelity=False)])
    assert diff["worse"] == ["a"]
    assert "query lost key terms" in pr_eval.describe(diff["pr"]["a"])


def test_report_links_every_case_even_when_nothing_changed(monkeypatch):
    monkeypatch.setattr(pr_eval, "LANGFUSE_UI", "https://langfuse.example.org")
    rec = {"dataset_id": "ds1", "items": {"a": {"trace": "t", "agent_trace": "at-main"}}}
    base = {"summary": summary(), "rows": [row("a")], "recorded": rec}
    cand = {
        "summary": summary(cases=1),
        "rows": [row("a")],
        "recorded": {"dataset_id": "ds1", "items": {"a": {"trace": "t2", "agent_trace": "at-pr"}}},
    }
    text = pr_eval.render("7", "abcdef12", "1234567", base, cand, "proj1")
    assert "All cases (1)" in text
    assert "[main](https://langfuse.example.org/project/proj1/traces/at-main)" in text
    assert "[this PR](https://langfuse.example.org/project/proj1/traces/at-pr)" in text
    assert "weak evidence" in text
