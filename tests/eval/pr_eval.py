"""Pre-merge eval of a pull request, reported as a PR comment.

A maintainer comments /eval on a PR (.github/workflows/pr-eval.yaml). That builds the
PR's MCP image and labels the PR `eval` and `eval-sha:<sha>`; Argo CD then deploys that
commit into the preview namespace and starts this script from main in
docs-agent-staging (see deploy/argocd/apps/docs-agent-pr-eval.yaml). It:

1. waits until every Deployment in the preview namespace is rolled out and the MCP
   server runs the PR's image,
2. runs the staging eval (staging_eval.evaluate) against staging, which runs main, and
   then against the preview, so both are measured with the same model, index and judge
   (main first, so nothing the PR deploys can load the shared services during main's run),
3. posts a comparison to the PR, with public Langfuse traces for every case,
4. removes both labels, so Argo CD tears the preview down.

The script, the dataset and the judge come from main, so a PR cannot change how it is
graded. Standard library only.

Environment (besides staging_eval's):
  PR_NUMBER, HEAD_SHA        the PR and the commit under test
  EXPECTED_IMAGE_TAG         MCP image tag the preview must run (pr-<n>-<sha8>)
  PREVIEW_NAMESPACE          default docs-agent-preview
  BASELINE_NAMESPACE         default docs-agent-staging
  MCP_DEPLOYMENT             default mcp-kubeflow-docs
  READY_TIMEOUT              seconds to wait for the preview (default 1200)
  GITHUB_TOKEN               may comment and edit labels on GITHUB_REPO (Secret
                             pr-eval-github, key token). Optional: without it, or if
                             GitHub rejects it, the report is only printed to the log
                             and the eval still succeeds; the labels then stay until
                             removed by hand or by the /eval workflow after 3 hours.
  EVAL_LABEL                 default eval
  LANGFUSE_UI_URL            public Langfuse URL used in links
"""

from __future__ import annotations

import json
import os
import re
import signal
import ssl
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

import staging_eval as se

PR_NUMBER = os.getenv("PR_NUMBER", "")
HEAD_SHA = os.getenv("HEAD_SHA", "")
EXPECTED_TAG = os.getenv("EXPECTED_IMAGE_TAG", "")
PREVIEW_NS = os.getenv("PREVIEW_NAMESPACE", "docs-agent-preview")
BASELINE_NS = os.getenv("BASELINE_NAMESPACE", "docs-agent-staging")
MCP_DEPLOYMENT = os.getenv("MCP_DEPLOYMENT", "mcp-kubeflow-docs")
READY_TIMEOUT = int(os.getenv("READY_TIMEOUT", "1200"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
EVAL_LABEL = os.getenv("EVAL_LABEL", "eval")
LANGFUSE_UI = os.getenv("LANGFUSE_UI_URL", "").rstrip("/")
MARKER = "<!-- flo-pr-eval -->"

GROUNDED_RANK = {"yes": 2, "partial": 1, "no": 0}


# ----- Kubernetes (service-account token, read-only) ---------------------------------------

SA = "/var/run/secrets/kubernetes.io/serviceaccount"


def kube_get(path: str) -> dict:
    with open(f"{SA}/token") as f:
        token = f.read().strip()
    ctx = ssl.create_default_context(cafile=f"{SA}/ca.crt")
    req = urllib.request.Request(f"https://kubernetes.default.svc{path}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
        return json.loads(r.read())


def image_tag(deployment: dict) -> str:
    image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
    return image.rsplit(":", 1)[-1] if ":" in image.rsplit("/", 1)[-1] else ""


def rolled_out(deployment: dict) -> bool:
    spec, status = deployment["spec"], deployment.get("status", {})
    want = spec.get("replicas", 1)
    return (
        status.get("observedGeneration", 0) >= deployment["metadata"].get("generation", 0)
        and status.get("updatedReplicas", 0) == want
        and status.get("availableReplicas", 0) == want
        and status.get("replicas", 0) == want
    )


def preview_problem(deployments: list[dict]) -> str | None:
    """Why the preview is not ready yet, or None when it is."""
    by_name = {d["metadata"]["name"]: d for d in deployments}
    mcp = by_name.get(MCP_DEPLOYMENT)
    if not mcp:
        return f"no {MCP_DEPLOYMENT} Deployment yet"
    if EXPECTED_TAG and image_tag(mcp) != EXPECTED_TAG:
        return f"{MCP_DEPLOYMENT} runs {image_tag(mcp) or '?'}, waiting for {EXPECTED_TAG}"
    for agent in se.AGENTS.values():
        if agent not in by_name:
            return f"no {agent} Deployment yet"
    for name, d in by_name.items():
        if not rolled_out(d):
            return f"{name} is still rolling out"
    return None


def wait_for_preview() -> None:
    deadline, problem = time.time() + READY_TIMEOUT, "not checked"
    while time.time() < deadline:
        try:
            problem = preview_problem(kube_get(f"/apis/apps/v1/namespaces/{PREVIEW_NS}/deployments")["items"])
        except (urllib.error.URLError, OSError, KeyError) as e:
            problem = f"cannot read Deployments: {e}"
        if problem is None:
            return
        print(f"preview not ready: {problem}")
        time.sleep(20)
    raise RuntimeError(f"preview {PREVIEW_NS} not ready after {READY_TIMEOUT}s: {problem}")


def baseline_revision() -> str:
    """Staging tags its MCP image with the short SHA of the main commit it runs."""
    try:
        return image_tag(kube_get(f"/apis/apps/v1/namespaces/{BASELINE_NS}/deployments/{MCP_DEPLOYMENT}"))
    except (urllib.error.URLError, OSError, KeyError) as e:
        print(f"warn: baseline revision unknown: {e}", file=sys.stderr)
        return ""


# ----- Comparison ----------------------------------------------------------------------


def case_state(row: dict) -> dict:
    judge = row.get("judge") or {}
    return {
        "routing": bool(row["routing"]),
        "fidelity": row.get("fidelity"),
        "grounded": judge.get("grounded"),
        "error": bool(row["out"].get("error")),
        "reason": judge.get("reason", ""),
        "tools": [c["name"] for c in row["out"]["calls"]],
    }


def case_score(state: dict) -> tuple:
    """Higher is better: no error, then routing, then query fidelity, then groundedness."""
    return (
        not state["error"],
        state["routing"],
        state["fidelity"] is not False,
        GROUNDED_RANK.get(state["grounded"] or "", -1),
    )


def compare(base_rows: list[dict], pr_rows: list[dict]) -> dict:
    base = {r["case"]["id"]: case_state(r) for r in base_rows}
    pr = {r["case"]["id"]: case_state(r) for r in pr_rows}
    worse, better = [], []
    for cid in pr:
        if cid not in base:
            continue
        b, p = case_score(base[cid]), case_score(pr[cid])
        if p < b:
            worse.append(cid)
        elif p > b:
            better.append(cid)
    return {"base": base, "pr": pr, "worse": worse, "better": better}


def fmt(v, digits=2) -> str:
    return "–" if v is None else f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def delta(b, p) -> str:
    if not isinstance(b, (int, float)) or not isinstance(p, (int, float)):
        return ""
    d = p - b
    if abs(d) < 1e-9:
        return "0"
    return f"{d:+.2f}"


def plain(text: str, limit: int = 160) -> str:
    """Judge output is model text about PR-controlled answers: no markup, links or mentions."""
    text = re.sub(r"<[^>]*>|!\[|\]\(", " ", text or "")
    text = text.replace("@", "@\u200b").replace("|", "\\|").replace("`", "'")
    return " ".join(text.split())[:limit]


def describe(state: dict) -> str:
    if state["error"]:
        return "error"
    route = "routing ok" if state["routing"] else f"routing ✗ ({', '.join(state['tools']) or 'no tool'})"
    query = ", query lost key terms" if state["fidelity"] is False else ""
    return f"{route}{query}, grounded {state['grounded'] or '–'}"


def trace_link(recorded: dict, cid: str, project_id: str, label: str = "trace") -> str:
    item = (recorded or {}).get("items", {}).get(cid) or {}
    trace = item.get("agent_trace") or item.get("trace")
    if not (trace and LANGFUSE_UI and project_id):
        return ""
    return f"[{label}]({LANGFUSE_UI}/project/{project_id}/traces/{trace})"


def render(pr: str, head: str, base_rev: str, base: dict, cand: dict, project_id: str) -> str:
    bs, ps = base["summary"], cand["summary"]
    diff = compare(base["rows"], cand["rows"])
    metrics = [
        ("Tool routing", "routing"),
        ("Query fidelity", "query_fidelity"),
        ("Grounded (judge)", "grounded"),
        ("Answers the question (judge)", "answers"),
        ("Errors", "errors"),
        ("Latency p50 (s)", "latency_p50_s"),
        ("Latency p95 (s)", "latency_p95_s"),
    ]
    lines = [
        MARKER,
        f"### Flo eval: #{pr} at `{head[:8]}` vs main at `{(base_rev or '?')[:8]}`",
        "",
        "| | main | this PR | Δ |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {label} | {fmt(bs.get(k))} | {fmt(ps.get(k))} | {delta(bs.get(k), ps.get(k))} |" for label, k in metrics
    ]
    gates = ps["gates"]
    lines += [
        "",
        f"**Gates** (routing ≥ {gates['min_routing']}, grounded ≥ {gates['min_grounded']}, no errors): "
        + ("✅ pass" if ps["passed"] else "❌ fail"),
        "",
        f"**{len(diff['worse'])} case(s) got worse, {len(diff['better'])} got better** out of {ps['cases']}."
        + (
            f" One case moves a rate by {1 / ps['cases']:.2f}, and the model and judge vary a little between"
            " runs, so a single changed case is weak evidence; look at what changed in its trace."
            if ps["cases"]
            else ""
        ),
    ]
    for title, ids in (("Got worse", diff["worse"]), ("Got better", diff["better"])):
        if not ids:
            continue
        lines += [
            "",
            f"<details><summary>{title} ({len(ids)})</summary>",
            "",
            "| case | main | this PR | judge on this PR | |",
            "|---|---|---|---|---|",
        ]
        for cid in ids:
            b, p = diff["base"][cid], diff["pr"][cid]
            reason = plain(p["reason"])
            lines.append(
                f"| `{cid}` | {describe(b)} | {describe(p)} | {reason} | {trace_link(cand.get('recorded'), cid, project_id)} |"
            )
        lines += ["", "</details>"]
    # Every case, so contributors can open any trace, not only the ones that changed.
    lines += [
        "",
        f"<details><summary>All cases ({len(diff['pr'])})</summary>",
        "",
        "| case | main | this PR | traces |",
        "|---|---|---|---|",
    ]
    for cid, p in diff["pr"].items():
        b = diff["base"].get(cid)
        traces = " · ".join(
            t
            for t in (
                trace_link(base.get("recorded"), cid, project_id, "main"),
                trace_link(cand.get("recorded"), cid, project_id, "this PR"),
            )
            if t
        )
        lines.append(f"| `{cid}` | {describe(b) if b else '–'} | {describe(p)} | {traces} |")
    lines += ["", "</details>"]
    links = []
    if LANGFUSE_UI and project_id and (cand.get("recorded") or {}).get("dataset_id"):
        links.append(
            f"[both runs in Langfuse]({LANGFUSE_UI}/project/{project_id}/datasets/{cand['recorded']['dataset_id']})"
            f" (experiments `{bs['run']}` and `{ps['run']}`)"
        )
    lines += [
        "",
        "Each trace link opens without a login and shows the agent's tool calls, the retrieved text and the"
        " answer. " + (" ".join(links) + " (maintainer login)." if links else ""),
        "",
        "<sub>Same model, index and judge for both runs; the dataset and this script come from main."
        " Commits pushed after `/eval` are not included; a maintainer can run `/eval` again.</sub>",
    ]
    return "\n".join(lines)


# ----- GitHub --------------------------------------------------------------------------


def github(method: str, path: str, body=None):
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {GITHUB_TOKEN}"}
    return se.http(method, f"https://api.github.com/repos/{se.REPO}{path}", body, headers, timeout=30)


def comment(text: str) -> None:
    """Best effort: the report is always in the log; posting it must not fail the eval."""
    print(text)
    if not GITHUB_TOKEN:
        print("warn: no GITHUB_TOKEN (Secret pr-eval-github), report not posted", file=sys.stderr)
        return
    try:
        github("POST", f"/issues/{PR_NUMBER}/comments", {"body": text})
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"warn: report not posted: {e}", file=sys.stderr)


def release_preview() -> None:
    """Removing the label makes the ApplicationSets delete the preview and this Job."""
    if not GITHUB_TOKEN:
        return
    for label in (EVAL_LABEL, f"eval-sha:{HEAD_SHA}"):
        try:
            github("DELETE", f"/issues/{PR_NUMBER}/labels/{urllib.parse.quote(label)}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"warn: could not remove label {label}: {e}", file=sys.stderr)


def langfuse_project_id() -> str:
    try:
        return (se.lf("GET", "/projects").get("data") or [{}])[0].get("id", "")
    except Exception as e:  # links are a convenience
        print(f"warn: Langfuse project id: {e}", file=sys.stderr)
        return ""


# ----- Main ----------------------------------------------------------------------------


def main() -> int:
    if not (PR_NUMBER and HEAD_SHA):
        print("PR_NUMBER and HEAD_SHA are required", file=sys.stderr)
        return 2
    # activeDeadlineSeconds and evictions stop the pod with SIGTERM; turn it into an
    # exception so the finally block still removes the label.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))
    try:
        wait_for_preview()
        base_rev = baseline_revision()
        # Recorded apart from the post-merge staging runs that promotions are judged on.
        se.ENVIRONMENT, se.PUBLIC_TRACES = "pr-eval", True
        base = se.evaluate(BASELINE_NS, base_rev, run_prefix=f"pr-{PR_NUMBER} main ")
        cand = se.evaluate(PREVIEW_NS, HEAD_SHA, run_prefix=f"pr-{PR_NUMBER} ")
        comment(render(PR_NUMBER, HEAD_SHA, base_rev, base, cand, langfuse_project_id()))
        return 0
    except (Exception, SystemExit) as e:
        traceback.print_exc()
        comment(f"{MARKER}\n### Flo eval for `{HEAD_SHA[:8]}` could not run\n\n```\n{str(e)[:800]}\n```")
        return 1
    finally:
        release_preview()


if __name__ == "__main__":
    sys.exit(main())
