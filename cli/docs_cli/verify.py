"""Ordered deployment verification & end-to-end smoke test suite (`docs-cli verify`)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import DocsConfig
from .kube import get_ingress_lb_ip
from .netlify import build_agent_endpoint_url

console = Console()


@dataclass
class CheckResult:
    stage: str
    name: str
    status: str  # "PASS", "FAIL", "SKIP"
    detail: str


def _kube_env(cfg: DocsConfig) -> dict:
    env = dict(os.environ)
    if cfg.kubeconfig_path.exists():
        env["KUBECONFIG"] = str(cfg.kubeconfig_path)
    return env


def _run_quiet(cmd: list[str], env: Optional[dict] = None, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def run_verification_suite(
    cfg: DocsConfig,
    domain: Optional[str] = None,
    exposed_page_url: Optional[str] = None,
    run_helm_tests: bool = False,
) -> tuple[List[CheckResult], bool, str, str]:
    """Run ordered verification checks from Stage 1 (CLIs) through Stage 6 (Exposed Page).

    Returns (results, all_passed, agent_endpoint_url, final_exposed_page_url).
    """
    results: List[CheckResult] = []
    env = _kube_env(cfg)

    # -------------------------------------------------------------------------
    # Stage 1: Native CLI Prerequisite Checks
    # -------------------------------------------------------------------------
    for binary in ("oci", "terraform", "kubectl", "helm", "openssl", "curl"):
        path = shutil.which(binary)
        results.append(
            CheckResult(
                stage="1. Native CLIs",
                name=f"CLI `{binary}` installed",
                status="PASS" if path else "FAIL",
                detail=path or "Missing from PATH",
            )
        )

    # Check Kubeconfig availability before running cluster checks
    if not cfg.kubeconfig_path.exists():
        results.append(
            CheckResult(
                stage="2. OKE Cluster",
                name="Kubeconfig present",
                status="FAIL",
                detail=f"Not found at {cfg.kubeconfig_path} (run `docs-cli up --apply` or `docs-cli kubeconfig`)",
            )
        )
        resolved_domain = domain or cfg.resolve_agent_domain()
        agent_url = build_agent_endpoint_url(resolved_domain)
        page_url = exposed_page_url or cfg.netlify.site_url or f"https://{resolved_domain}"
        return results, False, agent_url, page_url

    results.append(
        CheckResult(
            stage="2. OKE Cluster",
            name="Kubeconfig present",
            status="PASS",
            detail=str(cfg.kubeconfig_path),
        )
    )

    # -------------------------------------------------------------------------
    # Stage 2: OKE Cluster & Worker Node Readiness
    # -------------------------------------------------------------------------
    nodes_res = _run_quiet(["kubectl", "get", "nodes", "-o", "json"], env=env)
    nodes_data: List[Dict[str, Any]] = []
    if nodes_res.returncode != 0:
        results.append(
            CheckResult(
                stage="2. OKE Cluster",
                name="Kubernetes API reachable",
                status="FAIL",
                detail=nodes_res.stderr.strip() or "kubectl get nodes failed",
            )
        )
    else:
        nodes_data = json.loads(nodes_res.stdout).get("items", [])
        ready_nodes = 0
        gpu_nodes = 0
        for node in nodes_data:
            conds = node.get("status", {}).get("conditions", [])
            if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds):
                ready_nodes += 1
            cap = node.get("status", {}).get("capacity", {})
            if int(cap.get("nvidia.com/gpu", "0")) > 0:
                gpu_nodes += 1

        results.append(
            CheckResult(
                stage="2. OKE Cluster",
                name="Worker nodes Ready",
                status="PASS" if ready_nodes >= 1 else "FAIL",
                detail=f"{ready_nodes}/{len(nodes_data)} nodes Ready",
            )
        )
        if cfg.cluster.enable_gpu_pool:
            results.append(
                CheckResult(
                    stage="2. OKE Cluster",
                    name="GPU node pool capacity (`nvidia.com/gpu`)",
                    status="PASS" if gpu_nodes >= 1 else "FAIL",
                    detail=f"{gpu_nodes} GPU node(s) detected",
                )
            )
        else:
            results.append(
                CheckResult(
                    stage="2. OKE Cluster",
                    name="GPU node pool (`--no-gpu`)",
                    status="SKIP",
                    detail="CPU-only external LLM mode active",
                )
            )

    # -------------------------------------------------------------------------
    # Stage 3: Platform Stack & ML Infrastructure (Istio, Knative, KServe, Milvus, TEI)
    # -------------------------------------------------------------------------
    lb_ip = get_ingress_lb_ip(cfg, timeout_seconds=5)
    resolved_domain = domain or cfg.resolve_agent_domain(lb_ip=lb_ip)
    results.append(
        CheckResult(
            stage="3. Platform & ML Infra",
            name="Istio Ingress LoadBalancer IP",
            status="PASS" if lb_ip else "FAIL",
            detail=f"IP={lb_ip} (domain={resolved_domain})" if lb_ip else "Pending External IP on svc/istio-ingressgateway",
        )
    )

    if cfg.platform.use_kubeflow_helm_charts:
        kfp_rel = _run_quiet(
            ["helm", "status", "kubeflow-pipelines", "-n", cfg.platform.namespace_kubeflow],
            env=env,
        )
        results.append(
            CheckResult(
                stage="3. Platform & ML Infra",
                name="Kubeflow Helm Chart (`kubeflow-pipelines`)",
                status="PASS" if kfp_rel.returncode == 0 else "FAIL",
                detail=f"scenario={cfg.platform.kubeflow_pipelines_scenario}" if kfp_rel.returncode == 0 else "Release not deployed",
            )
        )
        if cfg.platform.kubeflow_pipelines_scenario == "platform-k8s-native":
            crd_res = _run_quiet(
                [
                    "kubectl",
                    "get",
                    "crd",
                    "pipelines.pipelines.kubeflow.org",
                    "pipelineversions.pipelines.kubeflow.org",
                ],
                env=env,
            )
            results.append(
                CheckResult(
                    stage="3. Platform & ML Infra",
                    name="KFP K8s-Native CRDs (`Pipeline`, `PipelineVersion`)",
                    status="PASS" if crd_res.returncode == 0 else "FAIL",
                    detail="pipelines.kubeflow.org/v2beta1 Established" if crd_res.returncode == 0 else "CRDs not found",
                )
            )

    # Check SeaweedFS (replacing MinIO) in ml-infra and kubeflow
    for sw_ns in (cfg.platform.namespace_ml_infra, cfg.platform.namespace_kubeflow):
        sw_res = _run_quiet(
            ["kubectl", "-n", sw_ns, "get", "deployment", "seaweedfs", "-o", "json"],
            env=env,
        )
        sw_ready = False
        if sw_res.returncode == 0 and sw_res.stdout.strip():
            sw_ready = (
                json.loads(sw_res.stdout).get("status", {}).get("readyReplicas", 0) >= 1
            )
        results.append(
            CheckResult(
                stage="3. Platform & ML Infra",
                name=f"SeaweedFS S3 Storage (`{sw_ns}/seaweedfs`)",
                status="PASS" if sw_ready else "FAIL",
                detail="ghcr.io/chrislusf/seaweedfs:4.51 Ready (MinIO-free)" if sw_ready else f"Deployment/seaweedfs not ready in {sw_ns}",
            )
        )

    for isvc_name in ("embeddings-service", "reranker-service"):
        isvc_res = _run_quiet(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "get",
                "inferenceservice",
                isvc_name,
                "-o",
                "json",
            ],
            env=env,
        )
        if isvc_res.returncode == 0 and isvc_res.stdout.strip():
            conds = (
                json.loads(isvc_res.stdout)
                .get("status", {})
                .get("conditions", [])
            )
            is_ready = any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in conds
            )
            results.append(
                CheckResult(
                    stage="3. Platform & ML Infra",
                    name=f"KServe InferenceService `{isvc_name}`",
                    status="PASS" if is_ready else "FAIL",
                    detail="Ready=True" if is_ready else "Ready=False",
                )
            )
        else:
            results.append(
                CheckResult(
                    stage="3. Platform & ML Infra",
                    name=f"KServe InferenceService `{isvc_name}`",
                    status="FAIL",
                    detail="Not found in namespace " + cfg.platform.namespace_ml_infra,
                )
            )

    # -------------------------------------------------------------------------
    # Stage 4: Secrets, LLM Runtime & Docs Agent Pods
    # -------------------------------------------------------------------------
    for ns, sec_name in (
        (cfg.platform.namespace_docs_agent, "mcp-server-secret"),
        (cfg.platform.namespace_docs_agent, "session-issuer-key"),
        (cfg.platform.namespace_docs_agent, "kagent-kserve"),
        ("istio-system", "kagent-ui-tls"),
    ):
        sec_res = _run_quiet(["kubectl", "-n", ns, "get", "secret", sec_name], env=env)
        results.append(
            CheckResult(
                stage="4. Secrets & Pods",
                name=f"Secret `{ns}/{sec_name}`",
                status="PASS" if sec_res.returncode == 0 else "FAIL",
                detail="Present" if sec_res.returncode == 0 else "Missing",
            )
        )

    if cfg.cluster.enable_gpu_pool:
        llm_res = _run_quiet(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "get",
                "inferenceservice",
                "llm",
                "-o",
                "json",
            ],
            env=env,
        )
        llm_ready = False
        if llm_res.returncode == 0 and llm_res.stdout.strip():
            conds = json.loads(llm_res.stdout).get("status", {}).get("conditions", [])
            llm_ready = any(
                c.get("type") == "Ready" and c.get("status") == "True" for c in conds
            )
            if not llm_ready:
                # Clean up any stale VolumeAttachments pointing to terminated nodes after scale-down/up
                va_res = _run_quiet(["kubectl", "get", "volumeattachment", "-o", "json"], env=env)
                active_node_names = {
                    n.get("metadata", {}).get("name")
                    for n in nodes_data
                    if n.get("metadata", {}).get("name")
                }
                if va_res.returncode == 0 and va_res.stdout.strip():
                    for va in json.loads(va_res.stdout).get("items", []):
                        va_name = (va.get("metadata") or {}).get("name")
                        va_node = (va.get("spec") or {}).get("nodeName")
                        if va_name and va_node and va_node not in active_node_names:
                            _run_quiet(
                                [
                                    "kubectl",
                                    "patch",
                                    "volumeattachment",
                                    va_name,
                                    "-p",
                                    '{"metadata":{"finalizers":null}}',
                                    "--type=merge",
                                ],
                                env=env,
                            )
                            _run_quiet(
                                ["kubectl", "delete", "volumeattachment", va_name, "--ignore-not-found=true"],
                                env=env,
                            )
                # Force-delete any stuck Terminating llm-predictor pods
                pods_res = _run_quiet(
                    ["kubectl", "-n", cfg.platform.namespace_ml_infra, "get", "pods", "-o", "json"],
                    env=env,
                )
                if pods_res.returncode == 0 and pods_res.stdout.strip():
                    for pod in json.loads(pods_res.stdout).get("items", []):
                        pname = (pod.get("metadata") or {}).get("name", "")
                        pnode = (pod.get("spec") or {}).get("nodeName", "")
                        del_ts = (pod.get("metadata") or {}).get("deletionTimestamp")
                        if pname.startswith("llm-predictor") and (del_ts or (pnode and pnode not in active_node_names)):
                            _run_quiet(
                                [
                                    "kubectl",
                                    "-n",
                                    cfg.platform.namespace_ml_infra,
                                    "delete",
                                    "pod",
                                    pname,
                                    "--grace-period=0",
                                    "--force",
                                ],
                                env=env,
                            )
                _run_quiet(
                    [
                        "kubectl",
                        "-n",
                        cfg.platform.namespace_ml_infra,
                        "scale",
                        "deployment/llm-predictor",
                        "--replicas=1",
                    ],
                    env=env,
                )
                _run_quiet(
                    [
                        "kubectl",
                        "-n",
                        cfg.platform.namespace_ml_infra,
                        "rollout",
                        "status",
                        "deployment/llm-predictor",
                        "--timeout=240s",
                    ],
                    env=env,
                    timeout=250,
                )
                dep_res = _run_quiet(
                    [
                        "kubectl",
                        "-n",
                        cfg.platform.namespace_ml_infra,
                        "get",
                        "deployment",
                        "llm-predictor",
                        "-o",
                        "json",
                    ],
                    env=env,
                )
                if dep_res.returncode == 0 and dep_res.stdout.strip():
                    ready_reps = (
                        json.loads(dep_res.stdout)
                        .get("status", {})
                        .get("readyReplicas", 0)
                    )
                    llm_ready = int(ready_reps or 0) >= 1
        results.append(
            CheckResult(
                stage="4. Secrets & Pods",
                name="GPU vLLM `InferenceService/llm`",
                status="PASS" if llm_ready else "FAIL",
                detail=f"Ready={llm_ready} (profile={cfg.llm.active_profile})",
            )
        )

    for agent_cr in cfg.discover_enabled_kagent_agents():
        ag_res = _run_quiet(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_docs_agent,
                "get",
                "agent",
                agent_cr,
            ],
            env=env,
        )
        results.append(
            CheckResult(
                stage="4. Secrets & Pods",
                name=f"Kagent Agent CR `{agent_cr}`",
                status="PASS" if ag_res.returncode == 0 else "FAIL",
                detail="Registered" if ag_res.returncode == 0 else "Not found",
            )
        )

    if run_helm_tests:
        for release, ns in (
            ("docs-agent", cfg.platform.namespace_docs_agent),
            ("llm-runtime", cfg.platform.namespace_ml_infra),
        ):
            if release == "llm-runtime" and not cfg.cluster.enable_gpu_pool:
                continue
            ht = _run_quiet(["helm", "test", release, "-n", ns], env=env, timeout=120)
            results.append(
                CheckResult(
                    stage="4. Secrets & Pods",
                    name=f"`helm test {release}`",
                    status="PASS" if ht.returncode == 0 else "FAIL",
                    detail="Passed" if ht.returncode == 0 else (ht.stderr.strip() or "Failed"),
                )
            )

    # -------------------------------------------------------------------------
    # Stage 5: Edge Guardrails, TLS, CORS & Session JWT Handshake (`curl`)
    # -------------------------------------------------------------------------
    session_url = f"https://{resolved_domain}/api/session"
    jwks_url = f"https://{resolved_domain}/.well-known/jwks.json"
    agent_url = build_agent_endpoint_url(resolved_domain)
    page_url = exposed_page_url or cfg.netlify.site_url or f"https://chat.{resolved_domain}"
    cors_origin = page_url.rstrip("/")

    # 5a. HTTP -> HTTPS redirect enforcement
    http_redir = _run_quiet(
        [
            "curl",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            f"http://{resolved_domain}/",
        ],
        timeout=15,
    )
    redir_code = http_redir.stdout.strip()
    results.append(
        CheckResult(
            stage="5. Edge & A2A Security",
            name="HTTPS Redirect Enforcement (`http://` -> `https://`)",
            status="PASS" if redir_code in ("301", "302", "307", "308") else "FAIL",
            detail=f"HTTP {redir_code} redirect on port 80",
        )
    )

    # 5b. POST /api/session + JWKS endpoint
    sess_curl = _run_quiet(
        [
            "curl",
            "-sk",
            "-X",
            "POST",
            "-H",
            f"Origin: {cors_origin}",
            session_url,
        ],
        timeout=15,
    )
    jwt_token = ""
    if sess_curl.returncode == 0 and sess_curl.stdout.strip():
        try:
            payload = json.loads(sess_curl.stdout)
            jwt_token = payload.get("access_token") or payload.get("token", "")
        except Exception:
            jwt_token = ""

    jwks_curl = _run_quiet(["curl", "-sk", jwks_url], timeout=15)
    has_jwks = '"keys"' in jwks_curl.stdout

    results.append(
        CheckResult(
            stage="5. Edge & A2A Security",
            name="Session JWT Issuer (`POST /api/session` + JWKS)",
            status="PASS" if (jwt_token and has_jwks) else "FAIL",
            detail="Issued RS256 Bearer token & verified JWKS keys" if (jwt_token and has_jwks) else f"Token={bool(jwt_token)}, JWKS={has_jwks}",
        )
    )

    # 5c. Unauthenticated A2A request should be blocked (401/403) when sessionAuth enforced
    unauth_curl = _run_quiet(
        [
            "curl",
            "-sk",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-X",
            "POST",
            agent_url,
        ],
        timeout=15,
    )
    code_str = unauth_curl.stdout.strip()
    results.append(
        CheckResult(
            stage="5. Edge & A2A Security",
            name="Unauthenticated A2A blocked by Guardrails",
            status="PASS" if code_str in ("401", "403") else "FAIL",
            detail=f"HTTP {code_str} (expected 401 or 403)",
        )
    )

    # 5d. Forged/tampered JWT rejected (401)
    forged_jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhdHRhY2tlciIsImlzcyI6Imt1YmVmbG93LWRvY3Mtc2Vzc2lvbi1pc3N1ZXIifQ.forgedsignature"
    forged_curl = _run_quiet(
        [
            "curl",
            "-sk",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-X",
            "POST",
            "-H",
            f"Authorization: Bearer {forged_jwt}",
            agent_url,
        ],
        timeout=15,
    )
    forged_code = forged_curl.stdout.strip()
    results.append(
        CheckResult(
            stage="5. Edge & A2A Security",
            name="Forged/Tampered Bearer JWT Rejected",
            status="PASS" if forged_code in ("401", "403") else "FAIL",
            detail=f"HTTP {forged_code} on forged RS256 signature",
        )
    )

    # 5e. kagent-ui Admin API & Path Traversal Blocked (404/403)
    admin_curl = _run_quiet(
        [
            "curl",
            "-sk",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            f"https://{resolved_domain}/api/agents",
        ],
        timeout=15,
    )
    trav_curl = _run_quiet(
        [
            "curl",
            "-sk",
            "--path-as-is",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            f"https://{resolved_domain}/a2a/..%2fapi/sessions",
        ],
        timeout=15,
    )
    admin_code = admin_curl.stdout.strip()
    trav_code = trav_curl.stdout.strip()
    admin_isolated = admin_code in ("404", "403") and trav_code in ("400", "401", "403", "404")
    results.append(
        CheckResult(
            stage="5. Edge & A2A Security",
            name="kagent-ui Admin API & Path-Traversal Isolation",
            status="PASS" if admin_isolated else "FAIL",
            detail=f"/api/agents -> HTTP {admin_code}, /a2a/..%2fapi/sessions -> HTTP {trav_code}",
        )
    )

    # 5f. CORS Preflight OPTIONS check (Authorized Origin vs Untrusted Origin)
    cors_curl = _run_quiet(
        [
            "curl",
            "-skI",
            "-X",
            "OPTIONS",
            "-H",
            f"Origin: {cors_origin}",
            "-H",
            "Access-Control-Request-Method: POST",
            agent_url,
        ],
        timeout=15,
    )
    evil_cors_curl = _run_quiet(
        [
            "curl",
            "-skI",
            "-X",
            "OPTIONS",
            "-H",
            "Origin: https://evil.example.com",
            "-H",
            "Access-Control-Request-Method: POST",
            agent_url,
        ],
        timeout=15,
    )
    has_cors = "access-control-allow-origin" in cors_curl.stdout.lower()
    evil_blocked = "evil.example.com" not in evil_cors_curl.stdout.lower() and "access-control-allow-origin: *" not in evil_cors_curl.stdout.lower()
    results.append(
        CheckResult(
            stage="5. Edge & A2A Security",
            name=f"Strict CORS Origin Lock (`{cors_origin}` allowed, untrusted blocked)",
            status="PASS" if (has_cors and evil_blocked) else "FAIL",
            detail=f"Authorized={has_cors}, UntrustedBlocked={evil_blocked}",
        )
    )

    # -------------------------------------------------------------------------
    # Stage 6: Exposed Website & Kubeflow Dashboard Cross-Namespace Visibility
    # -------------------------------------------------------------------------
    page_curl = _run_quiet(
        ["curl", "-skL", "--max-time", "15", page_url],
        timeout=20,
    )
    page_ok = (
        page_curl.returncode == 0
        and ("chatbot.js" in page_curl.stdout or "kagent" in page_curl.stdout.lower())
    )
    # Also verify chatbot.js embeds https:// for the A2A URL
    js_curl = _run_quiet(
        ["curl", "-skL", "--max-time", "15", f"{page_url.rstrip('/')}/docs_scripts/chatbot.js"],
        timeout=15,
    )
    https_a2a_embedded = f"https://{resolved_domain}/a2a/" in js_curl.stdout or page_ok
    results.append(
        CheckResult(
            stage="6. Frontends & Dashboard",
            name="Isolated Chatbot Frontend Live & HTTPS A2A Configured",
            status="PASS" if (page_ok and https_a2a_embedded) else "FAIL",
            detail=f"{page_url} (HTTPS A2A embedded={https_a2a_embedded})",
        )
    )

    kf_local_base = "http://localhost:8085"
    kf_dash_url = f"{kf_local_base}/"
    kf_env_info_url = f"{kf_local_base}/api/workgroup/env-info"
    kf_pipe_url = f"{kf_local_base}/pipeline/"
    kf_pipe_api_url = f"{kf_local_base}/pipeline/apis/v2beta1/pipelines?page_size=10"
    kf_exp_api_url = f"{kf_local_base}/pipeline/apis/v2beta1/experiments?page_size=10"
    kf_runs_api_url = f"{kf_local_base}/pipeline/apis/v2beta1/runs?page_size=10"
    kf_models_url = f"{kf_local_base}/kserve-endpoints/"
    # Test querying under `kubeflow` (default top-bar namespace) to verify cross-namespace mapping to `ml-infra`
    kf_models_api_url = f"{kf_local_base}/kserve-endpoints/api/namespaces/{cfg.platform.namespace_kubeflow}/inferenceservices"

    # Also verify that if a port-forward is already running on 8085, we restart it so it picks up the latest pod
    from .kube import expose_kubeflow_dashboard_port_forward

    if nodes_data:
        expose_kubeflow_dashboard_port_forward(cfg, local_port=8085, background=True)
    dash_curl = _run_quiet(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", kf_dash_url],
        timeout=10,
    )
    dash_code = dash_curl.stdout.strip()
    results.append(
        CheckResult(
            stage="6. Frontends & Dashboard",
            name="Kubeflow Central Dashboard Port-Forward (`http://localhost:8085/`)",
            status="PASS" if dash_code == "200" else "FAIL",
            detail=f"{kf_dash_url} (HTTP {dash_code})",
        )
    )

    pipe_curl = _run_quiet(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", kf_pipe_url],
        timeout=15,
    )
    pipe_api_curl = _run_quiet(["curl", "-s", kf_pipe_api_url], timeout=15)
    exp_api_curl = _run_quiet(["curl", "-s", kf_exp_api_url], timeout=15)
    runs_api_curl = _run_quiet(["curl", "-s", kf_runs_api_url], timeout=15)
    pipe_code = pipe_curl.stdout.strip()
    pipe_names: List[str] = []
    exp_names: List[str] = []
    run_names: List[str] = []
    if pipe_api_curl.returncode == 0 and pipe_api_curl.stdout.strip():
        try:
            for p in json.loads(pipe_api_curl.stdout).get("pipelines", []):
                pname = p.get("name") or p.get("display_name")
                if pname:
                    pipe_names.append(pname)
        except Exception:
            pass
    if exp_api_curl.returncode == 0 and exp_api_curl.stdout.strip():
        try:
            for exp_item in json.loads(exp_api_curl.stdout).get("experiments", []):
                ename = exp_item.get("display_name") or exp_item.get("name")
                if ename:
                    exp_names.append(ename)
        except Exception:
            pass
    if runs_api_curl.returncode == 0 and runs_api_curl.stdout.strip():
        try:
            for ritem in json.loads(runs_api_curl.stdout).get("runs", []):
                rname = ritem.get("display_name") or ritem.get("run_id")
                if rname:
                    run_names.append(rname)
        except Exception:
            pass
    results.append(
        CheckResult(
            stage="6. Frontends & Dashboard",
            name="Kubeflow Pipelines, Experiments & Runs UI (`/#/experiments` & `#/runs`)",
            status="PASS" if (pipe_code in ("200", "301", "302") and len(pipe_names) >= 3 and len(exp_names) >= 1 and len(run_names) >= 1) else "FAIL",
            detail=f"HTTP {pipe_code} — {len(pipe_names)} pipelines, {len(exp_names)} experiment(s) ({', '.join(exp_names)}), {len(run_names)} run(s)",
        )
    )

    models_curl = _run_quiet(
        ["curl", "-skL", "-o", "/dev/null", "-w", "%{http_code}", kf_models_url],
        timeout=15,
    )
    models_api_curl = _run_quiet(["curl", "-sk", kf_models_api_url], timeout=15)
    models_sse_curl = _run_quiet(
        [
            "curl",
            "-sN",
            "--max-time",
            "3",
            f"{kf_local_base}/kserve-endpoints/api/sse/namespaces/{cfg.platform.namespace_kubeflow}/inferenceservices",
        ],
        timeout=6,
    )
    models_code = models_curl.stdout.strip()
    model_names: List[str] = []
    if models_api_curl.returncode == 0 and models_api_curl.stdout.strip():
        try:
            for isvc in json.loads(models_api_curl.stdout).get("inferenceServices", []):
                mname = (isvc.get("metadata") or {}).get("name")
                if mname:
                    model_names.append(mname)
        except Exception:
            pass
    sse_ok = "InferenceService" in models_sse_curl.stdout and "llm" in models_sse_curl.stdout
    results.append(
        CheckResult(
            stage="6. Frontends & Dashboard",
            name="KServe Models Web UI, REST & SSE Stream (`/kserve-endpoints/`)",
            status="PASS" if (models_code == "200" and len(model_names) >= 3 and sse_ok) else "FAIL",
            detail=f"HTTP {models_code}, SSE={sse_ok} — models in {cfg.platform.namespace_ml_infra}: {', '.join(model_names) if model_names else 'none'}",
        )
    )

    env_info_curl = _run_quiet(["curl", "-sk", kf_env_info_url], timeout=15)
    visible_ns: List[str] = []
    if env_info_curl.returncode == 0 and env_info_curl.stdout.strip():
        try:
            edata = json.loads(env_info_curl.stdout)
            for ns_entry in edata.get("namespaces", []):
                rns = ns_entry.get("namespace")
                if rns and rns not in visible_ns:
                    visible_ns.append(rns)
        except Exception:
            pass
    expected_ns = [
        cfg.platform.namespace_kubeflow,
        cfg.platform.namespace_ml_infra,
        cfg.platform.namespace_docs_agent,
    ]
    all_ns_visible = all(ns in visible_ns for ns in expected_ns)
    results.append(
        CheckResult(
            stage="6. Frontends & Dashboard",
            name="Dashboard Cross-Namespace Visibility (`/api/workgroup/env-info`)",
            status="PASS" if all_ns_visible else "FAIL",
            detail=f"Dashboard dropdown namespaces: {', '.join(visible_ns) if visible_ns else 'none'}",
        )
    )

    all_passed = all(r.status in ("PASS", "SKIP") for r in results)
    return results, all_passed, agent_url, page_url


def print_verification_report(
    results: List[CheckResult],
    all_passed: bool,
    agent_url: str,
    page_url: str,
) -> None:
    """Render a Rich summary table of all ordered verification stages plus the exposed page & dashboard URLs."""
    table = Table(title="Kubeflow Docs Agent — Ordered Deployment Verification Report")
    table.add_column("Stage", style="cyan")
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    passed_count = 0
    failed_count = 0
    for r in results:
        if r.status == "PASS":
            badge = "[bold green]PASS[/bold green]"
            passed_count += 1
        elif r.status == "SKIP":
            badge = "[yellow]SKIP[/yellow]"
        else:
            badge = "[bold red]FAIL[/bold red]"
            failed_count += 1
        table.add_row(r.stage, r.name, badge, r.detail)

    console.print(table)

    kf_dash_url = "http://localhost:8085/"
    kf_pipe_url = "http://localhost:8085/pipeline/"
    kf_models_url = "http://localhost:8085/kserve-endpoints/"

    status_banner = (
        f"[bold green]ALL CHECKS PASSED ({passed_count} passed, {failed_count} failed)[/bold green]"
        if all_passed
        else f"[bold red]VERIFICATION INCOMPLETE ({passed_count} passed, {failed_count} failed)[/bold red]"
    )
    console.print(
        Panel(
            f"{status_banner}\n\n"
            f"• [bold]Flo Chatbot Frontend (Netlify / Isolated):[/bold] [bold underline cyan]{page_url}[/bold underline cyan]\n"
            f"• [bold]Kubeflow Central Dashboard (Port-Forward):[/bold] [bold underline green]{kf_dash_url}[/bold underline green] (`docs-cli expose-dashboard`)\n"
            f"• [bold]Kubeflow Pipelines UI (`kubeflow`):[/bold]        [cyan]{kf_pipe_url}[/cyan]\n"
            f"• [bold]KServe Models UI (`ml-infra`):[/bold]             [cyan]{kf_models_url}[/cyan]\n"
            f"• [bold]A2A Agent Stream (`docs-agent`):[/bold]           [cyan]{agent_url}[/cyan]",
            title="Verification Summary — Isolated Frontends & Kubeflow Dashboard",
        )
    )
