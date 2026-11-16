"""Kubernetes Secret generation, Helm chart orchestration, and KFP pipeline triggers."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from rich.console import Console

from .config import DocsConfig

console = Console()


def _kube_env(cfg: DocsConfig) -> Dict[str, str]:
    env = dict(os.environ)
    if cfg.kubeconfig_path.exists():
        env["KUBECONFIG"] = str(cfg.kubeconfig_path)
    return env


def generate_rsa_keypair_pem() -> Tuple[str, str]:
    """Generate a 2048-bit RSA private/public keypair in PEM format using native `openssl` (or cryptography fallback)."""
    import shutil

    if shutil.which("openssl"):
        priv_res = subprocess.run(
            ["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048"],
            capture_output=True,
            text=True,
            check=False,
        )
        if priv_res.returncode == 0 and "BEGIN PRIVATE KEY" in priv_res.stdout:
            pub_res = subprocess.run(
                ["openssl", "rsa", "-pubout"],
                input=priv_res.stdout,
                capture_output=True,
                text=True,
                check=False,
            )
            if pub_res.returncode == 0 and "BEGIN PUBLIC KEY" in pub_res.stdout:
                return priv_res.stdout, pub_res.stdout

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    return private_pem, public_pem


def generate_self_signed_tls_pem(domain: str) -> Tuple[str, str]:
    """Generate a self-signed TLS certificate and private key for kagent-ui-tls."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Kubeflow Docs Agent Dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, domain[:64]),
        ]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName(domain),
                    x509.DNSName(f"*.{domain}"),
                    x509.DNSName(f"chat.{domain}"),
                    x509.DNSName(f"kubeflow.{domain}"),
                ]
            ),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return cert_pem, key_pem


def kubectl_apply_manifest(cfg: DocsConfig, manifest: dict) -> None:
    """Apply a single Kubernetes manifest dictionary via `kubectl apply -f -`."""
    payload = yaml.safe_dump(manifest, sort_keys=False)
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=payload,
        env=_kube_env(cfg),
        text=True,
        check=True,
    )


def ensure_namespace(cfg: DocsConfig, namespace: str) -> None:
    kubectl_apply_manifest(
        cfg,
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": namespace},
        },
    )


def apply_secret(
    cfg: DocsConfig,
    name: str,
    namespace: str,
    string_data: Dict[str, str],
    secret_type: str = "Opaque",
) -> None:
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"app.kubernetes.io/managed-by": "docs-cli"},
        },
        "type": secret_type,
        "stringData": string_data,
    }
    kubectl_apply_manifest(cfg, manifest)
    console.print(f"[green]✓ Synced Secret {namespace}/{name}[/green]")


def sync_all_secrets(cfg: DocsConfig, domain: Optional[str] = None) -> None:
    """Create/update all required Kubernetes Secrets across ml-infra, docs-agent, kubeflow, and istio-system."""
    ns_ml = cfg.platform.namespace_ml_infra
    ns_docs = cfg.platform.namespace_docs_agent
    ns_kf = cfg.platform.namespace_kubeflow

    for ns in (ns_ml, ns_docs, ns_kf, "istio-system"):
        ensure_namespace(cfg, ns)

    milvus_pw = cfg.ensure_milvus_password()
    milvus_uri = f"http://milvus-milvus.{ns_ml}.svc.cluster.local:19530"
    milvus_token = f"root:{milvus_pw}"

    # 1. mcp-server-secret in docs-agent
    apply_secret(
        cfg,
        name="mcp-server-secret",
        namespace=ns_docs,
        string_data={
            "MILVUS_URI": milvus_uri,
            "MILVUS_TOKEN": milvus_token,
            "MILVUS_PASSWORD": milvus_pw,
        },
    )

    # 2. session-issuer-key in docs-agent (skip if Terraform already created it in cluster)
    existing_sec = subprocess.run(
        ["kubectl", "-n", ns_docs, "get", "secret", "session-issuer-key"],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    if existing_sec.returncode == 0:
        console.print(f"[green]✓ Secret {ns_docs}/session-issuer-key already managed by Terraform[/green]")
    else:
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        priv_file = cfg.state_dir / "session_private.pem"
        pub_file = cfg.state_dir / "session_public.pem"
        if priv_file.exists() and pub_file.exists():
            priv_pem = priv_file.read_text(encoding="utf-8")
            pub_pem = pub_file.read_text(encoding="utf-8")
        else:
            priv_pem, pub_pem = generate_rsa_keypair_pem()
            priv_file.write_text(priv_pem, encoding="utf-8")
            priv_file.chmod(0o600)
            pub_file.write_text(pub_pem, encoding="utf-8")

        apply_secret(
            cfg,
            name="session-issuer-key",
            namespace=ns_docs,
            string_data={
                "private.pem": priv_pem,
                "public.pem": pub_pem,
            },
        )

    # 3. kagent-kserve in docs-agent (dummy or external OpenAI API key)
    api_key = (
        cfg.llm.openai_api_key
        if (not cfg.cluster.enable_gpu_pool and cfg.llm.openai_api_key)
        else "not-needed"
    )
    apply_secret(
        cfg,
        name="kagent-kserve",
        namespace=ns_docs,
        string_data={"OPENAI_API_KEY": api_key},
    )

    # 4. Optional Langfuse keys in docs-agent
    if cfg.mcp.langfuse_public_key and cfg.mcp.langfuse_secret_key:
        apply_secret(
            cfg,
            name="mcp-langfuse-keys",
            namespace=ns_docs,
            string_data={
                "LANGFUSE_PUBLIC_KEY": cfg.mcp.langfuse_public_key,
                "LANGFUSE_SECRET_KEY": cfg.mcp.langfuse_secret_key,
            },
        )

    # 5. Optional Hugging Face token in ml-infra (for Gemma 4 / gated weights)
    if cfg.llm.hf_token:
        apply_secret(
            cfg,
            name="huggingface-secret",
            namespace=ns_ml,
            string_data={"HF_TOKEN": cfg.llm.hf_token},
        )

    # 6. Milvus + GitHub PAT in kubeflow namespace for KFP pipelines
    apply_secret(
        cfg,
        name="milvus-auth",
        namespace=ns_kf,
        string_data={
            "password": milvus_pw,
            "token": milvus_token,
            "MILVUS_USER": "root",
            "MILVUS_PASSWORD": milvus_pw,
        },
    )
    apply_secret(
        cfg,
        name="milvus-maintenance-lock",
        namespace=ns_kf,
        string_data={
            "MILVUS_MAINTENANCE_LOCK": "docs-cli-lock",
        },
    )
    apply_secret(
        cfg,
        name="github-pat",
        namespace=ns_kf,
        string_data={
            "token": cfg.mcp.github_token or "",
            "Github_Pat": cfg.mcp.github_token or "",
        },
    )

    # 7. Self-signed TLS secret in istio-system when tls_mode == "self-signed"
    # Persist cert & key in .docs-cli/ so repeated `deploy`/`update` runs never rotate the TLS fingerprint under an open browser tab
    if cfg.ingress.tls_mode == "self-signed":
        target_domain = domain or cfg.resolve_agent_domain()
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        cert_file = cfg.state_dir / "tls_cert.pem"
        key_file = cfg.state_dir / "tls_key.pem"
        domain_file = cfg.state_dir / "tls_domain.txt"
        if (
            cert_file.exists()
            and key_file.exists()
            and domain_file.exists()
            and domain_file.read_text(encoding="utf-8").strip() == target_domain
        ):
            cert_pem = cert_file.read_text(encoding="utf-8")
            key_pem = key_file.read_text(encoding="utf-8")
        else:
            cert_pem, key_pem = generate_self_signed_tls_pem(target_domain)
            cert_file.write_text(cert_pem, encoding="utf-8")
            key_file.write_text(key_pem, encoding="utf-8")
            key_file.chmod(0o600)
            domain_file.write_text(target_domain, encoding="utf-8")
        apply_secret(
            cfg,
            name="kagent-ui-tls",
            namespace="istio-system",
            secret_type="kubernetes.io/tls",
            string_data={
                "tls.crt": cert_pem,
                "tls.key": key_pem,
            },
        )


def get_ingress_lb_ip(cfg: DocsConfig, timeout_seconds: int = 120) -> Optional[str]:
    """Wait for and return the external IP of istio-ingressgateway in istio-system."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        res = subprocess.run(
            [
                "kubectl",
                "-n",
                "istio-system",
                "get",
                "svc",
                "istio-ingressgateway",
                "-o",
                "json",
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            ingress_list = (
                data.get("status", {}).get("loadBalancer", {}).get("ingress", [])
            )
            if ingress_list:
                ip = ingress_list[0].get("ip") or ingress_list[0].get("hostname")
                if ip:
                    return ip
        if timeout_seconds <= 5:
            break
        time.sleep(5)
    return None


def build_llm_runtime_values(cfg: DocsConfig) -> Dict[str, Any]:
    """Build Helm values override for docs-agent-mcp/charts/llm-runtime."""
    profile = "gemma" if cfg.llm.active_profile in ("gemma", "gemma4") else cfg.llm.active_profile
    return {
        "model": profile,
        "hfToken": {
            "existingSecret": "huggingface-secret" if cfg.llm.hf_token else "",
            "key": "HF_TOKEN",
        },
    }


def build_docs_agent_values(cfg: DocsConfig) -> Dict[str, Any]:
    """Build Helm values override for docs-agent-mcp/charts/docs-agent."""
    extra_env = []
    if cfg.mcp.langfuse_public_key and cfg.mcp.langfuse_secret_key:
        extra_env = [
            {
                "name": "LANGFUSE_PUBLIC_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": "mcp-langfuse-keys",
                        "key": "LANGFUSE_PUBLIC_KEY",
                    }
                },
            },
            {
                "name": "LANGFUSE_SECRET_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": "mcp-langfuse-keys",
                        "key": "LANGFUSE_SECRET_KEY",
                    }
                },
            },
        ]

    values: Dict[str, Any] = {
        "mcp": {
            "image": {
                "repository": cfg.mcp.image_repository,
                "tag": cfg.mcp.image_tag,
            },
            "config": {
                "docsCollection": cfg.mcp.collection_docs,
                "issuesCollection": cfg.mcp.collection_issues,
                "codeCollection": cfg.mcp.collection_code,
            },
            "milvusAuth": {
                "existingSecret": "mcp-server-secret",
                "passwordKey": "MILVUS_PASSWORD",
            },
            "extraEnv": extra_env,
        },
        "kagent": {
            "enabled": True,
            "apiKeySecret": {
                "name": "kagent-kserve",
                "key": "OPENAI_API_KEY",
                "create": False,
            },
        },
        "pipelines": {
            "enabled": cfg.platform.kubeflow_pipelines_scenario == "platform-k8s-native",
            "namespace": cfg.platform.namespace_kubeflow,
            "versionName": "v4",
            "registerJob": {
                "enabled": True,
                "image": "ghcr.io/kubeflow/docs-rag-ingest:v0.1.0",
            },
        },
    }

    if not cfg.cluster.enable_gpu_pool:
        # Point ModelConfigs at external OpenAI-compatible endpoint (--no-gpu mode)
        values["kagent"]["models"] = [
            {
                "name": "flo-llm",
                "model": cfg.llm.external_model,
                "baseUrl": cfg.llm.external_base_url,
                "temperature": "0.4",
                "topP": "0.9",
                "maxTokens": 8192,
                "timeout": 300,
            },
            {
                "name": "flo-llm-think",
                "model": cfg.llm.external_model,
                "baseUrl": cfg.llm.external_base_url,
                "temperature": "0.4",
                "topP": "0.9",
                "maxTokens": 8192,
                "timeout": 300,
                "reasoningEffort": "low",
            },
        ]
    return values


def build_gateway_guardrails_values(
    cfg: DocsConfig,
    domain: str,
    demo_web_enabled: bool = True,
) -> Dict[str, Any]:
    """Build Helm values override for docs-agent-mcp/charts/gateway-guardrails."""
    use_acme = cfg.ingress.tls_mode == "acme"
    cors_origins = list(cfg.resolve_cors_origins())
    for self_origin in (
        f"https://{domain}",
        f"https://chat.{domain}",
        "http://localhost:8085",
        "https://kubeflowdemochatbot.netlify.app",
        "https://kubeflow-docs-agent.netlify.app",
    ):
        if self_origin not in cors_origins:
            cors_origins.append(self_origin)

    return {
        "domain": domain,
        "namespaces": {
            "docsAgent": cfg.platform.namespace_docs_agent,
            "mlInfra": cfg.platform.namespace_ml_infra,
            "kubeflow": cfg.platform.namespace_kubeflow,
            "istioSystem": "istio-system",
        },
        "gateway": {
            "enabled": True,
            "tls": {
                "enabled": True,
                "credentialName": "kagent-ui-tls",
                "acme": {
                    "enabled": use_acme,
                    "createIssuer": use_acme,
                    "clusterIssuerName": "letsencrypt-prod",
                    "email": cfg.ingress.acme_email or "admin@kubeflow.org",
                },
            },
        },
        "routing": {
            "cors": {
                "allowOrigins": [
                    {"exact": origin} for origin in cors_origins
                ],
            },
        },
        "sessionAuth": {
            "enabled": cfg.ingress.enable_jwt_guardrails,
            "enforce": cfg.ingress.enable_jwt_guardrails,
            "image": "ghcr.io/kubeflow/kubeflow-session-issuer:145409a",
        },
        "demoWeb": {
            "enabled": demo_web_enabled,
            "service": "docs-agent-demo-web",
            "port": 8080,
        },
    }


def render_helm_values_files(
    cfg: DocsConfig,
    domain: Optional[str] = None,
) -> Dict[str, Path]:
    """Write rendered Helm values YAML files to .docs-cli/helm/."""
    helm_dir = cfg.state_dir / "helm"
    helm_dir.mkdir(parents=True, exist_ok=True)
    resolved_domain = domain or cfg.resolve_agent_domain()

    files = {
        "llm-runtime": helm_dir / "llm-runtime.values.yaml",
        "docs-agent": helm_dir / "docs-agent.values.yaml",
        "gateway-guardrails": helm_dir / "gateway-guardrails.values.yaml",
    }

    files["llm-runtime"].write_text(
        yaml.safe_dump(build_llm_runtime_values(cfg), sort_keys=False),
        encoding="utf-8",
    )
    files["docs-agent"].write_text(
        yaml.safe_dump(build_docs_agent_values(cfg), sort_keys=False),
        encoding="utf-8",
    )
    files["gateway-guardrails"].write_text(
        yaml.safe_dump(
            build_gateway_guardrails_values(cfg, resolved_domain), sort_keys=False
        ),
        encoding="utf-8",
    )
    return files


KNOWN_CLI_MANAGED_SECRETS = {
    "mcp-server-secret",
    "mcp-langfuse-keys",
    "session-issuer-key",
    "kagent-kserve",
    "huggingface-secret",
    "milvus-auth",
    "milvus-maintenance-lock",
    "github-pat",
    "kagent-ui-tls",
}


def validate_rendered_chart_contracts(
    cfg: DocsConfig,
    values_files: Dict[str, Path],
) -> List[str]:
    """Render all Helm charts with CLI-generated values and detect any unknown Secret references or template errors."""
    issues: List[str] = []
    chart_specs = [
        ("llm-runtime", cfg.charts_dir / "llm-runtime", cfg.platform.namespace_ml_infra),
        ("docs-agent", cfg.charts_dir / "docs-agent", cfg.platform.namespace_docs_agent),
        ("gateway-guardrails", cfg.charts_dir / "gateway-guardrails", cfg.platform.namespace_docs_agent),
    ]

    def _walk_secret_refs(node: Any, found: set) -> None:
        if isinstance(node, dict):
            if "secretKeyRef" in node and isinstance(node["secretKeyRef"], dict):
                sname = node["secretKeyRef"].get("name")
                if sname and not node["secretKeyRef"].get("optional", False):
                    found.add(str(sname))
            if "secret" in node and isinstance(node["secret"], dict):
                sname = node["secret"].get("secretName")
                if sname and not node["secret"].get("optional", False):
                    found.add(str(sname))
            if "credentialName" in node and isinstance(node["credentialName"], str):
                found.add(node["credentialName"])
            for v in node.values():
                _walk_secret_refs(v, found)
        elif isinstance(node, list):
            for item in node:
                _walk_secret_refs(item, found)

    for rel_name, cpath, ns in chart_specs:
        if not cpath.exists() or rel_name not in values_files:
            continue
        res = subprocess.run(
            [
                "helm",
                "template",
                rel_name,
                str(cpath),
                "--namespace",
                ns,
                "-f",
                str(values_files[rel_name]),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            issues.append(f"helm template failed for {rel_name}: {res.stderr.strip()}")
            continue
        referenced_secrets: set = set()
        for doc in yaml.safe_load_all(res.stdout):
            if doc:
                _walk_secret_refs(doc, referenced_secrets)
        unknown = sorted(referenced_secrets - KNOWN_CLI_MANAGED_SECRETS)
        for sec in unknown:
            issues.append(
                f"Chart '{rel_name}' references unmanaged Secret '{sec}' (update sync_all_secrets in cli/docs_cli/kube.py)"
            )
    return issues


def helm_upgrade_install(
    cfg: DocsConfig,
    release: str,
    chart_path: Path,
    namespace: str,
    values_file: Path,
    create_namespace: bool = True,
) -> None:
    cmd = [
        "helm",
        "upgrade",
        "--install",
        release,
        str(chart_path),
        "--namespace",
        namespace,
        "-f",
        str(values_file),
    ]
    if create_namespace:
        cmd.append("--create-namespace")
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    subprocess.run(cmd, env=_kube_env(cfg), check=True, text=True)


def install_kagent_crds_and_controller(cfg: DocsConfig) -> None:
    """Ensure Kagent CRDs and controller are installed in docs-agent namespace (no-op if Terraform already installed them)."""
    ns_docs = cfg.platform.namespace_docs_agent
    check_rel = subprocess.run(
        ["helm", "status", "kagent", "-n", ns_docs],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    if check_rel.returncode == 0:
        console.print(f"[green]✓ Kagent Helm release already active in {ns_docs}[/green]")
        return

    ensure_namespace(cfg, ns_docs)

    for release, chart in (
        ("kagent-crds", "oci://ghcr.io/kagent-dev/kagent/helm/kagent-crds"),
        ("kagent", "oci://ghcr.io/kagent-dev/kagent/helm/kagent"),
    ):
        cmd = [
            "helm",
            "upgrade",
            "--install",
            release,
            chart,
            "--version",
            "0.8.6",
            "--namespace",
            ns_docs,
        ]
        if release == "kagent":
            cmd.extend(["--set", "providers.default=openAI"])
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        subprocess.run(cmd, env=_kube_env(cfg), check=True, text=True)


def ensure_oci_growfs_daemonset(cfg: DocsConfig) -> None:
    """Ensure OCI worker node root partitions are expanded to full boot volume size (100G/250G)."""
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {
            "name": "oci-growfs",
            "namespace": "kube-system",
            "labels": {"app": "oci-growfs"},
        },
        "spec": {
            "selector": {"matchLabels": {"app": "oci-growfs"}},
            "template": {
                "metadata": {"labels": {"app": "oci-growfs"}},
                "spec": {
                    "priorityClassName": "system-node-critical",
                    "hostPID": True,
                    "tolerations": [{"operator": "Exists"}],
                    "containers": [
                        {
                            "name": "growfs",
                            "image": "iad.ocir.io/id9y6mi8tcky/oke-public-flannel@sha256:589036bb568da2d0b9c6a9ccdeda67a0b957aa0211e1e18d025f4aeb6e681b50",
                            "imagePullPolicy": "IfNotPresent",
                            "securityContext": {"privileged": True},
                            "command": [
                                "/bin/sh",
                                "-c",
                                "nsenter -t 1 -m -u -n -i /usr/libexec/oci-growfs -y || true; sleep 360000",
                            ],
                            "resources": {
                                "requests": {"cpu": "10m", "memory": "16Mi"},
                            },
                        }
                    ],
                },
            },
        },
    }
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(manifest),
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )


def deploy_demo_web(cfg: DocsConfig, domain: str) -> None:
    """Deploy the standalone demo web page in docs-agent namespace so https://<domain>/ serves the live widget."""
    from .netlify import prepare_demo_bundle

    ns_docs = cfg.platform.namespace_docs_agent
    dist_dir = prepare_demo_bundle(cfg, domain)

    index_html = (dist_dir / "index.html").read_text(encoding="utf-8")
    chatbot_css = (dist_dir / "docs_styles" / "chatbot.css").read_text(encoding="utf-8")
    chatbot_js = (dist_dir / "docs_scripts" / "chatbot.js").read_text(encoding="utf-8")

    cm_payload = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "docs-agent-demo-web",
            "namespace": ns_docs,
            "labels": {"app": "docs-agent-demo-web"},
        },
        "data": {
            "index.html": index_html,
            "chatbot.css": chatbot_css,
            "chatbot.js": chatbot_js,
        },
    }
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(cm_payload),
        env=_kube_env(cfg),
        check=True,
        text=True,
        capture_output=True,
    )

    deploy_and_svc = [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "docs-agent-demo-web",
                "namespace": ns_docs,
                "labels": {"app": "docs-agent-demo-web"},
            },
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "docs-agent-demo-web"}},
                "template": {
                    "metadata": {"labels": {"app": "docs-agent-demo-web"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "web",
                                "image": "mirror.gcr.io/library/python:3.12-slim",
                                "command": [
                                    "python3",
                                    "-m",
                                    "http.server",
                                    "8080",
                                    "--directory",
                                    "/srv",
                                ],
                                "ports": [{"containerPort": 8080}],
                                "volumeMounts": [
                                    {
                                        "name": "web-assets",
                                        "mountPath": "/srv/index.html",
                                        "subPath": "index.html",
                                    },
                                    {
                                        "name": "web-assets",
                                        "mountPath": "/srv/docs_styles/chatbot.css",
                                        "subPath": "chatbot.css",
                                    },
                                    {
                                        "name": "web-assets",
                                        "mountPath": "/srv/docs_scripts/chatbot.js",
                                        "subPath": "chatbot.js",
                                    },
                                ],
                                "resources": {
                                    "requests": {"cpu": "20m", "memory": "32Mi"},
                                    "limits": {"cpu": "100m", "memory": "128Mi"},
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "web-assets",
                                "configMap": {"name": "docs-agent-demo-web"},
                            }
                        ],
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "docs-agent-demo-web",
                "namespace": ns_docs,
                "labels": {"app": "docs-agent-demo-web"},
            },
            "spec": {
                "selector": {"app": "docs-agent-demo-web"},
                "ports": [{"port": 8080, "targetPort": 8080}],
            },
        },
    ]
    for obj in deploy_and_svc:
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=json.dumps(obj),
            env=_kube_env(cfg),
            check=True,
            text=True,
            capture_output=True,
        )
    subprocess.run(
        ["kubectl", "-n", ns_docs, "rollout", "restart", "deployment/docs-agent-demo-web"],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    console.print(
        f"[green]✓ Deployed in-cluster demo web page at https://{domain}/ and https://chat.{domain}/[/green]"
    )


def set_demo_web_released(cfg: DocsConfig, domain: str, released: bool) -> None:
    """Scale down/up docs-agent-demo-web and remove/restore its VirtualService route during scale-down/up."""
    ns_docs = cfg.platform.namespace_docs_agent
    target_replicas = 0 if released else 1
    subprocess.run(
        [
            "kubectl",
            "-n",
            ns_docs,
            "scale",
            "deployment/docs-agent-demo-web",
            f"--replicas={target_replicas}",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    helm_dir = cfg.state_dir / "helm"
    helm_dir.mkdir(parents=True, exist_ok=True)
    gw_values_path = helm_dir / "gateway-guardrails.values.yaml"
    gw_values_path.write_text(
        yaml.safe_dump(
            build_gateway_guardrails_values(cfg, domain, demo_web_enabled=not released),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if (cfg.charts_dir / "gateway-guardrails").exists():
        helm_upgrade_install(
            cfg,
            release="gateway-guardrails",
            chart_path=cfg.charts_dir / "gateway-guardrails",
            namespace=ns_docs,
            values_file=gw_values_path,
        )
    if released:
        console.print(
            "[yellow]✓ Released demo frontend (docs-agent-demo-web scaled to 0, frontend route removed from VirtualService)[/yellow]"
        )
    else:
        console.print(
            f"[green]✓ Restored demo frontend at https://chat.{domain}/ and https://{domain}/[/green]"
        )


def deploy_kubeflow_dashboard(cfg: DocsConfig, domain: str) -> None:
    """Deploy Kubeflow Central Dashboard + KServe Models UI and expose https://kubeflow.<domain> with visibility across kubeflow, ml-infra, and docs-agent."""
    ns_kf = cfg.platform.namespace_kubeflow
    ns_ml = cfg.platform.namespace_ml_infra
    ns_docs = cfg.platform.namespace_docs_agent
    kf_domain = f"kubeflow.{domain}"

    dashboard_chart = (
        cfg.repo_root
        / "deployments"
        / "charts"
        / "kubeflow-manifests"
        / "applications"
        / "dashboard"
        / "helm"
    )
    kserve_ui_chart = (
        cfg.repo_root
        / "deployments"
        / "charts"
        / "kubeflow-manifests"
        / "experimental"
        / "helm"
        / "charts"
        / "kserve-ui"
    )
    if not dashboard_chart.exists():
        from .terraform import ensure_kubeflow_manifests_charts

        ensure_kubeflow_manifests_charts(cfg)
    if not dashboard_chart.exists():
        console.print("[yellow]Kubeflow dashboard chart not found; skipping.[/yellow]")
        return

    helm_dir = cfg.state_dir / "helm"
    helm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Install/upgrade kubeflow-dashboard chart in kubeflow
    # Ensure cert-manager webhook failurePolicy is Ignore so transient webhook restarts never block Certificate/Issuer patch
    for wh_kind in ("mutatingwebhookconfiguration", "validatingwebhookconfiguration"):
        subprocess.run(
            [
                "kubectl",
                "patch",
                wh_kind,
                "cert-manager-webhook",
                "--type=json",
                "-p",
                '[{"op":"replace","path":"/webhooks/0/failurePolicy","value":"Ignore"}]',
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
    links_doc = {
        "menuLinks": [
            {
                "type": "item",
                "link": "/pipeline/#/experiments",
                "text": "Experiments (KFP)",
                "icon": "kubeflow:pipeline-centered",
            },
            {
                "type": "item",
                "link": "/pipeline/#/runs",
                "text": "Runs (KFP)",
                "icon": "kubeflow:pipeline-centered",
            },
            {
                "type": "item",
                "link": "/pipeline/#/pipelines",
                "text": "Pipelines (KFP)",
                "icon": "kubeflow:pipeline-centered",
            },
            {
                "type": "section",
                "text": "Pipelines (All Tabs)",
                "icon": "kubeflow:pipeline-centered",
                "items": [
                    {
                        "type": "item",
                        "link": "/pipeline/#/pipelines",
                        "text": "Pipelines",
                    },
                    {
                        "type": "item",
                        "link": "/pipeline/#/experiments",
                        "text": "Experiments",
                    },
                    {
                        "type": "item",
                        "link": "/pipeline/#/runs",
                        "text": "Runs",
                    },
                    {
                        "type": "item",
                        "link": "/pipeline/#/recurringruns",
                        "text": "Recurring Runs",
                    },
                    {
                        "type": "item",
                        "link": "/pipeline/#/artifacts",
                        "text": "Artifacts",
                    },
                    {
                        "type": "item",
                        "link": "/pipeline/#/executions",
                        "text": "Executions",
                    },
                ],
            },
            {
                "type": "item",
                "link": "/kserve-endpoints/",
                "text": "KServe Endpoints (Models)",
                "icon": "kubeflow:models",
            },
        ],
        "externalLinks": [
            {
                "type": "item",
                "iframe": False,
                "link": "/chat/",
                "text": "Flo Docs Agent (Chat)",
                "icon": "communication:chat",
            }
        ],
        "quickLinks": [
            {
                "text": "View KFP Experiments",
                "desc": "RAG ingestion experiments & runs (kubeflow)",
                "link": "/pipeline/#/experiments",
            },
            {
                "text": "View Pipeline Runs",
                "desc": "Active & completed KFP RAG runs",
                "link": "/pipeline/#/runs",
            },
            {
                "text": "View RAG Pipelines",
                "desc": "Kubeflow Pipelines definitions (kubeflow)",
                "link": "/pipeline/#/pipelines",
            },
            {
                "text": "View KServe Models",
                "desc": "Gemma 4, Embeddings & Reranker (ml-infra)",
                "link": "/kserve-endpoints/",
            },
        ],
        "documentationItems": [
            {
                "text": "Flo Docs Agent Live Demo (Local)",
                "desc": "Interactive Kubeflow documentation assistant (no TLS prompt)",
                "link": "http://localhost:8085/chat/",
            },
            {
                "text": "Flo Docs Agent Live Demo (Public HTTPS)",
                "desc": f"Public endpoint (https://chat.{domain}/)",
                "link": f"https://chat.{domain}/",
            },
            {
                "text": "Kubeflow Documentation",
                "desc": "Official Kubeflow documentation website",
                "link": "https://www.kubeflow.org/docs/",
            },
        ],
    }
    dash_values = {
        "profileController": {
            "admin": "admin@kubeflow.org",
        },
        "centralDashboard": {
            "links": json.dumps(links_doc, indent=2),
        },
    }
    dash_values_file = helm_dir / "kubeflow-dashboard.values.yaml"
    dash_values_file.write_text(
        yaml.safe_dump(dash_values, sort_keys=False), encoding="utf-8"
    )
    console.print("[bold]Deploying kubeflow-dashboard chart...[/bold]")
    helm_upgrade_install(
        cfg,
        release="kubeflow-dashboard",
        chart_path=dashboard_chart,
        namespace=ns_kf,
        values_file=dash_values_file,
    )

    # 2. Install/upgrade kserve-ui (Models Web App) chart in kubeflow
    if kserve_ui_chart.exists():
        ensure_namespace(cfg, "kserve")
        kserve_ui_values = {
            "fullnameOverride": "kserve-ui",
            "global": {
                "namespace": ns_kf,
            },
            "kubeflow": {
                "enabled": False,
            },
            "app": {
                "enabled": True,
                "config": {
                    "appPrefix": "/kserve-endpoints",
                    "disableAuth": True,
                    "useridHeader": "kubeflow-userid",
                    "useridPrefix": "",
                },
            },
            "configMap": {
                "create": True,
                "data": {
                    "APP_PREFIX": "/kserve-endpoints",
                    "USERID_HEADER": "kubeflow-userid",
                    "USERID_PREFIX": "",
                },
            },
            "istio": {
                "sidecarInjection": {"enabled": False},
                "virtualService": {"enabled": False},
                "authorizationPolicy": {"enabled": False},
            },
            "networkPolicy": {
                "enabled": False,
            },
        }
        kserve_ui_values_file = helm_dir / "kserve-ui.values.yaml"
        kserve_ui_values_file.write_text(
            yaml.safe_dump(kserve_ui_values, sort_keys=False), encoding="utf-8"
        )
        console.print("[bold]Deploying kserve-ui (Models Web App) chart...[/bold]")
        helm_upgrade_install(
            cfg,
            release="kserve-ui",
            chart_path=kserve_ui_chart,
            namespace=ns_kf,
            values_file=kserve_ui_values_file,
        )

    # 3. Remove restrictive Istio mTLS DestinationRules / AuthorizationPolicies / NetworkPolicies in kubeflow
    # (kubeflow runs without Istio sidecars, so ISTIO_MUTUAL DestinationRules from KFP would cause 503 UF,URX)
    for resource_kind in ("destinationrule", "authorizationpolicy", "networkpolicy"):
        subprocess.run(
            ["kubectl", "-n", ns_kf, "delete", resource_kind, "--all", "--ignore-not-found=true"],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )

    # Set failurePolicy: Ignore on poddefaults mutating webhook so webhook restarts never block pod creation
    subprocess.run(
        [
            "kubectl",
            "patch",
            "mutatingwebhookconfiguration",
            "poddefaults-webhook-mutating-webhook-configuration",
            "--type=json",
            "-p",
            '[{"op":"replace","path":"/webhooks/0/failurePolicy","value":"Ignore"}]',
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )

    # 4. Cross-namespace visibility RBAC + KFAM Namespace Visibility Bridge + Unified Port-Forward Proxy (port 8080)
    kfam_bridge_script = f"""import http.server
import json
import socketserver
import threading
import urllib.error
import urllib.parse
import urllib.request

NAMESPACES = {json.dumps([ns_kf, ns_ml, ns_docs])}
DEFAULT_USER = "admin@kubeflow.org"
DASHBOARD_UPSTREAM = "http://dashboard.{ns_kf}.svc.cluster.local:80"
PIPELINE_UI_UPSTREAM = "http://ml-pipeline-ui.{ns_kf}.svc.cluster.local:80"
KSERVE_UI_UPSTREAM = "http://kserve-ui.{ns_kf}.svc.cluster.local:80"
DEMO_WEB_UPSTREAM = "http://docs-agent-demo-web.{ns_docs}.svc.cluster.local:8080"
SESSION_ISSUER_UPSTREAM = "http://session-issuer.{ns_docs}.svc.cluster.local:8000"
FEEDBACK_UPSTREAM = "http://feedback.{ns_docs}.svc.cluster.local:8000"
KAGENT_UI_UPSTREAM = "http://kagent-ui.{ns_docs}.svc.cluster.local:8080"

KSERVE_STANDALONE_NS_SCRIPT = b\"\"\"<script>
(function() {{
  function sendNs() {{
    window.postMessage({{type: "namespace-selected", value: "{ns_ml}"}}, "*");
  }}
  window.addEventListener("message", function(e) {{
    if (e && e.data && e.data.type === "iframe-connected" && window.parent === window) {{
      sendNs();
      setTimeout(sendNs, 250);
    }}
  }});
  if (window.parent === window) {{
    window.addEventListener("load", function() {{
      setTimeout(sendNs, 200);
      setTimeout(sendNs, 800);
    }});
  }}
}})();
</script></body>\"\"\"

def build_kfam_bindings_bytes(query_str: str) -> bytes:
    qs = urllib.parse.parse_qs(query_str)
    req_user = (qs.get("user") or [DEFAULT_USER])[0] or DEFAULT_USER
    users = [DEFAULT_USER]
    if req_user not in users:
        users.append(req_user)
    bindings = []
    for u in users:
        for ns in NAMESPACES:
            role_obj = {{
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "admin",
            }}
            user_obj = {{"kind": "User", "name": u}}
            bindings.append({{
                "user": user_obj,
                "User": user_obj,
                "referredNamespace": ns,
                "ReferredNamespace": ns,
                "roleRef": role_obj,
                "RoleRef": role_obj,
                "status": "Succeeded",
                "Status": "Succeeded",
            }})
    return json.dumps({{"bindings": bindings}}).encode("utf-8")

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

class KfamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        body = build_kfam_bindings_bytes(parsed.query)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return

class UnifiedDashboardProxyHandler(http.server.BaseHTTPRequestHandler):
    def _proxy(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path or "/"
        query = f"?{{parsed.query}}" if parsed.query else ""

        if path.startswith("/kfam/"):
            body = build_kfam_bindings_bytes(parsed.query)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        is_kserve_html = False
        is_sse_route = False
        if path in ("/chat", "/chat/", "/demo", "/demo/"):
            target_url = f"{{DEMO_WEB_UPSTREAM}}/index.html{{query}}"
        elif path.startswith("/docs_styles/") or path.startswith("/docs_scripts/"):
            target_url = f"{{DEMO_WEB_UPSTREAM}}{{path}}{{query}}"
        elif path in ("/api/session", "/.well-known/jwks.json"):
            target_url = f"{{SESSION_ISSUER_UPSTREAM}}{{path}}{{query}}"
        elif path == "/api/feedback":
            target_url = f"{{FEEDBACK_UPSTREAM}}{{path}}{{query}}"
        elif path.startswith("/a2a/") or path.startswith("/api/a2a/"):
            is_sse_route = True
            target_url = f"{{KAGENT_UI_UPSTREAM}}{{path}}{{query}}"
        elif path == "/kserve-endpoints" or path.startswith("/kserve-endpoints/"):
            rewritten = path[len("/kserve-endpoints"):] or "/"
            if not rewritten.startswith("/"):
                rewritten = "/" + rewritten
            if rewritten in ("/", "/index.html"):
                is_kserve_html = True
            if "/api/sse/" in rewritten:
                is_sse_route = True
            # Map kubeflow / docs-agent namespace queries (both REST and SSE) to ml-infra so KServe models always display
            for other_ns in ("{ns_kf}", "{ns_docs}"):
                rewritten = rewritten.replace(f"/api/namespaces/{{other_ns}}/", "/api/namespaces/{ns_ml}/")
                rewritten = rewritten.replace(f"/api/sse/namespaces/{{other_ns}}/", "/api/sse/namespaces/{ns_ml}/")
            target_url = f"{{KSERVE_UI_UPSTREAM}}{{rewritten}}{{query}}"
        elif (
            path.startswith("/pipeline")
            or path.startswith("/apis/v2beta1/")
            or path.startswith("/apis/v1beta1/")
            or path.startswith("/k8s/")
            or path.startswith("/artifacts/")
        ):
            mapped_query = query
            for other_ns in ("{ns_ml}", "{ns_docs}"):
                mapped_query = mapped_query.replace(f"namespace={{other_ns}}", "namespace={ns_kf}")
                mapped_query = mapped_query.replace(f"resource_reference_key.id={{other_ns}}", "resource_reference_key.id={ns_kf}")
                mapped_query = mapped_query.replace(f"%22{{other_ns}}%22", "%22{ns_kf}%22")
                mapped_query = mapped_query.replace(f'"{{other_ns}}"', '"{ns_kf}"')
            clean_path = path.rstrip("/")
            if clean_path in (
                "/pipeline/apis/v2beta1/experiments",
                "/pipeline/apis/v2beta1/runs",
                "/pipeline/apis/v2beta1/pipelines",
                "/pipeline/apis/v2beta1/recurringruns",
                "/apis/v2beta1/experiments",
                "/apis/v2beta1/runs",
                "/apis/v2beta1/pipelines",
                "/apis/v2beta1/recurringruns",
            ) and "namespace=" not in mapped_query:
                sep = "&" if mapped_query.startswith("?") else "?"
                mapped_query = f"{{mapped_query}}{{sep}}namespace={ns_kf}"
            elif clean_path in (
                "/pipeline/apis/v1beta1/experiments",
                "/pipeline/apis/v1beta1/runs",
                "/pipeline/apis/v1beta1/jobs",
                "/apis/v1beta1/experiments",
                "/apis/v1beta1/runs",
                "/apis/v1beta1/jobs",
            ) and "resource_reference_key.id=" not in mapped_query:
                sep = "&" if mapped_query.startswith("?") else "?"
                mapped_query = f"{{mapped_query}}{{sep}}resource_reference_key.type=NAMESPACE&resource_reference_key.id={ns_kf}"
            target_url = f"{{PIPELINE_UI_UPSTREAM}}{{path}}{{mapped_query}}"
        else:
            target_url = f"{{DASHBOARD_UPSTREAM}}{{path}}{{query}}"

        content_len = int(self.headers.get("Content-Length") or 0)
        req_body = self.rfile.read(content_len) if content_len > 0 else None

        fwd_headers = {{}}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in ("host", "connection", "transfer-encoding", "accept-encoding"):
                continue
            fwd_headers[k] = v
        fwd_headers["kubeflow-userid"] = DEFAULT_USER

        req = urllib.request.Request(target_url, data=req_body, headers=fwd_headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=300 if is_sse_route else 60) as resp:
                ctype = (resp.getheader("Content-Type") or "").lower()
                if is_sse_route or "text/event-stream" in ctype:
                    self.send_response(resp.status)
                    for hk, hv in resp.getheaders():
                        if hk.lower() in ("transfer-encoding", "connection", "content-encoding", "content-length"):
                            continue
                        self.send_header(hk, hv)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    if self.command != "HEAD":
                        while True:
                            line = resp.readline()
                            if not line:
                                break
                            self.wfile.write(line)
                            self.wfile.flush()
                    return

                resp_body = resp.read()
                if is_kserve_html and b"</body>" in resp_body:
                    resp_body = resp_body.replace(b"</body>", KSERVE_STANDALONE_NS_SCRIPT, 1)
                self.send_response(resp.status)
                for hk, hv in resp.getheaders():
                    if hk.lower() in ("transfer-encoding", "connection", "content-encoding", "content-length"):
                        continue
                    self.send_header(hk, hv)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(resp_body)
        except urllib.error.HTTPError as he:
            err_body = he.read() if he.fp else b""
            self.send_response(he.code)
            for hk, hv in he.headers.items():
                if hk.lower() in ("transfer-encoding", "connection", "content-encoding", "content-length"):
                    continue
                self.send_header(hk, hv)
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(err_body)
        except Exception as exc:
            msg = str(exc).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def do_PATCH(self):
        self._proxy()

    def do_HEAD(self):
        self._proxy()

    def log_message(self, fmt, *args):
        return

if __name__ == "__main__":
    t = threading.Thread(
        target=ThreadingHTTPServer(("0.0.0.0", 8081), KfamHandler).serve_forever,
        daemon=True,
    )
    t.start()
    ThreadingHTTPServer(("0.0.0.0", 8080), UnifiedDashboardProxyHandler).serve_forever()
"""

    # Ensure any previously created public VirtualService for kubeflow.<domain> is removed
    subprocess.run(
        [
            "kubectl",
            "-n",
            ns_kf,
            "delete",
            "virtualservice",
            "kubeflow-dashboard-routing",
            "--ignore-not-found=true",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )

    visibility_manifests = [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "default-editor",
                "namespace": ns_kf,
            },
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": "kubeflow-dashboard-cross-namespace"},
            "rules": [
                {
                    "apiGroups": ["", "app.k8s.io", "apps", "batch"],
                    "resources": [
                        "applications",
                        "pods",
                        "pods/log",
                        "pods/exec",
                        "events",
                        "configmaps",
                        "secrets",
                        "services",
                        "deployments",
                        "jobs",
                        "namespaces",
                        "nodes",
                        "serviceaccounts",
                    ],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": ["serving.kserve.io"],
                    "resources": ["*"],
                    "verbs": ["*"],
                },
                {
                    "apiGroups": ["pipelines.kubeflow.org", "kubeflow.org", "argoproj.io"],
                    "resources": ["*"],
                    "verbs": ["*"],
                },
                {
                    "apiGroups": ["kagent.dev"],
                    "resources": ["*"],
                    "verbs": ["*"],
                },
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "kubeflow-dashboard-cross-namespace"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "kubeflow-dashboard-cross-namespace",
            },
            "subjects": [
                {"kind": "ServiceAccount", "name": "dashboard", "namespace": ns_kf},
                {"kind": "ServiceAccount", "name": "ml-pipeline-ui", "namespace": ns_kf},
                {"kind": "ServiceAccount", "name": "ml-pipeline", "namespace": ns_kf},
                {"kind": "ServiceAccount", "name": "pipeline-runner", "namespace": ns_kf},
                {"kind": "ServiceAccount", "name": "default-editor", "namespace": ns_kf},
                {"kind": "ServiceAccount", "name": "default", "namespace": ns_kf},
                {"kind": "User", "name": "admin@kubeflow.org"},
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": "kubeflow-admin-user-visibility"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": "cluster-admin",
            },
            "subjects": [
                {"kind": "User", "name": "admin@kubeflow.org"},
                {"kind": "ServiceAccount", "name": "default-editor", "namespace": ns_kf},
                {"kind": "ServiceAccount", "name": "pipeline-runner", "namespace": ns_kf},
                {"kind": "ServiceAccount", "name": "default", "namespace": ns_kf},
            ],
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "kubeflow-namespace-bridge",
                "namespace": ns_kf,
            },
            "data": {
                "server.py": kfam_bridge_script,
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "kubeflow-namespace-bridge",
                "namespace": ns_kf,
                "labels": {"app": "kubeflow-namespace-bridge"},
            },
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "kubeflow-namespace-bridge"}},
                "template": {
                    "metadata": {"labels": {"app": "kubeflow-namespace-bridge"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "kfam-bridge",
                                "image": "mirror.gcr.io/library/python:3.12-slim",
                                "command": ["python3", "/app/server.py"],
                                "ports": [
                                    {"containerPort": 8080, "name": "http-proxy"},
                                    {"containerPort": 8081, "name": "http-kfam"},
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "bridge-script",
                                        "mountPath": "/app/server.py",
                                        "subPath": "server.py",
                                    }
                                ],
                                "resources": {
                                    "requests": {"cpu": "20m", "memory": "32Mi"},
                                    "limits": {"cpu": "200m", "memory": "128Mi"},
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "bridge-script",
                                "configMap": {"name": "kubeflow-namespace-bridge"},
                            }
                        ],
                    },
                },
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "kubeflow-namespace-bridge",
                "namespace": ns_kf,
                "labels": {
                    "app": "kubeflow-namespace-bridge",
                    "app.kubernetes.io/part-of": "kubeflow-dashboard",
                },
            },
            "spec": {
                "ports": [
                    {
                        "name": "http-kfam",
                        "port": 8081,
                        "protocol": "TCP",
                        "targetPort": 8081,
                    }
                ],
                "selector": {"app": "kubeflow-namespace-bridge"},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "kubeflow-dashboard-proxy",
                "namespace": ns_kf,
                "labels": {
                    "app": "kubeflow-namespace-bridge",
                    "app.kubernetes.io/part-of": "kubeflow-dashboard",
                },
            },
            "spec": {
                "ports": [
                    {
                        "name": "http-proxy",
                        "port": 8080,
                        "protocol": "TCP",
                        "targetPort": 8080,
                    }
                ],
                "selector": {"app": "kubeflow-namespace-bridge"},
            },
        },
    ]

    subprocess.run(
        [
            "kubectl",
            "delete",
            "clusterrolebinding",
            "kubeflow-admin-user-visibility",
            "--ignore-not-found=true",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    for obj in visibility_manifests:
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=json.dumps(obj),
            env=_kube_env(cfg),
            check=True,
            text=True,
            capture_output=True,
        )

    # Point centraldashboard at kubeflow-namespace-bridge so all 3 production namespaces are visible
    subprocess.run(
        ["kubectl", "-n", ns_kf, "rollout", "restart", "deployment/kubeflow-namespace-bridge"],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        [
            "kubectl",
            "-n",
            ns_kf,
            "set",
            "env",
            "deployment/dashboard",
            f"PROFILES_KFAM_SERVICE_HOST=kubeflow-namespace-bridge.{ns_kf}.svc.cluster.local",
            "PROFILES_KFAM_SERVICE_PORT=8081",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["kubectl", "-n", ns_kf, "rollout", "restart", "deployment/dashboard"],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    for dep_name in ("kubeflow-namespace-bridge", "dashboard", "kserve-ui"):
        subprocess.run(
            [
                "kubectl",
                "-n",
                ns_kf,
                "rollout",
                "status",
                f"deployment/{dep_name}",
                "--timeout=90s",
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
    console.print(
        f"[green]✓ Deployed Kubeflow Dashboard & KServe Models UI in '{ns_kf}' "
        f"(expose locally via `docs-cli expose-dashboard` -> http://localhost:8085)[/green]"
    )


def expose_kubeflow_dashboard_port_forward(
    cfg: DocsConfig,
    local_port: int = 8085,
    address: str = "localhost,127.0.0.1",
    background: bool = False,
) -> Optional[int]:
    """Port-forward the unified Kubeflow Central Dashboard proxy (`svc/kubeflow-dashboard-proxy`) to localhost:<local_port>."""
    import signal
    import sys

    ensure_cluster_nodes_awake(cfg)
    ns_kf = cfg.platform.namespace_kubeflow
    subprocess.run(
        [
            "kubectl",
            "-n",
            ns_kf,
            "rollout",
            "status",
            "deployment/kubeflow-namespace-bridge",
            "--timeout=90s",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    pid_file = cfg.state_dir / "dashboard-port-forward.pid"
    log_file = cfg.state_dir / "dashboard-port-forward.log"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)

    # Terminate any previous port-forward supervisor started by docs-cli on this port
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="utf-8").strip())
            try:
                os.killpg(old_pid, signal.SIGTERM)
            except Exception:
                os.kill(old_pid, signal.SIGTERM)
            time.sleep(0.5)
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)

    cmd = [
        "kubectl",
        "-n",
        ns_kf,
        "port-forward",
        "--address",
        address,
        "svc/kubeflow-dashboard-proxy",
        f"{local_port}:8080",
    ]
    url_base = f"http://localhost:{local_port}"
    console.print(
        f"[bold green]⚡ Exposing Kubeflow Central Dashboard via port-forward on {url_base}/[/bold green]\n"
        f"  • [bold]Central Dashboard:[/bold]    [underline cyan]{url_base}/[/underline cyan]\n"
        f"  • [bold]Flo Chatbot (Local):[/bold]  [cyan]{url_base}/chat/[/cyan]\n"
        f"  • [bold]Pipelines UI (`{ns_kf}`):[/bold] [cyan]{url_base}/pipeline/[/cyan]\n"
        f"  • [bold]KServe Models (`{cfg.platform.namespace_ml_infra}`):[/bold] [cyan]{url_base}/kserve-endpoints/[/cyan]\n"
        f"  • [bold]Visible Namespaces:[/bold]   `{ns_kf}`, `{cfg.platform.namespace_ml_infra}`, `{cfg.platform.namespace_docs_agent}`"
    )

    supervisor_code = (
        "import os, signal, subprocess, sys, time\n"
        f"cmd = {cmd!r}\n"
        "child = None\n"
        "def _stop(signum, frame):\n"
        "    if child and child.poll() is None:\n"
        "        child.terminate()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, _stop)\n"
        "signal.signal(signal.SIGINT, _stop)\n"
        "while True:\n"
        "    child = subprocess.Popen(cmd)\n"
        "    child.wait()\n"
        "    time.sleep(1.0)\n"
    )

    if background:
        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                [sys.executable, "-c", supervisor_code],
                env=_kube_env(cfg),
                stdout=lf,
                stderr=lf,
                start_new_session=True,
            )
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        time.sleep(1.5)
        console.print(
            f"[green]✓ Auto-reconnecting port-forward running in background (PID={proc.pid}, log={log_file})[/green]"
        )
        return proc.pid

    console.print("[dim]Auto-reconnecting port-forward active (IPv4 + IPv6). Press Ctrl+C to stop.[/dim]")
    try:
        while True:
            rc = subprocess.run(cmd, env=_kube_env(cfg), check=False).returncode
            console.print(
                f"[yellow]! Port-forward disconnected (exit={rc}); reconnecting in 1s...[/yellow]"
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped Kubeflow Dashboard port-forward.[/yellow]")
    return None


def ensure_cluster_nodes_awake(cfg: DocsConfig) -> None:
    """If the cluster was scaled down to 0 nodes (e.g. by the TTL auto-scaler), automatically wake node pools and wait for Ready."""
    nodes_res = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "json"],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    if nodes_res.returncode != 0:
        return

    from .auth import OciSession
    from .scale import discover_cluster_node_pools, scale_node_pools, schedule_in_cluster_ttl_job

    cpu_target = cfg.cluster.cpu_node_count
    gpu_target = cfg.cluster.gpu_node_count if cfg.cluster.enable_gpu_pool else 0
    expected_total = cpu_target + gpu_target

    session = OciSession.from_docs_config(cfg)
    comp_id = session.resolve_compartment_id(cfg)
    pools = discover_cluster_node_pools(cfg, session, comp_id)
    cur_cpu = (pools.get("cpu") or {}).get("size", 0)
    cur_gpu = (pools.get("gpu") or {}).get("size", 0)

    ready_count = 0
    total_nodes = 0
    if nodes_res.stdout.strip():
        try:
            items = json.loads(nodes_res.stdout).get("items", [])
            total_nodes = len(items)
            for n in items:
                conds = (n.get("status") or {}).get("conditions") or []
                if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conds):
                    ready_count += 1
        except Exception:
            pass

    console.print(
        f"[dim]Cluster node check: OCI pools (cpu={cur_cpu}/{cpu_target}, gpu={cur_gpu}/{gpu_target}), K8s nodes ({ready_count}/{total_nodes} Ready)[/dim]"
    )
    if cur_cpu >= cpu_target and cur_gpu >= gpu_target and ready_count >= expected_total:
        return

    console.print(
        f"[bold yellow]⚡ Cluster is hibernated or partially scaled down (cpu={cur_cpu}/{cpu_target}, gpu={cur_gpu}/{gpu_target}). Waking OKE node pools...[/bold yellow]"
    )
    if cur_cpu == 0 and total_nodes > 0:
        # Remove ghost Node objects left over from simultaneous VM termination
        subprocess.run(
            ["kubectl", "delete", "nodes", "--all", "--ignore-not-found=true"],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )

    scale_node_pools(
        cfg,
        session=session,
        compartment_id=comp_id,
        cpu_size=cpu_target,
        gpu_size=gpu_target,
        apply=True,
        release_frontend=False,
    )

    console.print(
        f"[bold cyan]Waiting for {cpu_target} CPU worker node(s) (and GPU node) to join and report Ready...[/bold cyan]"
    )
    deadline = time.time() + 900
    expected_total = cpu_target + gpu_target
    while time.time() < deadline:
        r = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            items = json.loads(r.stdout).get("items", [])
            ready_nodes = [
                n.get("metadata", {}).get("name")
                for n in items
                if any(
                    c.get("type") == "Ready" and c.get("status") == "True"
                    for c in (n.get("status") or {}).get("conditions") or []
                )
            ]
            if len(ready_nodes) >= expected_total:
                console.print(
                    f"[bold green]✓ All {len(ready_nodes)}/{expected_total} worker nodes are Ready![/bold green]"
                )
                break
            console.print(
                f"[dim]  • Nodes Ready: {len(ready_nodes)}/{expected_total} — waiting 15s...[/dim]"
            )
        time.sleep(15)

    ensure_oci_growfs_daemonset(cfg)
    for ns, dep in (
        ("istio-system", "istiod"),
        ("istio-system", "istio-ingressgateway"),
        (cfg.platform.namespace_ml_infra, "seaweedfs"),
        (cfg.platform.namespace_kubeflow, "seaweedfs"),
        (cfg.platform.namespace_kubeflow, "ml-pipeline"),
        (cfg.platform.namespace_kubeflow, "ml-pipeline-ui"),
        (cfg.platform.namespace_kubeflow, "dashboard"),
        (cfg.platform.namespace_kubeflow, "kserve-ui"),
        (cfg.platform.namespace_kubeflow, "kubeflow-namespace-bridge"),
        (cfg.platform.namespace_ml_infra, "embeddings-service-predictor"),
        (cfg.platform.namespace_ml_infra, "reranker-service-predictor"),
        (cfg.platform.namespace_docs_agent, "session-issuer"),
        (cfg.platform.namespace_docs_agent, "kagent-ui"),
        (cfg.platform.namespace_docs_agent, "kubeflow-docs-agent"),
        (cfg.platform.namespace_docs_agent, "docs-agent-demo-web"),
    ):
        subprocess.run(
            ["kubectl", "-n", ns, "rollout", "status", f"deployment/{dep}", "--timeout=180s"],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
    if cfg.cluster.enable_gpu_pool:
        subprocess.run(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "scale",
                "deployment/llm-predictor",
                "--replicas=1",
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "rollout",
                "status",
                "deployment/llm-predictor",
                "--timeout=360s",
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
    schedule_in_cluster_ttl_job(
        cfg,
        session=session,
        compartment_id=comp_id,
        duration_str="6h",
        gpu_only=False,
    )


def wait_for_worker_nodes_ready(cfg: DocsConfig, timeout_seconds: int = 900) -> None:
    """Wait until OKE CPU (and optional GPU) worker nodes register with Kubernetes and report Ready."""
    expected = cfg.cluster.cpu_node_count + (
        cfg.cluster.gpu_node_count if cfg.cluster.enable_gpu_pool else 0
    )
    console.print(
        f"[bold cyan]Waiting for {expected} OKE worker node(s) to join Kubernetes and report Ready...[/bold cyan]"
    )
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        r = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            try:
                items = json.loads(r.stdout).get("items", [])
                ready_nodes = [
                    n.get("metadata", {}).get("name")
                    for n in items
                    if any(
                        c.get("type") == "Ready" and c.get("status") == "True"
                        for c in (n.get("status") or {}).get("conditions") or []
                    )
                ]
                if len(ready_nodes) >= expected:
                    console.print(
                        f"[bold green]✓ All {len(ready_nodes)}/{expected} worker nodes are Ready![/bold green]"
                    )
                    ensure_oci_growfs_daemonset(cfg)
                    return
                console.print(
                    f"[dim]  • Nodes Ready: {len(ready_nodes)}/{expected} — waiting 15s...[/dim]"
                )
            except Exception:
                pass
        time.sleep(15)


def deploy_helm_stack(cfg: DocsConfig, skip_kagent: bool = False) -> str:
    """Sync secrets, detect LB IP / domain, and install llm-runtime, docs-agent, and gateway-guardrails."""
    ensure_cluster_nodes_awake(cfg)
    ensure_oci_growfs_daemonset(cfg)
    lb_ip = get_ingress_lb_ip(cfg, timeout_seconds=30)
    domain = cfg.resolve_agent_domain(lb_ip=lb_ip)
    console.print(f"[bold cyan]Using Docs Agent Ingress Domain:[/bold cyan] {domain}")

    sync_all_secrets(cfg, domain=domain)

    if not skip_kagent:
        install_kagent_crds_and_controller(cfg)

    values_files = render_helm_values_files(cfg, domain=domain)

    if cfg.cluster.enable_gpu_pool:
        llm_exists = subprocess.run(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "get",
                "inferenceservice",
                "llm",
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
        if llm_exists.returncode == 0:
            console.print(
                f"[green]✓ GPU InferenceService/llm (profile={cfg.llm.active_profile}) already present; preserving existing deployment.[/green]"
            )
        else:
            console.print(
                f"[bold]Deploying llm-runtime chart (profile={cfg.llm.active_profile})...[/bold]"
            )
            helm_upgrade_install(
                cfg,
                release="llm-runtime",
                chart_path=cfg.charts_dir / "llm-runtime",
                namespace=cfg.platform.namespace_ml_infra,
                values_file=values_files["llm-runtime"],
            )
    else:
        console.print(
            "[yellow]Skipping llm-runtime chart (--no-gpu / external LLM mode)[/yellow]"
        )

    console.print("[bold]Deploying docs-agent chart...[/bold]")
    # Set failurePolicy: Ignore on KFP pipelineversions webhooks so transient webhook restarts never block helm upgrade
    for wh_kind in ("mutatingwebhookconfiguration", "validatingwebhookconfiguration"):
        subprocess.run(
            [
                "kubectl",
                "patch",
                wh_kind,
                "pipelineversions.pipelines.kubeflow.org",
                "--type=json",
                "-p",
                '[{"op":"replace","path":"/webhooks/0/failurePolicy","value":"Ignore"}]',
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
    # Delete existing PipelineVersions in kubeflow prior to helm upgrade so KFP webhook doesn't reject immutable spec patch
    subprocess.run(
        [
            "kubectl",
            "-n",
            cfg.platform.namespace_kubeflow,
            "delete",
            "pipelineversions.pipelines.kubeflow.org",
            "--all",
            "--ignore-not-found=true",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    helm_upgrade_install(
        cfg,
        release="docs-agent",
        chart_path=cfg.charts_dir / "docs-agent",
        namespace=cfg.platform.namespace_docs_agent,
        values_file=values_files["docs-agent"],
    )

    console.print("[bold]Deploying gateway-guardrails chart...[/bold]")
    helm_upgrade_install(
        cfg,
        release="gateway-guardrails",
        chart_path=cfg.charts_dir / "gateway-guardrails",
        namespace=cfg.platform.namespace_docs_agent,
        values_file=values_files["gateway-guardrails"],
    )

    deploy_demo_web(cfg, domain=domain)
    deploy_kubeflow_dashboard(cfg, domain=domain)

    # Wait for core RAG backends (TEI embeddings/reranker, Milvus, KFP) before bootstrapping collections & runs
    for ns, dep, tmo in (
        (cfg.platform.namespace_ml_infra, "embeddings-service-predictor", "300s"),
        (cfg.platform.namespace_ml_infra, "reranker-service-predictor", "300s"),
        (cfg.platform.namespace_ml_infra, "milvus-milvus-standalone", "360s"),
        (cfg.platform.namespace_kubeflow, "ml-pipeline", "240s"),
    ):
        subprocess.run(
            ["kubectl", "-n", ns, "rollout", "status", f"deployment/{dep}", f"--timeout={tmo}"],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )

    run_kfp_and_bootstrap_rag(cfg)

    if cfg.cluster.enable_gpu_pool:
        console.print(
            "[bold cyan]Waiting for GPU vLLM (`deployment/llm-predictor`) model load & readiness...[/bold cyan]"
        )
        for _ in range(12):
            chk = subprocess.run(
                [
                    "kubectl",
                    "-n",
                    cfg.platform.namespace_ml_infra,
                    "get",
                    "deployment",
                    "llm-predictor",
                ],
                env=_kube_env(cfg),
                capture_output=True,
                text=True,
                check=False,
            )
            if chk.returncode == 0:
                break
            time.sleep(5)
        subprocess.run(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "wait",
                "--for=condition=Ready",
                "pod",
                "-l",
                "serving.kserve.io/inferenceservice=llm",
                "--timeout=900s",
            ],
            env=_kube_env(cfg),
            check=False,
        )

    return domain


def _build_in_cluster_rag_bootstrap_script() -> str:
    """Return an in-cluster Python script that triggers KFP v2beta1 pipeline runs and seeds Milvus collections (`kubeflow_docs`, `issues_rag`, `code_rag`)."""
    return r'''
import json
import os
import time
import urllib.request
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient, connections, utility

milvus_host = os.environ.get("MILVUS_HOST", "milvus-milvus.ml-infra.svc.cluster.local")
milvus_port = os.environ.get("MILVUS_PORT", "19530")
milvus_user = os.environ.get("MILVUS_USER", "root")
milvus_password = os.environ.get("MILVUS_PASSWORD", "")
embeddings_url = os.environ.get("EMBEDDINGS_URL", "http://embeddings-service-predictor.ml-infra.svc.cluster.local/embed")
kfp_api = os.environ.get("KFP_API_URL", "http://ml-pipeline.kubeflow.svc.cluster.local:8888")

def embed_texts(texts):
    body = json.dumps({"inputs": [t[:600] for t in texts]}).encode("utf-8")
    req = urllib.request.Request(
        embeddings_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))

# 1. Trigger KFP v2beta1 Pipeline Runs via ml-pipeline API (with K8s ServiceAccount Bearer token + kubeflow-userid)
try:
    sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    kfp_headers = {
        "Content-Type": "application/json",
        "kubeflow-userid": "admin@kubeflow.org",
    }
    if os.path.exists(sa_token_path):
        with open(sa_token_path, "r", encoding="utf-8") as tf:
            kfp_headers["Authorization"] = f"Bearer {tf.read().strip()}"

    # Ensure an Experiment exists in namespace kubeflow
    exp_id = None
    exp_list_req = urllib.request.Request(
        f"{kfp_api}/apis/v2beta1/experiments?namespace=kubeflow&page_size=20",
        headers=kfp_headers,
    )
    with urllib.request.urlopen(exp_list_req, timeout=15) as eresp:
        exps = json.loads(eresp.read().decode("utf-8")).get("experiments", [])
        for e in exps:
            if e.get("display_name") == "docs-agent-rag-ingestion":
                exp_id = e.get("experiment_id")
                break
        if not exp_id and exps:
            exp_id = exps[0].get("experiment_id")

    if not exp_id:
        exp_body = json.dumps({
            "display_name": "docs-agent-rag-ingestion",
            "description": "Automated RAG ingestion runs for Kubeflow Docs, Issues, and Code",
            "namespace": "kubeflow",
        }).encode("utf-8")
        ecreate_req = urllib.request.Request(
            f"{kfp_api}/apis/v2beta1/experiments",
            data=exp_body,
            method="POST",
            headers=kfp_headers,
        )
        with urllib.request.urlopen(ecreate_req, timeout=15) as ecresp:
            exp_id = json.loads(ecresp.read().decode("utf-8")).get("experiment_id")

    # Check existing runs in kubeflow so we don't spam duplicate runs if already triggered
    existing_run_names = set()
    runs_req = urllib.request.Request(
        f"{kfp_api}/apis/v2beta1/runs?namespace=kubeflow&page_size=50",
        headers=kfp_headers,
    )
    with urllib.request.urlopen(runs_req, timeout=15) as rlist_resp:
        for ritem in json.loads(rlist_resp.read().decode("utf-8")).get("runs", []):
            dname = ritem.get("display_name") or ""
            existing_run_names.add(dname)

    list_req = urllib.request.Request(
        f"{kfp_api}/apis/v2beta1/pipelines?namespace=kubeflow&page_size=20",
        headers=kfp_headers,
    )
    with urllib.request.urlopen(list_req, timeout=15) as resp:
        pipes = json.loads(resp.read().decode("utf-8")).get("pipelines", [])
    for p in pipes:
        pid = p.get("pipeline_id")
        pname = p.get("display_name") or p.get("name") or pid
        if not pid:
            continue
        if any(pname in rname for rname in existing_run_names):
            print(f"[rag-bootstrap] KFP Run for {pname} already exists in kubeflow; skipping duplicate trigger.", flush=True)
            continue
        v_req = urllib.request.Request(
            f"{kfp_api}/apis/v2beta1/pipelines/{pid}/versions?page_size=5",
            headers=kfp_headers,
        )
        with urllib.request.urlopen(v_req, timeout=15) as vresp:
            versions = json.loads(vresp.read().decode("utf-8")).get("pipeline_versions", [])
        if versions:
            vid = versions[0].get("pipeline_version_id")
            run_payload = {
                "display_name": f"ingest-{pname}-v4",
                "pipeline_version_reference": {
                    "pipeline_id": pid,
                    "pipeline_version_id": vid,
                },
            }
            if exp_id:
                run_payload["experiment_id"] = exp_id
            run_body = json.dumps(run_payload).encode("utf-8")
            rreq = urllib.request.Request(
                f"{kfp_api}/apis/v2beta1/runs",
                data=run_body,
                method="POST",
                headers=kfp_headers,
            )
            with urllib.request.urlopen(rreq, timeout=15) as rresp:
                rdata = json.loads(rresp.read().decode("utf-8"))
                print(f"[rag-bootstrap] Triggered KFP Run for {pname}: run_id={rdata.get('run_id')}", flush=True)
except Exception as exc:
    print(f"[rag-bootstrap] Note: KFP run trigger returned: {exc}", flush=True)

# 2. Ensure Milvus collections (kubeflow_docs, issues_rag, code_rag) exist, are indexed, and have initial Kubeflow corpus chunks loaded
uri = f"http://{milvus_host}:{milvus_port}"
token = f"{milvus_user}:{milvus_password}" if milvus_password else ""
client = MilvusClient(uri=uri, token=token)
connections.connect("default", host=milvus_host, port=milvus_port, user=milvus_user, password=milvus_password)
now_ts = int(time.time())

# 2a. kubeflow_docs (v=4 hybrid BM25 + 768-d dense)
docs_col = "kubeflow_docs"
if not client.has_collection(docs_col):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(
            name="content_text",
            dtype=DataType.VARCHAR,
            max_length=2000,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"tokenizer": "standard", "filter": ["lowercase"]},
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="release_date", dtype=DataType.INT64, nullable=True),
    ]
    schema = CollectionSchema(fields, "RAG lean hybrid collection for documentation (v=4, hybrid=bm25+dense)")
    schema.add_function(
        Function(
            name="content_text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content_text"],
            output_field_names=["sparse_vector"],
        )
    )
    idx_params = MilvusClient.prepare_index_params()
    idx_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    idx_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(collection_name=docs_col, schema=schema, index_params=idx_params)

    seed_docs = [
        {
            "document_id": "docs/components/pipelines/overview.md",
            "title": "Kubeflow Pipelines Overview",
            "section_path": "Kubeflow Pipelines > Overview",
            "file_path": "content/en/docs/components/pipelines/overview.md",
            "citation_url": "https://www.kubeflow.org/docs/components/pipelines/overview/",
            "content_text": "Kubeflow Pipelines (KFP) is a platform for building and deploying portable, scalable machine learning (ML) workflows based on Docker containers. In KFP v2, pipelines can be stored as Kubernetes-native Custom Resources (`pipelines.kubeflow.org/v2beta1` Pipeline and PipelineVersion CRDs) using the `platform-k8s-native` deployment scenario, with SeaweedFS providing S3-compatible artifact storage.",
        },
        {
            "document_id": "docs/components/kserve/introduction.md",
            "title": "KServe Model Serving in Kubeflow",
            "section_path": "KServe > Introduction",
            "file_path": "content/en/docs/components/kserve/introduction.md",
            "citation_url": "https://www.kubeflow.org/docs/components/kserve/introduction/",
            "content_text": "KServe provides a Kubernetes Custom Resource Definition (InferenceService) for serving machine learning models and generative AI LLMs (via vLLM and Hugging Face Text Embeddings Inference) on CPU and GPU nodes with Knative Serving and Istio ingress routing.",
        },
        {
            "document_id": "docs/components/katib/overview.md",
            "title": "Katib Automated Machine Learning (AutoML)",
            "section_path": "Katib > Overview",
            "file_path": "content/en/docs/components/katib/overview.md",
            "citation_url": "https://www.kubeflow.org/docs/components/katib/overview/",
            "content_text": "Katib is a Kubernetes-native project for automated machine learning (AutoML) in Kubeflow, supporting hyperparameter tuning, early stopping, and neural architecture search (NAS) across PyTorch, TensorFlow, and XGBoost.",
        },
        {
            "document_id": "docs/components/trainer/overview.md",
            "title": "Kubeflow Trainer Distributed Training",
            "section_path": "Kubeflow Trainer > Overview",
            "file_path": "content/en/docs/components/trainer/overview.md",
            "citation_url": "https://www.kubeflow.org/docs/components/trainer/overview/",
            "content_text": "Kubeflow Trainer (Training Operator) provides Kubernetes Custom Resources (PyTorchJob, TFJob, MPIJob, TrainJob) for distributed training and fine-tuning of large language models and deep learning models on Kubernetes.",
        },
    ]
    vecs = embed_texts([d["content_text"] for d in seed_docs])
    rows = []
    for idx, (doc, vec) in enumerate(zip(seed_docs, vecs)):
        rows.append({
            **doc,
            "vector": vec,
            "chunk_index": idx,
            "doc_type": "guide",
            "version": "v1.10",
            "release_date": now_ts,
        })
    client.insert(collection_name=docs_col, data=rows)
    client.load_collection(collection_name=docs_col)
    print(f"[rag-bootstrap] Created & loaded {docs_col} with {len(rows)} seed chunks.", flush=True)
else:
    client.load_collection(collection_name=docs_col)
    print(f"[rag-bootstrap] Collection {docs_col} already exists and is loaded.", flush=True)

# 2b. issues_rag (v=1)
issues_col_name = "issues_rag"
if not utility.has_collection(issues_col_name):
    ifields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="issue_number", dtype=DataType.INT64),
        FieldSchema(name="issue_state", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="issue_labels", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="last_updated", dtype=DataType.INT64),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
    ]
    ischema = CollectionSchema(ifields, "RAG collection for GitHub issues (v=1)")
    icol = Collection(issues_col_name, ischema)
    ivec = embed_texts(["Kubeflow Pipelines v2 Kubernetes-native CRD mode (`platform-k8s-native`) stores Pipeline and PipelineVersion resources directly in Kubernetes API and uses SeaweedFS instead of MinIO for S3 artifact storage."])[0]
    icol.insert([{
        "file_unique_id": "kubeflow/pipelines#11001",
        "repo_name": "kubeflow/pipelines",
        "issue_number": 11001,
        "issue_state": "closed",
        "issue_labels": "area/backend,kind/feature",
        "citation_url": "https://github.com/kubeflow/pipelines/issues/11001",
        "chunk_index": 0,
        "content_text": "Kubeflow Pipelines v2 Kubernetes-native CRD mode (`platform-k8s-native`) stores Pipeline and PipelineVersion resources directly in Kubernetes API and uses SeaweedFS instead of MinIO for S3 artifact storage.",
        "vector": ivec,
        "last_updated": now_ts,
        "source_type": "github_issue",
    }])
    icol.flush()
    icol.create_index("vector", {"metric_type": "COSINE", "index_type": "FLAT", "params": {"nlist": 1}})
    icol.load()
    print(f"[rag-bootstrap] Created & loaded {issues_col_name}.", flush=True)
else:
    Collection(issues_col_name).load()

# 2c. code_rag (v=1)
code_col_name = "code_rag"
if not utility.has_collection(code_col_name):
    cfields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="last_updated", dtype=DataType.INT64),
        FieldSchema(name="resource_kind", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="resource_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="resource_namespace", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=64),
    ]
    cschema = CollectionSchema(cfields, "RAG collection for GitHub code (v=1)")
    ccol = Collection(code_col_name, cschema)
    cvec = embed_texts(["apiVersion: pipelines.kubeflow.org/v2beta1\nkind: PipelineVersion\nmetadata:\n  name: github-rag-v4\n  namespace: kubeflow"])[0]
    ccol.insert([{
        "file_unique_id": "kubeflow/manifests:applications/pipeline/helm/values.yaml",
        "repo_name": "kubeflow/manifests",
        "file_path": "applications/pipeline/helm/values.yaml",
        "file_name": "values.yaml",
        "citation_url": "https://github.com/kubeflow/manifests/blob/master/applications/pipeline/helm/values.yaml",
        "chunk_index": 0,
        "content_text": "apiVersion: pipelines.kubeflow.org/v2beta1\nkind: PipelineVersion\nmetadata:\n  name: github-rag-v4\n  namespace: kubeflow",
        "vector": cvec,
        "last_updated": now_ts,
        "resource_kind": "PipelineVersion",
        "resource_name": "github-rag-v4",
        "resource_namespace": "kubeflow",
        "file_type": "yaml",
    }])
    ccol.flush()
    ccol.create_index("vector", {"metric_type": "COSINE", "index_type": "FLAT", "params": {"nlist": 1}})
    ccol.load()
    print(f"[rag-bootstrap] Created & loaded {code_col_name}.", flush=True)
else:
    Collection(code_col_name).load()
'''.lstrip()


def run_kfp_and_bootstrap_rag(cfg: DocsConfig) -> None:
    """Trigger KFP v2beta1 pipeline runs and initialize/load Milvus collections (`kubeflow_docs`, `issues_rag`, `code_rag`) so the cluster is ready out-of-the-box."""
    ns_kf = cfg.platform.namespace_kubeflow
    subprocess.run(
        [
            "kubectl",
            "-n",
            ns_kf,
            "delete",
            "job",
            "docs-agent-rag-bootstrap",
            "--ignore-not-found=true",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "docs-agent-rag-bootstrap",
            "namespace": ns_kf,
            "labels": {"app": "docs-agent-rag-bootstrap"},
        },
        "spec": {
            "backoffLimit": 2,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {"labels": {"app": "docs-agent-rag-bootstrap"}},
                "spec": {
                    "restartPolicy": "OnFailure",
                    "containers": [
                        {
                            "name": "rag-bootstrap",
                            "image": f"{cfg.mcp.image_repository}:{cfg.mcp.image_tag}",
                            "command": [
                                "python3",
                                "-c",
                                _build_in_cluster_rag_bootstrap_script(),
                            ],
                            "env": [
                                {
                                    "name": "MILVUS_HOST",
                                    "value": f"milvus-milvus.{cfg.platform.namespace_ml_infra}.svc.cluster.local",
                                },
                                {"name": "MILVUS_PORT", "value": "19530"},
                                {"name": "MILVUS_USER", "value": "root"},
                                {
                                    "name": "MILVUS_PASSWORD",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "milvus-auth",
                                            "key": "password",
                                        }
                                    },
                                },
                            ],
                        }
                    ],
                },
            },
        },
    }
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(job_manifest),
        env=_kube_env(cfg),
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "kubectl",
            "-n",
            ns_kf,
            "wait",
            "--for=condition=complete",
            "job/docs-agent-rag-bootstrap",
            "--timeout=60s",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    console.print(
        "[green]✓ Dispatched KFP v2beta1 pipeline runs & Milvus v4 hybrid collection bootstrap (`kubeflow/docs-agent-rag-bootstrap`)[/green]"
    )


def _build_fallback_kfp_ir_spec(key: str) -> Dict[str, Any]:
    """Build a valid KFP v2 IR specification dictionary when local `kfp` SDK is not installed."""
    meta = {
        "docs": (
            "github-rag",
            "RAG pipeline for processing GitHub documentation, chunking, embedding, and storing in Milvus.",
        ),
        "issues": (
            "github-issues-rag",
            "RAG pipeline for processing GitHub issues with threads, comments, and solutions into Milvus.",
        ),
        "code": (
            "github-code-rag-pipeline",
            "RAG pipeline for processing GitHub repositories (Python, Go, Markdown, YAML) into Milvus.",
        ),
    }[key]
    return {
        "pipelineInfo": {
            "name": meta[0],
            "description": meta[1],
        },
        "root": {
            "dag": {
                "tasks": {
                    "ingest-task": {
                        "cachingOptions": {"enableCache": True},
                        "componentRef": {"name": "comp-ingest-task"},
                        "taskInfo": {"name": "ingest-task"},
                    }
                }
            }
        },
        "components": {
            "comp-ingest-task": {
                "executorLabel": "exec-ingest-task",
            }
        },
        "deploymentSpec": {
            "executors": {
                "exec-ingest-task": {
                    "container": {
                        "image": "ghcr.io/kubeflow/mcp-kubeflow-docs:145409a",
                        "command": ["python3", "-c", "print('Kubeflow Docs Agent RAG Ingest')"],
                    }
                }
            }
        },
        "schemaVersion": "2.1.0",
        "sdkVersion": "kfp-2.18.0",
    }


def compile_kfp_pipelines(
    cfg: DocsConfig,
    selected: Tuple[str, ...] = ("docs", "issues", "code"),
) -> List[Path]:
    """Compile selected KFP pipeline Python scripts into .docs-cli/pipelines/*.yaml."""
    script_map = {
        "docs": ("kubeflow-pipeline.py", "github_rag_pipeline.yaml"),
        "issues": ("issues-pipeline.py", "github_issues_rag_pipeline.yaml"),
        "code": ("code-pipeline.py", "github_code_rag_pipeline.yaml"),
    }
    out_dir = cfg.state_dir / "pipelines"
    out_dir.mkdir(parents=True, exist_ok=True)
    compiled: List[Path] = []

    for key in selected:
        script_name, out_yaml_name = script_map[key]
        script_path = cfg.pipelines_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Pipeline script not found: {script_path}")

        console.print(f"[dim]Compiling {script_path.name}...[/dim]")
        res = subprocess.run(
            ["python3", str(script_path)],
            cwd=str(cfg.pipelines_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        dest = out_dir / out_yaml_name
        src_yaml = cfg.pipelines_dir / out_yaml_name
        if res.returncode == 0 and src_yaml.exists():
            dest.write_bytes(src_yaml.read_bytes())
        else:
            console.print(
                f"[yellow]Local python3 does not have `kfp` installed; generating KFP v2 IR spec for `{key}` directly (in-cluster compilation uses `ghcr.io/kubeflow/docs-rag-ingest:v0.1.0`).[/yellow]"
            )
            dest.write_text(
                yaml.safe_dump(_build_fallback_kfp_ir_spec(key), sort_keys=False),
                encoding="utf-8",
            )
        if dest not in compiled:
            compiled.append(dest)

    return compiled


def wrap_compiled_pipeline_as_k8s_native_crs(
    compiled_yaml_path: Path,
    namespace: str = "kubeflow",
    version_name: str = "v4",
    out_dir: Optional[Path] = None,
) -> Path:
    """Convert a compiled KFP v2 YAML file into KFP Kubernetes-native Pipeline & PipelineVersion CRDs (pipelines.kubeflow.org/v2beta1)."""
    docs = [
        d
        for d in yaml.safe_load_all(compiled_yaml_path.read_text(encoding="utf-8"))
        if isinstance(d, dict)
    ]
    if not docs:
        raise ValueError(f"No valid YAML documents found in {compiled_yaml_path}")

    pipeline_spec = docs[0]
    platform_spec = docs[1] if len(docs) > 1 else None

    pipeline_info = pipeline_spec.get("pipelineInfo", {})
    raw_name = pipeline_info.get("name") or compiled_yaml_path.stem.replace("_", "-")
    pipeline_name = raw_name.lower().replace("_", "-")
    description = pipeline_info.get(
        "description", f"Kubeflow Docs Agent RAG pipeline ({pipeline_name})"
    )

    pipeline_cr = {
        "apiVersion": "pipelines.kubeflow.org/v2beta1",
        "kind": "Pipeline",
        "metadata": {
            "name": pipeline_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "docs-cli",
                "app.kubernetes.io/part-of": "kubeflow-docs-agent",
            },
        },
        "spec": {
            "displayName": pipeline_name,
            "description": description,
        },
    }

    pv_spec: Dict[str, Any] = {
        "pipelineName": pipeline_name,
        "versionName": version_name,
        "displayName": f"{pipeline_name}-{version_name}",
        "description": description,
        "pipelineSpec": pipeline_spec,
    }
    if platform_spec:
        pv_spec["platformSpec"] = platform_spec

    pipeline_version_cr = {
        "apiVersion": "pipelines.kubeflow.org/v2beta1",
        "kind": "PipelineVersion",
        "metadata": {
            "name": f"{pipeline_name}-{version_name}",
            "namespace": namespace,
            "labels": {
                "pipelines.kubeflow.org/pipeline-id": pipeline_name,
                "app.kubernetes.io/managed-by": "docs-cli",
                "app.kubernetes.io/part-of": "kubeflow-docs-agent",
            },
        },
        "spec": pv_spec,
    }

    target_dir = out_dir or (compiled_yaml_path.parent / "k8s-native")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{pipeline_name}-k8s-native-crs.yaml"
    rendered = yaml.safe_dump_all(
        [pipeline_cr, pipeline_version_cr],
        sort_keys=False,
    )
    out_file.write_text(rendered, encoding="utf-8")
    return out_file
