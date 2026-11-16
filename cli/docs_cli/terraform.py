"""Terraform templating, execution, and OKE kubeconfig retrieval for docs-cli."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console

from .auth import OciSession
from .config import DocsConfig

console = Console()


KUBEFLOW_MANIFESTS_PIN = "fb08d2728003f69560f880949414b3f914d6085b"


def ensure_kubeflow_manifests_charts(cfg: DocsConfig) -> Path:
    """Ensure `deployments/charts/kubeflow-manifests` is present at the pinned upstream revision (with CRI-O image qualification)."""
    target = cfg.repo_root / "deployments" / "charts" / "kubeflow-manifests"
    if not (target / "applications" / "pipeline" / "helm" / "Chart.yaml").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        console.print(
            f"[bold cyan]Fetching kubeflow/manifests Helm charts ({KUBEFLOW_MANIFESTS_PIN[:8]})...[/bold cyan]"
        )
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/kubeflow/manifests.git",
                str(target),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", KUBEFLOW_MANIFESTS_PIN],
            check=True,
        )
    common_res = target / "applications" / "pipeline" / "helm" / "manifests" / "common-resources.yaml"
    if common_res.exists():
        txt = common_res.read_text(encoding="utf-8")
        if "image: mysql:8.4" in txt:
            common_res.write_text(
                txt.replace("image: mysql:8.4", "image: docker.io/library/mysql:8.4"),
                encoding="utf-8",
            )
    return target


def discover_available_gpu_ads(session: OciSession, gpu_shape: str) -> list[str]:
    """Query OCI Compute Capacity Report to find Availability Domains with AVAILABLE GPU host capacity."""
    import oci

    available: list[str] = []
    try:
        iam = oci.identity.IdentityClient(session.config)
        compute = oci.core.ComputeClient(session.config)
        ads = [a.name for a in iam.list_availability_domains(session.tenancy_id).data]
        for ad in ads:
            rep = compute.create_compute_capacity_report(
                oci.core.models.CreateComputeCapacityReportDetails(
                    compartment_id=session.tenancy_id,
                    availability_domain=ad,
                    shape_availabilities=[
                        oci.core.models.CreateCapacityReportShapeAvailabilityDetails(
                            instance_shape=gpu_shape
                        )
                    ],
                )
            ).data
            if (
                rep.shape_availabilities
                and rep.shape_availabilities[0].availability_status == "AVAILABLE"
            ):
                available.append(ad)
        if available:
            console.print(
                f"[green]✓ OCI GPU Capacity Report ({gpu_shape}): AVAILABLE in {', '.join(available)}[/green]"
            )
    except Exception:
        pass
    return available


def build_oke_tfvars(
    cfg: DocsConfig,
    session: Optional[OciSession] = None,
    compartment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build variables dict for deployments/terraform/oke-cluster/."""
    comp_id = (
        compartment_id
        or cfg.oci.compartment_id
        or "<SET_OCI_COMPARTMENT_ID_IN_ENV>"
    )
    tenancy = (
        session.tenancy_id
        if session
        else (cfg.oci.tenancy_ocid or "<SET_OCI_TENANCY_IN_ENV>")
    )
    user = (
        session.user_id
        if session
        else (cfg.oci.user_ocid or "<SET_OCI_USER_IN_ENV>")
    )
    fingerprint = (
        session.fingerprint
        if session
        else (cfg.oci.fingerprint or "<SET_OCI_FINGERPRINT_IN_ENV>")
    )
    key_path = (
        session.key_file
        if session
        else (cfg.oci.private_key_path or "~/.oci/oci_api_key.pem")
    )
    region = session.region if session else cfg.oci.region
    gpu_ads = (
        discover_available_gpu_ads(session, cfg.cluster.gpu_node_shape)
        if (session and cfg.cluster.enable_gpu_pool)
        else []
    )

    tfvars: Dict[str, Any] = {
        "tenancy_ocid": tenancy,
        "user_ocid": user,
        "fingerprint": fingerprint,
        "private_key_path": key_path,
        "region": region,
        "compartment_ocid": comp_id,
        "cluster_name": cfg.cluster.name,
        "kubernetes_version": cfg.cluster.kubernetes_version,
        "cpu_node_shape": cfg.cluster.cpu_node_shape,
        "cpu_node_count": cfg.cluster.cpu_node_count,
        "cpu_node_ocpus": cfg.cluster.cpu_node_ocpus,
        "cpu_node_memory_gbs": cfg.cluster.cpu_node_memory_gb,
        "cpu_boot_volume_gbs": cfg.cluster.cpu_boot_volume_gb,
        "enable_gpu_pool": cfg.cluster.enable_gpu_pool,
        "gpu_node_shape": cfg.cluster.gpu_node_shape,
        "gpu_node_count": cfg.cluster.gpu_node_count,
        "gpu_node_ocpus": cfg.cluster.gpu_node_ocpus,
        "gpu_node_memory_gbs": cfg.cluster.gpu_node_memory_gb,
        "gpu_boot_volume_gbs": cfg.cluster.gpu_boot_volume_gb,
        "gpu_availability_domains": gpu_ads,
    }
    return tfvars


def build_platform_tfvars(cfg: DocsConfig) -> Dict[str, Any]:
    """Build variables dict for docs-agent-mcp/terraform/."""
    cors_origins = cfg.resolve_cors_origins()
    tfvars: Dict[str, Any] = {
        "kubeconfig_path": str(cfg.kubeconfig_path),
        "enable_istio_ingressgateway": True,
        "oci_lb_min_bandwidth_mbps": str(cfg.ingress.lb_min_bandwidth_mbps),
        "oci_lb_max_bandwidth_mbps": str(cfg.ingress.lb_max_bandwidth_mbps),
        "namespace_ml_infra": cfg.platform.namespace_ml_infra,
        "namespace_docs_agent": cfg.platform.namespace_docs_agent,
        "namespace_kubeflow": cfg.platform.namespace_kubeflow,
        "cert_manager_version": cfg.platform.cert_manager_version,
        "istio_version": cfg.platform.istio_version,
        "knative_version": cfg.platform.knative_version,
        "kserve_version": cfg.platform.kserve_version,
        "kfp_version": cfg.platform.kfp_version,
        "use_kubeflow_helm_charts": cfg.platform.use_kubeflow_helm_charts,
        "kubeflow_pipelines_scenario": cfg.platform.kubeflow_pipelines_scenario,
        "enable_kubeflow_full_platform_helm": cfg.platform.enable_kubeflow_full_platform_helm,
        "milvus_version": cfg.platform.milvus_version,
        "embeddings_model_id": cfg.platform.embeddings_model_id,
        "reranker_model_id": cfg.platform.reranker_model_id,
        "kagent_cors_allow_origins": cors_origins,
        "enable_session_auth": cfg.ingress.enable_jwt_guardrails,
        "enforce_session_auth": cfg.ingress.enable_jwt_guardrails,
        "enable_kagent_ingress": bool(cfg.ingress.domain and cfg.ingress.acme_email),
        "kagent_domain_name": cfg.ingress.domain,
        "kagent_acme_email": cfg.ingress.acme_email,
    }
    return tfvars


def render_tfvars_files(
    cfg: DocsConfig,
    session: Optional[OciSession] = None,
    compartment_id: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Write .docs-cli/oke.tfvars.json and .docs-cli/platform.tfvars.json."""
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    oke_path = cfg.state_dir / "oke.tfvars.json"
    platform_path = cfg.state_dir / "platform.tfvars.json"

    oke_vars = build_oke_tfvars(cfg, session=session, compartment_id=compartment_id)
    platform_vars = build_platform_tfvars(cfg)

    oke_path.write_text(json.dumps(oke_vars, indent=2) + "\n", encoding="utf-8")
    oke_path.chmod(0o600)

    platform_path.write_text(
        json.dumps(platform_vars, indent=2) + "\n", encoding="utf-8"
    )
    platform_path.chmod(0o600)

    return oke_path, platform_path


def run_terraform(
    tf_dir: Path,
    args: list[str],
    extra_env: Optional[Dict[str, str]] = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a terraform command in tf_dir."""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    cmd = ["terraform", f"-chdir={tf_dir}"] + args
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return subprocess.run(
        cmd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def get_terraform_outputs(tf_dir: Path) -> Dict[str, Any]:
    """Read `terraform output -json` from tf_dir and unwrap `.value`."""
    res = subprocess.run(
        ["terraform", f"-chdir={tf_dir}", "output", "-json"],
        check=False,
        text=True,
        capture_output=True,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    raw = json.loads(res.stdout)
    return {k: v.get("value") for k, v in raw.items() if isinstance(v, dict)}


def fetch_oke_kubeconfig(
    cfg: DocsConfig,
    session: OciSession,
    cluster_id: str,
) -> Path:
    """Fetch kubeconfig for an OKE cluster using the native `oci` CLI (with OCI SDK fallback)."""
    import shutil

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    kc_path = cfg.kubeconfig_path

    if shutil.which("oci"):
        cmd = [
            "oci",
            "ce",
            "cluster",
            "create-kubeconfig",
            "--cluster-id",
            cluster_id,
            "--file",
            str(kc_path),
            "--region",
            session.region,
            "--token-version",
            "2.0.0",
            "--kube-endpoint",
            "PUBLIC_ENDPOINT",
            "--overwrite",
        ]
        if cfg.oci.profile and not cfg.oci.tenancy_ocid:
            cmd.extend(["--profile", cfg.oci.profile])
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        res = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if res.returncode == 0 and kc_path.exists():
            kc_path.chmod(0o600)
            console.print(f"[green]✓ Wrote OKE kubeconfig via native `oci` CLI to {kc_path}[/green]")
            return kc_path

    import oci.container_engine.models as ce_models

    oke_client = session.oke()
    details = ce_models.CreateClusterKubeconfigContentDetails(
        token_version="2.0.0",
        endpoint="PUBLIC_ENDPOINT",
    )
    response = oke_client.create_kube_config(
        cluster_id=cluster_id,
        create_cluster_kubeconfig_content_details=details,
    )
    content = b"".join(chunk for chunk in response.data.raw.stream(4096, decode_content=True))
    kc_path.write_bytes(content)
    kc_path.chmod(0o600)
    console.print(f"[green]✓ Wrote OKE kubeconfig to {kc_path}[/green]")
    return kc_path
