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
from .gcp_auth import GcpSession

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


def discover_available_gcp_gpu_zones(
    gcp_session: GcpSession, accelerator_type: str
) -> list[str]:
    """Query `gcloud compute accelerator-types list` to discover zones in `gcp_session.region` offering the target GPU."""
    import shutil

    if not shutil.which("gcloud"):
        return [gcp_session.zone]
    try:
        gcp_session.ensure_gcloud_auth_if_service_account()
        res = subprocess.run(
            [
                "gcloud",
                "compute",
                "accelerator-types",
                "list",
                f"--project={gcp_session.project_id}",
                f"--filter=name={accelerator_type}",
                "--format=json",
                "--quiet",
            ],
            env=gcp_session.gcloud_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if res.returncode == 0 and res.stdout.strip():
            items = json.loads(res.stdout)
            zones = sorted(
                {
                    str(item.get("zone", "")).rsplit("/", 1)[-1]
                    for item in items
                    if str(item.get("zone", "")).rsplit("/", 1)[-1].startswith(f"{gcp_session.region}-")
                }
            )
            if zones:
                chosen = [gcp_session.zone] if gcp_session.zone in zones else [zones[0]]
                console.print(
                    f"[green]✓ GCP GPU Zone Discovery ({accelerator_type}): using {', '.join(chosen)} (available in {', '.join(zones)})[/green]"
                )
                return chosen
    except Exception:
        pass
    return [gcp_session.zone]


def build_gke_tfvars(
    cfg: DocsConfig,
    gcp_session: Optional[GcpSession] = None,
) -> Dict[str, Any]:
    """Build variables dict for deployments/terraform/gke-cluster/."""
    project_id = (
        gcp_session.project_id
        if gcp_session
        else (cfg.gcp.project_id or "<SET_GCP_PROJECT_ID_IN_ENV>")
    )
    region = gcp_session.region if gcp_session else cfg.gcp.region
    zone = gcp_session.zone if gcp_session else cfg.gcp.zone
    creds_file = (
        gcp_session.credentials_file
        if gcp_session
        else (cfg.gcp.credentials_file or None)
    )
    gpu_zones = (
        discover_available_gcp_gpu_zones(gcp_session, cfg.gcp.gpu_accelerator_type)
        if (gcp_session and cfg.cluster.enable_gpu_pool)
        else [zone]
    )

    tfvars: Dict[str, Any] = {
        "project_id": project_id,
        "region": region,
        "zone": zone,
        "credentials_file": creds_file,
        "cluster_name": cfg.cluster.name,
        "kubernetes_version": (
            ""
            if cfg.cluster.kubernetes_version == "v1.36.4"
            else cfg.cluster.kubernetes_version
        ),
        "vpc_cidr": cfg.cluster.vcn_cidr,
        "cpu_machine_type": cfg.gcp.cpu_machine_type,
        "cpu_node_count": cfg.cluster.cpu_node_count,
        "cpu_disk_size_gb": cfg.gcp.cpu_disk_size_gb,
        "cpu_disk_type": cfg.gcp.cpu_disk_type,
        "enable_gpu_pool": cfg.cluster.enable_gpu_pool,
        "gpu_machine_type": cfg.gcp.gpu_machine_type,
        "gpu_accelerator_type": cfg.gcp.gpu_accelerator_type,
        "gpu_accelerator_count": cfg.gcp.gpu_accelerator_count,
        "gpu_node_count": cfg.cluster.gpu_node_count,
        "gpu_disk_size_gb": cfg.gcp.gpu_disk_size_gb,
        "gpu_disk_type": cfg.gcp.gpu_disk_type,
        "gpu_zones": gpu_zones,
    }
    return tfvars


def build_platform_tfvars(cfg: DocsConfig) -> Dict[str, Any]:
    """Build variables dict for docs-agent-mcp/terraform/."""
    cors_origins = cfg.resolve_cors_origins()
    tfvars: Dict[str, Any] = {
        "kubeconfig_path": str(cfg.kubeconfig_path),
        "cloud_provider": cfg.cloud,
        "lb_service_annotations": dict(cfg.k8s.lb_service_annotations),
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


def build_k8s_descriptor(cfg: DocsConfig) -> Dict[str, Any]:
    """Build portable BYO-Kubernetes cluster descriptor written to .docs-cli/k8s.cluster.json."""
    return {
        "mode": "byo-kubernetes",
        "cluster_name": cfg.cluster.name,
        "kubeconfig_path": str(cfg.kubeconfig_path),
        "context": cfg.k8s.context,
        "storage_class": cfg.k8s.storage_class,
        "lb_service_annotations": dict(cfg.k8s.lb_service_annotations),
        "enable_gpu_pool": cfg.cluster.enable_gpu_pool,
    }


def render_tfvars_files(
    cfg: DocsConfig,
    session: Optional[OciSession] = None,
    compartment_id: Optional[str] = None,
    gcp_session: Optional[GcpSession] = None,
) -> Tuple[Path, Path]:
    """Write .docs-cli/{oke.tfvars.json,gke.tfvars.json,k8s.cluster.json} and .docs-cli/platform.tfvars.json."""
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    platform_path = cfg.state_dir / "platform.tfvars.json"
    platform_vars = build_platform_tfvars(cfg)
    platform_path.write_text(
        json.dumps(platform_vars, indent=2) + "\n", encoding="utf-8"
    )
    platform_path.chmod(0o600)

    if cfg.cloud == "k8s":
        k8s_path = cfg.state_dir / "k8s.cluster.json"
        k8s_path.write_text(
            json.dumps(build_k8s_descriptor(cfg), indent=2) + "\n", encoding="utf-8"
        )
        k8s_path.chmod(0o600)
        return k8s_path, platform_path

    if cfg.cloud == "gcp":
        gke_path = cfg.state_dir / "gke.tfvars.json"
        gke_vars = build_gke_tfvars(cfg, gcp_session=gcp_session)
        gke_path.write_text(json.dumps(gke_vars, indent=2) + "\n", encoding="utf-8")
        gke_path.chmod(0o600)
        return gke_path, platform_path

    oke_path = cfg.state_dir / "oke.tfvars.json"
    oke_vars = build_oke_tfvars(cfg, session=session, compartment_id=compartment_id)
    oke_path.write_text(json.dumps(oke_vars, indent=2) + "\n", encoding="utf-8")
    oke_path.chmod(0o600)
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


def fetch_gke_kubeconfig(
    cfg: DocsConfig,
    gcp_session: GcpSession,
    cluster_name: Optional[str] = None,
    location: Optional[str] = None,
) -> Path:
    """Fetch kubeconfig for a GKE cluster into `.docs-cli/kubeconfig` using `gcloud container clusters get-credentials`."""
    import shutil

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    kc_path = cfg.kubeconfig_path
    target_cluster = cluster_name or cfg.cluster.name
    target_loc = location or gcp_session.zone

    if not shutil.which("gcloud"):
        raise RuntimeError(
            "Google Cloud SDK (`gcloud`) is required to fetch GKE credentials. "
            "Install `google-cloud-cli` and `gke-gcloud-auth-plugin`."
        )

    gcp_session.ensure_gcloud_auth_if_service_account()
    env = gcp_session.gcloud_env()
    env["KUBECONFIG"] = str(kc_path)

    cmd = [
        "gcloud",
        "container",
        "clusters",
        "get-credentials",
        target_cluster,
        f"--location={target_loc}",
        f"--project={gcp_session.project_id}",
        "--quiet",
    ]
    console.print(f"[dim]$ KUBECONFIG={kc_path} {' '.join(cmd)}[/dim]")
    res = subprocess.run(cmd, env=env, check=False, text=True, capture_output=True)
    if res.returncode != 0 or not kc_path.exists():
        raise RuntimeError(
            f"Failed to fetch GKE kubeconfig for '{target_cluster}' ({target_loc}): {res.stderr.strip() or res.stdout.strip()}"
        )
    kc_path.chmod(0o600)
    console.print(f"[green]✓ Wrote GKE kubeconfig via `gcloud` to {kc_path}[/green]")
    return kc_path


def import_k8s_kubeconfig(
    cfg: DocsConfig,
    source_path: Optional[Path] = None,
    context: Optional[str] = None,
) -> Path:
    """Import and flatten a Bring-Your-Own Kubernetes kubeconfig into `.docs-cli/kubeconfig-k8s` without mutating ~/.kube/config."""
    import shutil

    src = (source_path or cfg.source_kubeconfig_path).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(
            f"BYO Kubernetes kubeconfig not found at '{src}'. "
            "Pass `--kubeconfig /path/to/kubeconfig` or set `DOCS_KUBECONFIG` / `KUBECONFIG`."
        )

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.state_dir / "kubeconfig-k8s"
    target_ctx = (context if context is not None else cfg.k8s.context).strip()

    if shutil.which("kubectl"):
        cmd = [
            "kubectl",
            f"--kubeconfig={src}",
            "config",
            "view",
            "--raw",
            "--flatten",
        ]
        if target_ctx:
            cmd.extend(["--minify", f"--context={target_ctx}"])
        res = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to read kubeconfig '{src}'"
                + (f" (context='{target_ctx}')" if target_ctx else "")
                + f": {res.stderr.strip() or res.stdout.strip()}"
            )
        if res.stdout.strip():
            dest.write_text(res.stdout, encoding="utf-8")
            dest.chmod(0o600)
            ctx_note = f" (context={target_ctx})" if target_ctx else ""
            console.print(
                f"[green]✓ Imported flattened BYO Kubernetes kubeconfig{ctx_note} -> {dest}[/green]"
            )
            return dest

    if src != dest:
        shutil.copyfile(src, dest)
    dest.chmod(0o600)
    console.print(f"[green]✓ Copied BYO Kubernetes kubeconfig -> {dest}[/green]")
    return dest
