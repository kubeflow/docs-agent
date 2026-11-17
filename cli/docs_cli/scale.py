"""Cluster node-pool scaling (scale-down / hibernate, scale-up / wake) and in-cluster TTL auto-scaler."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .auth import OciSession
from .config import DocsConfig
from .gcp_auth import GcpSession
from .terraform import get_terraform_outputs

console = Console()


def parse_duration_seconds(duration_str: str) -> int:
    """Parse a human-friendly duration string (e.g. '2h', '90m', '1h30m', '3600s') into seconds."""
    raw = (duration_str or "").strip().lower()
    if not raw:
        raise ValueError("Duration cannot be empty (examples: '2h', '90m', '45m', '3600s').")

    if raw.isdigit():
        # Plain integer defaults to minutes for convenience (e.g. "60" -> 60 minutes)
        return int(raw) * 60

    pattern = re.compile(r"^((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s)?$")
    match = pattern.match(raw)
    if not match or not any(match.groupdict().values()):
        raise ValueError(
            f"Invalid duration '{duration_str}'. Use formats like '30m', '2h', '1h30m', or '3600s'."
        )

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError("Duration must be greater than 0 seconds.")
    return total


def _kube_env(cfg: DocsConfig) -> Dict[str, str]:
    env = dict(os.environ)
    if cfg.kubeconfig_path.exists():
        env["KUBECONFIG"] = str(cfg.kubeconfig_path)
    return env


def discover_cluster_node_pools(
    cfg: DocsConfig,
    session: OciSession,
    compartment_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Discover CPU and GPU node pools for the configured OKE cluster."""
    oke_client = session.oke()
    tf_outputs = get_terraform_outputs(cfg.oke_tf_dir)
    cluster_id = tf_outputs.get("cluster_id")

    pools = oke_client.list_node_pools(compartment_id=compartment_id).data
    result: Dict[str, Dict[str, Any]] = {}

    cpu_expected = f"{cfg.cluster.name}-cpu-pool"
    gpu_expected = f"{cfg.cluster.name}-gpu-pool"

    for p in pools:
        if cluster_id and p.cluster_id != cluster_id:
            continue
        if p.name == cpu_expected or p.id == tf_outputs.get("cpu_node_pool_id"):
            full_pool = oke_client.get_node_pool(p.id).data
            result["cpu"] = {
                "id": p.id,
                "name": p.name,
                "shape": p.node_shape,
                "size": full_pool.node_config_details.size if full_pool.node_config_details else 0,
                "cluster_id": p.cluster_id,
            }
        elif p.name == gpu_expected or p.id == tf_outputs.get("gpu_node_pool_id"):
            full_pool = oke_client.get_node_pool(p.id).data
            result["gpu"] = {
                "id": p.id,
                "name": p.name,
                "shape": p.node_shape,
                "size": full_pool.node_config_details.size if full_pool.node_config_details else 0,
                "cluster_id": p.cluster_id,
            }

    return result


def scale_node_pools(
    cfg: DocsConfig,
    session: OciSession,
    compartment_id: str,
    cpu_size: Optional[int],
    gpu_size: Optional[int],
    apply: bool = False,
    release_frontend: Optional[bool] = None,
) -> Dict[str, Dict[str, Any]]:
    """Scale OKE CPU and/or GPU node pools to the target node counts, and release/restore the demo frontend."""
    import oci
    from .kube import get_ingress_lb_ip, set_demo_web_released

    pools = discover_cluster_node_pools(cfg, session, compartment_id)
    if not pools:
        raise RuntimeError(
            f"No OKE node pools found for cluster '{cfg.cluster.name}' in compartment {compartment_id}."
        )

    table = Table(title=f"OKE Node Pool Scaling Plan — {cfg.cluster.name}")
    table.add_column("Pool", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Shape")
    table.add_column("Current Size", justify="right")
    table.add_column("Target Size", justify="right", style="bold green")

    targets: Dict[str, int] = {}
    if "cpu" in pools and cpu_size is not None:
        targets["cpu"] = cpu_size
        table.add_row(
            "CPU",
            pools["cpu"]["name"],
            pools["cpu"]["shape"],
            str(pools["cpu"]["size"]),
            str(cpu_size),
        )
    if "gpu" in pools and gpu_size is not None:
        targets["gpu"] = gpu_size
        table.add_row(
            "GPU",
            pools["gpu"]["name"],
            pools["gpu"]["shape"],
            str(pools["gpu"]["size"]),
            str(gpu_size),
        )

    console.print(table)

    if not apply:
        console.print(
            "[yellow]Dry-run mode.[/yellow] Pass [bold]--yes[/bold] (or [bold]--apply[/bold]) to execute node pool scaling."
        )
        return pools

    oke_client = session.oke()

    # If scaling GPU down to 0 and kubeconfig is reachable, gracefully scale llm-predictor to 0 first
    # and release the demo frontend so it does not serve stale/offline routes during scale-down.
    is_scaling_down = targets.get("gpu") == 0 or targets.get("cpu") == 0
    should_release_fe = release_frontend if release_frontend is not None else is_scaling_down
    if should_release_fe and cfg.kubeconfig_path.exists():
        lb_ip = get_ingress_lb_ip(cfg, timeout_seconds=5)
        domain = cfg.resolve_agent_domain(lb_ip=lb_ip)
        set_demo_web_released(cfg, domain=domain, released=True)

    if targets.get("gpu") == 0 and cfg.kubeconfig_path.exists():
        subprocess.run(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "scale",
                "deployment/llm-predictor",
                "--replicas=0",
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )

    # Scale GPU pool first when scaling down, or CPU pool first when scaling up
    ordered_keys = (
        ["gpu", "cpu"]
        if (targets.get("gpu", 1) == 0)
        else ["cpu", "gpu"]
    )

    for key in ordered_keys:
        if key not in targets or key not in pools:
            continue
        pool_info = pools[key]
        new_size = targets[key]
        if pool_info["size"] == new_size:
            console.print(
                f"[dim]• {pool_info['name']} is already at size={new_size}; skipping.[/dim]"
            )
            continue

        console.print(
            f"[bold cyan]Scaling {pool_info['name']} ({pool_info['shape']}) -> {new_size} node(s)...[/bold cyan]"
        )
        details = oci.container_engine.models.UpdateNodePoolDetails(
            node_config_details=oci.container_engine.models.UpdateNodePoolNodeConfigDetails(
                size=new_size
            )
        )
        oke_client.update_node_pool(
            node_pool_id=pool_info["id"],
            update_node_pool_details=details,
        )
        console.print(
            f"[green]✓ Dispatched OCI node-pool update for {pool_info['name']} (size={new_size})[/green]"
        )

    # If scaling GPU >= 1 and kubeconfig is reachable, ensure llm-predictor is scaled back to 1
    if targets.get("gpu", 0) >= 1 and cfg.kubeconfig_path.exists():
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
    if (targets.get("cpu", 0) >= 1 or targets.get("gpu", 0) >= 1) and release_frontend is False and cfg.kubeconfig_path.exists():
        lb_ip = get_ingress_lb_ip(cfg, timeout_seconds=5)
        domain = cfg.resolve_agent_domain(lb_ip=lb_ip)
        set_demo_web_released(cfg, domain=domain, released=False)

    return pools


def _build_in_cluster_oci_autoscale_script() -> str:
    """Return a self-contained Python 3 + OpenSSL script that signs OCI REST requests and scales node pools to 0."""
    return r'''
import base64
import datetime
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.request

ttl_seconds = int(os.environ.get("TTL_SECONDS", "7200"))
gpu_only = os.environ.get("GPU_ONLY", "false").lower() == "true"
tenancy = os.environ["OCI_TENANCY"]
user = os.environ["OCI_USER"]
fingerprint = os.environ["OCI_FINGERPRINT"]
region = os.environ["OCI_REGION"]
key_pem = os.environ["OCI_PRIVATE_KEY_PEM"]
cpu_pool_id = os.environ.get("CPU_POOL_ID", "")
gpu_pool_id = os.environ.get("GPU_POOL_ID", "")
placement_json = os.environ.get("PLACEMENT_CONFIGS_JSON", "{}")
placements_by_pool = json.loads(placement_json)

start_ts = int(time.time())
end_ts = start_ts + ttl_seconds
print(f"[docs-agent-ttl] Armed auto-scale-down timer for {ttl_seconds}s (until {datetime.datetime.fromtimestamp(end_ts, datetime.timezone.utc).isoformat()})", flush=True)

while True:
    now = int(time.time())
    remaining = end_ts - now
    if remaining <= 0:
        break
    sleep_chunk = min(60, remaining)
    if remaining % 300 < 60 or remaining <= 120:
        print(f"[docs-agent-ttl] {remaining}s remaining before auto-scale-down...", flush=True)
    time.sleep(sleep_chunk)

print("[docs-agent-ttl] TTL expired! Scaling OKE node pools down to 0...", flush=True)

with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as kf:
    kf.write(key_pem)
    key_path = kf.name

def oci_put_node_pool_size(pool_id: str, target_size: int) -> None:
    if not pool_id:
        return
    host = f"containerengine.{region}.oci.oraclecloud.com"
    path = f"/20180222/nodePools/{pool_id}"
    url = f"https://{host}{path}"
    p_cfgs = placements_by_pool.get(pool_id, [])
    node_cfg = {"size": target_size}
    if p_cfgs:
        node_cfg["placementConfigs"] = p_cfgs
    body_bytes = json.dumps({"nodeConfigDetails": node_cfg}).encode("utf-8")
    body_sha256 = base64.b64encode(hashlib.sha256(body_bytes).digest()).decode("ascii")
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    signing_lines = [
        f"date: {date_str}",
        f"(request-target): put {path}",
        f"host: {host}",
        f"content-length: {len(body_bytes)}",
        f"content-type: application/json",
        f"x-content-sha256: {body_sha256}",
    ]
    signing_str = "\n".join(signing_lines).encode("ascii")
    sig_proc = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", key_path],
        input=signing_str,
        capture_output=True,
        check=True,
    )
    signature = base64.b64encode(sig_proc.stdout).decode("ascii")
    key_id = f"{tenancy}/{user}/{fingerprint}"
    auth_header = (
        f'Signature version="1",'
        f'keyId="{key_id}",'
        f'algorithm="rsa-sha256",'
        f'headers="date (request-target) host content-length content-type x-content-sha256",'
        f'signature="{signature}"'
    )

    req = urllib.request.Request(
        url,
        data=body_bytes,
        method="PUT",
        headers={
            "Date": date_str,
            "Host": host,
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
            "x-content-sha256": body_sha256,
            "Authorization": auth_header,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"[docs-agent-ttl] Scaled node pool {pool_id} -> size={target_size} (HTTP {resp.status})", flush=True)
    except Exception as exc:
        print(f"[docs-agent-ttl] Warning scaling {pool_id}: {exc}", flush=True)

# Scale GPU pool to 0 first, then CPU pool to 0 (unless gpu_only)
if gpu_pool_id:
    oci_put_node_pool_size(gpu_pool_id, 0)
if not gpu_only and cpu_pool_id:
    oci_put_node_pool_size(cpu_pool_id, 0)

print("[docs-agent-ttl] Auto-scale-down complete.", flush=True)
'''.lstrip()


def schedule_in_cluster_ttl_job(
    cfg: DocsConfig,
    session: OciSession,
    compartment_id: str,
    duration_str: str,
    gpu_only: bool = False,
) -> int:
    """Deploy an in-cluster Kubernetes Job in kube-system that automatically scales down OKE node pools after `duration_str`."""
    ttl_seconds = parse_duration_seconds(duration_str)
    pools = discover_cluster_node_pools(cfg, session, compartment_id)
    if not pools:
        raise RuntimeError("Could not discover OKE node pools to attach TTL auto-scaler.")

    oke_client = session.oke()
    placements_by_pool: Dict[str, list] = {}
    for k, info in pools.items():
        np_data = oke_client.get_node_pool(info["id"]).data
        if np_data.node_config_details and np_data.node_config_details.placement_configs:
            placements_by_pool[info["id"]] = [
                {
                    "availabilityDomain": pc.availability_domain,
                    "subnetId": pc.subnet_id,
                }
                for pc in np_data.node_config_details.placement_configs
            ]

    key_pem = Path(session.key_file).expanduser().read_text(encoding="utf-8")

    secret_manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "docs-agent-autoscale-oci",
            "namespace": "kube-system",
            "labels": {"app": "docs-agent-autoscale-ttl"},
        },
        "type": "Opaque",
        "stringData": {
            "OCI_TENANCY": session.tenancy_id,
            "OCI_USER": session.user_id,
            "OCI_FINGERPRINT": session.fingerprint,
            "OCI_REGION": session.region,
            "OCI_PRIVATE_KEY_PEM": key_pem,
            "CPU_POOL_ID": pools.get("cpu", {}).get("id", ""),
            "GPU_POOL_ID": pools.get("gpu", {}).get("id", ""),
            "PLACEMENT_CONFIGS_JSON": json.dumps(placements_by_pool),
        },
    }
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(secret_manifest),
        env=_kube_env(cfg),
        check=True,
        text=True,
        capture_output=True,
    )

    # Delete any existing TTL job first
    subprocess.run(
        [
            "kubectl",
            "-n",
            "kube-system",
            "delete",
            "job",
            "docs-agent-autoscale-ttl",
            "--ignore-not-found=true",
        ],
        env=_kube_env(cfg),
        check=False,
        capture_output=True,
        text=True,
    )

    scheduled_at = int(time.time())
    expires_at = scheduled_at + ttl_seconds

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "docs-agent-autoscale-ttl",
            "namespace": "kube-system",
            "labels": {"app": "docs-agent-autoscale-ttl"},
            "annotations": {
                "docs-agent.kubeflow.org/ttl-duration": duration_str,
                "docs-agent.kubeflow.org/ttl-seconds": str(ttl_seconds),
                "docs-agent.kubeflow.org/scheduled-at": str(scheduled_at),
                "docs-agent.kubeflow.org/expires-at": str(expires_at),
                "docs-agent.kubeflow.org/gpu-only": str(gpu_only).lower(),
            },
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {"labels": {"app": "docs-agent-autoscale-ttl"}},
                "spec": {
                    "restartPolicy": "OnFailure",
                    "nodeSelector": {"docs-agent-pool": "cpu"},
                    "containers": [
                        {
                            "name": "autoscale-timer",
                            "image": "mirror.gcr.io/library/python:3.12-slim",
                            "command": [
                                "python3",
                                "-c",
                                _build_in_cluster_oci_autoscale_script(),
                            ],
                            "env": [
                                {"name": "TTL_SECONDS", "value": str(ttl_seconds)},
                                {"name": "GPU_ONLY", "value": str(gpu_only).lower()},
                            ],
                            "envFrom": [
                                {
                                    "secretRef": {
                                        "name": "docs-agent-autoscale-oci",
                                    }
                                }
                            ],
                            "resources": {
                                "requests": {"cpu": "10m", "memory": "32Mi"},
                                "limits": {"cpu": "100m", "memory": "128Mi"},
                            },
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

    scope_str = "GPU node pool only (size=0)" if gpu_only else "ALL node pools (GPU=0, CPU=0)"
    console.print(
        Panel(
            f"[bold green]✓ In-Cluster Auto-Scale-Down Timer Armed![/bold green]\n"
            f"• [bold]Job:[/bold]          [cyan]kube-system/docs-agent-autoscale-ttl[/cyan]\n"
            f"• [bold]TTL Duration:[/bold] [cyan]{duration_str} ({ttl_seconds}s)[/cyan]\n"
            f"• [bold]Action:[/bold]       Scale {scope_str}\n"
            f"• [bold]Cancel anytime:[/bold] `docs-cli ttl --cancel`",
            title="Cluster Auto-Scale Timer",
        )
    )
    return ttl_seconds


def cancel_in_cluster_ttl_job(cfg: DocsConfig) -> None:
    """Cancel any active in-cluster auto-scale-down TTL Job."""
    subprocess.run(
        [
            "kubectl",
            "-n",
            "kube-system",
            "delete",
            "job",
            "docs-agent-autoscale-ttl",
            "--ignore-not-found=true",
        ],
        env=_kube_env(cfg),
        check=False,
        capture_output=True,
        text=True,
    )
    console.print("[green]✓ Cancelled in-cluster auto-scale-down timer (`kube-system/docs-agent-autoscale-ttl`).[/green]")


def get_in_cluster_ttl_status(cfg: DocsConfig) -> Optional[Dict[str, Any]]:
    """Return status metadata for the active in-cluster TTL Job if present."""
    res = subprocess.run(
        [
            "kubectl",
            "-n",
            "kube-system",
            "get",
            "job",
            "docs-agent-autoscale-ttl",
            "-o",
            "json",
        ],
        env=_kube_env(cfg),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return None
    data = json.loads(res.stdout)
    ann = data.get("metadata", {}).get("annotations", {})
    expires_at = int(ann.get("docs-agent.kubeflow.org/expires-at", "0"))
    remaining = max(0, expires_at - int(time.time())) if expires_at else 0
    return {
        "duration": ann.get("docs-agent.kubeflow.org/ttl-duration", ""),
        "ttl_seconds": int(ann.get("docs-agent.kubeflow.org/ttl-seconds", "0")),
        "expires_at": expires_at,
        "remaining_seconds": remaining,
        "gpu_only": ann.get("docs-agent.kubeflow.org/gpu-only", "false") == "true",
        "active": data.get("status", {}).get("active", 0),
        "succeeded": data.get("status", {}).get("succeeded", 0),
    }


# =============================================================================
# Google Cloud GKE Node Pool Scaling & In-Cluster TTL Auto-Scaler
# =============================================================================


def discover_gke_node_pools(
    cfg: DocsConfig,
    gcp_session: GcpSession,
) -> Dict[str, Dict[str, Any]]:
    """Discover CPU and GPU node pools for the configured GKE cluster via `gcloud`."""
    import shutil

    if not shutil.which("gcloud"):
        raise RuntimeError(
            "Google Cloud SDK (`gcloud`) is required for GKE node-pool scaling."
        )
    gcp_session.ensure_gcloud_auth_if_service_account()
    tf_outputs = get_terraform_outputs(cfg.gke_tf_dir)
    cluster_name = str(tf_outputs.get("cluster_name") or cfg.cluster.name)
    location = str(tf_outputs.get("location") or gcp_session.zone)

    res = subprocess.run(
        [
            "gcloud",
            "container",
            "node-pools",
            "list",
            f"--cluster={cluster_name}",
            f"--location={location}",
            f"--project={gcp_session.project_id}",
            "--format=json",
            "--quiet",
        ],
        env=gcp_session.gcloud_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return {}

    pools_raw = json.loads(res.stdout)
    # Count live Ready/NotReady nodes per pool via kubectl when reachable, falling back to initialNodeCount
    live_counts: Dict[str, int] = {}
    if cfg.kubeconfig_path.exists():
        nres = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json", "--request-timeout=5s"],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )
        if nres.returncode == 0 and nres.stdout.strip():
            for item in json.loads(nres.stdout).get("items", []):
                pool_lbl = (
                    (item.get("metadata") or {})
                    .get("labels", {})
                    .get("cloud.google.com/gke-nodepool", "")
                )
                if pool_lbl:
                    live_counts[pool_lbl] = live_counts.get(pool_lbl, 0) + 1

    cpu_expected = f"{cluster_name}-cpu-pool"
    gpu_expected = f"{cluster_name}-gpu-pool"
    result: Dict[str, Dict[str, Any]] = {}
    for p in pools_raw:
        pname = str(p.get("name") or "")
        mtype = str((p.get("config") or {}).get("machineType") or "")
        size = live_counts.get(pname, int(p.get("initialNodeCount", 0)))
        entry = {
            "id": pname,
            "name": pname,
            "shape": mtype,
            "size": size,
            "cluster_name": cluster_name,
            "location": location,
        }
        if pname == cpu_expected or pname.endswith("-cpu-pool"):
            result["cpu"] = entry
        elif pname == gpu_expected or pname.endswith("-gpu-pool"):
            result["gpu"] = entry
    return result


def scale_gke_node_pools(
    cfg: DocsConfig,
    gcp_session: GcpSession,
    cpu_size: Optional[int],
    gpu_size: Optional[int],
    apply: bool = False,
    release_frontend: Optional[bool] = None,
) -> Dict[str, Dict[str, Any]]:
    """Scale GKE CPU and/or GPU node pools to target node counts and manage demo frontend routing."""
    from .kube import get_ingress_lb_ip, set_demo_web_released

    pools = discover_gke_node_pools(cfg, gcp_session)
    if not pools:
        raise RuntimeError(
            f"No GKE node pools found for cluster '{cfg.cluster.name}' in project '{gcp_session.project_id}' ({gcp_session.zone})."
        )

    table = Table(title=f"GKE Node Pool Scaling Plan — {cfg.cluster.name}")
    table.add_column("Pool", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Machine Type")
    table.add_column("Current Size", justify="right")
    table.add_column("Target Size", justify="right", style="bold green")

    targets: Dict[str, int] = {}
    if "cpu" in pools and cpu_size is not None:
        targets["cpu"] = cpu_size
        table.add_row(
            "CPU",
            pools["cpu"]["name"],
            pools["cpu"]["shape"],
            str(pools["cpu"]["size"]),
            str(cpu_size),
        )
    if "gpu" in pools and gpu_size is not None:
        targets["gpu"] = gpu_size
        table.add_row(
            "GPU",
            pools["gpu"]["name"],
            pools["gpu"]["shape"],
            str(pools["gpu"]["size"]),
            str(gpu_size),
        )

    console.print(table)
    if not apply:
        console.print(
            "[yellow]Dry-run mode.[/yellow] Pass [bold]--yes[/bold] (or [bold]--apply[/bold]) to execute GKE node pool scaling."
        )
        return pools

    is_scaling_down = targets.get("gpu") == 0 or targets.get("cpu") == 0
    should_release_fe = release_frontend if release_frontend is not None else is_scaling_down
    if should_release_fe and cfg.kubeconfig_path.exists():
        lb_ip = get_ingress_lb_ip(cfg, timeout_seconds=5)
        domain = cfg.resolve_agent_domain(lb_ip=lb_ip)
        set_demo_web_released(cfg, domain=domain, released=True)

    if targets.get("gpu") == 0 and cfg.kubeconfig_path.exists():
        subprocess.run(
            [
                "kubectl",
                "-n",
                cfg.platform.namespace_ml_infra,
                "scale",
                "deployment/llm-predictor",
                "--replicas=0",
            ],
            env=_kube_env(cfg),
            capture_output=True,
            text=True,
            check=False,
        )

    ordered_keys = ["gpu", "cpu"] if (targets.get("gpu", 1) == 0) else ["cpu", "gpu"]
    for key in ordered_keys:
        if key not in targets or key not in pools:
            continue
        pool_info = pools[key]
        new_size = targets[key]
        console.print(
            f"[bold cyan]Scaling GKE pool {pool_info['name']} ({pool_info['shape']}) -> {new_size} node(s)...[/bold cyan]"
        )
        subprocess.run(
            [
                "gcloud",
                "container",
                "clusters",
                "resize",
                pool_info["cluster_name"],
                f"--node-pool={pool_info['name']}",
                f"--num-nodes={new_size}",
                f"--location={pool_info['location']}",
                f"--project={gcp_session.project_id}",
                "--quiet",
            ],
            env=gcp_session.gcloud_env(),
            check=True,
        )
        console.print(
            f"[green]✓ Scaled GKE node pool {pool_info['name']} -> {new_size} node(s)[/green]"
        )

    if targets.get("gpu", 0) >= 1 and cfg.kubeconfig_path.exists():
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
    if (targets.get("cpu", 0) >= 1 or targets.get("gpu", 0) >= 1) and release_frontend is False and cfg.kubeconfig_path.exists():
        lb_ip = get_ingress_lb_ip(cfg, timeout_seconds=5)
        domain = cfg.resolve_agent_domain(lb_ip=lb_ip)
        set_demo_web_released(cfg, domain=domain, released=False)

    return pools


def _build_in_cluster_gcp_autoscale_script() -> str:
    """Return a self-contained Python 3 + OpenSSL script that authenticates to GCP (SA JWT or Metadata Server) and resizes GKE node pools to 0."""
    return r'''
import base64
import datetime
import json
import os
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

ttl_seconds = int(os.environ.get("TTL_SECONDS", "7200"))
gpu_only = os.environ.get("GPU_ONLY", "false").lower() == "true"
project_id = os.environ["GCP_PROJECT_ID"]
location = os.environ["GKE_LOCATION"]
cluster_name = os.environ["GKE_CLUSTER_NAME"]
cpu_pool = os.environ.get("CPU_POOL_NAME", "")
gpu_pool = os.environ.get("GPU_POOL_NAME", "")
sa_json_str = os.environ.get("GCP_SA_JSON", "")

start_ts = int(time.time())
end_ts = start_ts + ttl_seconds
print(f"[docs-agent-ttl-gke] Armed auto-scale-down timer for {ttl_seconds}s (until {datetime.datetime.fromtimestamp(end_ts, datetime.timezone.utc).isoformat()})", flush=True)

while True:
    now = int(time.time())
    remaining = end_ts - now
    if remaining <= 0:
        break
    sleep_chunk = min(60, remaining)
    if remaining % 300 < 60 or remaining <= 120:
        print(f"[docs-agent-ttl-gke] {remaining}s remaining before GKE auto-scale-down...", flush=True)
    time.sleep(sleep_chunk)

print("[docs-agent-ttl-gke] TTL expired! Scaling GKE node pools down to 0...", flush=True)

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def get_gcp_access_token() -> str:
    if sa_json_str.strip():
        sa = json.loads(sa_json_str)
        now_s = int(time.time())
        header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode("utf-8"))
        claims = _b64url(json.dumps({
            "iss": sa["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": sa.get("token_uri", "https://oauth2.googleapis.com/token"),
            "iat": now_s,
            "exp": now_s + 3600,
        }).encode("utf-8"))
        signing_input = f"{header}.{claims}".encode("ascii")
        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as kf:
            kf.write(sa["private_key"])
            key_path = kf.name
        sig = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", key_path],
            input=signing_input,
            capture_output=True,
            check=True,
        ).stdout
        jwt_assertion = f"{header}.{claims}.{_b64url(sig)}"
        body = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_assertion,
        }).encode("utf-8")
        req = urllib.request.Request(
            sa.get("token_uri", "https://oauth2.googleapis.com/token"),
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["access_token"]

    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]

token = get_gcp_access_token()

def gke_set_pool_size(pool_name: str, target_size: int) -> None:
    if not pool_name:
        return
    url = f"https://container.googleapis.com/v1/projects/{project_id}/locations/{location}/clusters/{cluster_name}/nodePools/{pool_name}:setSize"
    body_bytes = json.dumps({"nodeCount": target_size}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"[docs-agent-ttl-gke] Scaled GKE pool {pool_name} -> size={target_size} (HTTP {resp.status})", flush=True)
                return
        except Exception as exc:
            print(f"[docs-agent-ttl-gke] Attempt {attempt+1} scaling {pool_name}: {exc}", flush=True)
            time.sleep(15)

if gpu_pool:
    gke_set_pool_size(gpu_pool, 0)
if not gpu_only and cpu_pool:
    gke_set_pool_size(cpu_pool, 0)

print("[docs-agent-ttl-gke] Auto-scale-down complete.", flush=True)
'''.lstrip()


def schedule_in_cluster_gcp_ttl_job(
    cfg: DocsConfig,
    gcp_session: GcpSession,
    duration_str: str,
    gpu_only: bool = False,
) -> int:
    """Deploy an in-cluster Kubernetes Job in kube-system that automatically scales down GKE node pools after `duration_str`."""
    ttl_seconds = parse_duration_seconds(duration_str)
    pools = discover_gke_node_pools(cfg, gcp_session)
    cluster_name = (
        pools.get("cpu", {}).get("cluster_name")
        or pools.get("gpu", {}).get("cluster_name")
        or cfg.cluster.name
    )
    location = (
        pools.get("cpu", {}).get("location")
        or pools.get("gpu", {}).get("location")
        or gcp_session.zone
    )

    sa_json = ""
    if gcp_session.credentials_file and Path(gcp_session.credentials_file).exists():
        raw_txt = Path(gcp_session.credentials_file).read_text(encoding="utf-8")
        try:
            if json.loads(raw_txt).get("type") == "service_account":
                sa_json = raw_txt
        except Exception:
            pass

    secret_manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "docs-agent-autoscale-gcp",
            "namespace": "kube-system",
            "labels": {"app": "docs-agent-autoscale-ttl"},
        },
        "type": "Opaque",
        "stringData": {
            "GCP_PROJECT_ID": gcp_session.project_id,
            "GKE_LOCATION": str(location),
            "GKE_CLUSTER_NAME": str(cluster_name),
            "CPU_POOL_NAME": str(pools.get("cpu", {}).get("name", f"{cluster_name}-cpu-pool")),
            "GPU_POOL_NAME": str(pools.get("gpu", {}).get("name", f"{cluster_name}-gpu-pool" if cfg.cluster.enable_gpu_pool else "")),
            "GCP_SA_JSON": sa_json,
        },
    }
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(secret_manifest),
        env=_kube_env(cfg),
        check=True,
        text=True,
        capture_output=True,
    )

    subprocess.run(
        [
            "kubectl",
            "-n",
            "kube-system",
            "delete",
            "job",
            "docs-agent-autoscale-ttl",
            "--ignore-not-found=true",
        ],
        env=_kube_env(cfg),
        check=False,
        capture_output=True,
        text=True,
    )

    scheduled_at = int(time.time())
    expires_at = scheduled_at + ttl_seconds
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "docs-agent-autoscale-ttl",
            "namespace": "kube-system",
            "labels": {"app": "docs-agent-autoscale-ttl"},
            "annotations": {
                "docs-agent.kubeflow.org/cloud": "gcp",
                "docs-agent.kubeflow.org/ttl-duration": duration_str,
                "docs-agent.kubeflow.org/ttl-seconds": str(ttl_seconds),
                "docs-agent.kubeflow.org/scheduled-at": str(scheduled_at),
                "docs-agent.kubeflow.org/expires-at": str(expires_at),
                "docs-agent.kubeflow.org/gpu-only": str(gpu_only).lower(),
            },
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {"labels": {"app": "docs-agent-autoscale-ttl"}},
                "spec": {
                    "restartPolicy": "OnFailure",
                    "nodeSelector": {"docs-agent-pool": "cpu"},
                    "containers": [
                        {
                            "name": "autoscale-timer",
                            "image": "mirror.gcr.io/library/python:3.12-slim",
                            "command": [
                                "python3",
                                "-c",
                                _build_in_cluster_gcp_autoscale_script(),
                            ],
                            "env": [
                                {"name": "TTL_SECONDS", "value": str(ttl_seconds)},
                                {"name": "GPU_ONLY", "value": str(gpu_only).lower()},
                            ],
                            "envFrom": [
                                {
                                    "secretRef": {
                                        "name": "docs-agent-autoscale-gcp",
                                    }
                                }
                            ],
                            "resources": {
                                "requests": {"cpu": "10m", "memory": "32Mi"},
                                "limits": {"cpu": "100m", "memory": "128Mi"},
                            },
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

    scope_str = "GPU node pool only (size=0)" if gpu_only else "ALL GKE node pools (GPU=0, CPU=0)"
    console.print(
        Panel(
            f"[bold green]✓ In-Cluster GKE Auto-Scale-Down Timer Armed![/bold green]\n"
            f"• [bold]Job:[/bold]          [cyan]kube-system/docs-agent-autoscale-ttl[/cyan]\n"
            f"• [bold]TTL Duration:[/bold] [cyan]{duration_str} ({ttl_seconds}s)[/cyan]\n"
            f"• [bold]Action:[/bold]       Scale {scope_str}\n"
            f"• [bold]Cancel anytime:[/bold] `docs-cli --cloud gcp ttl --cancel`",
            title="GKE Cluster Auto-Scale Timer",
        )
    )
    return ttl_seconds
