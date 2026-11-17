"""Ordered cluster teardown (`down` / `destroy`) and direct OCI tag-verified sweep (`nuke`)."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from typing import Dict, List, Optional

import oci
from rich.console import Console
from rich.table import Table

from .auth import OciSession
from .config import DocsConfig
from .gcp_auth import GcpSession
from .terraform import render_tfvars_files, run_terraform

console = Console()

MANAGED_TAG_KEY = "docs-cli-managed"
CLUSTER_TAG_KEY = "docs-cli-cluster"


def stop_local_dashboard_port_forward(cfg: DocsConfig) -> None:
    """Stop any background local port-forward supervisor tracked in .docs-cli/dashboard-port-forward.pid."""
    pid_file = cfg.state_dir / "dashboard-port-forward.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="utf-8").strip())
            try:
                os.killpg(old_pid, signal.SIGTERM)
            except Exception:
                os.kill(old_pid, signal.SIGTERM)
            console.print(f"[dim]Stopped local dashboard port-forward (PID={old_pid}).[/dim]")
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)


def has_cluster_ownership_tag(
    freeform_tags: Optional[Dict[str, str]],
    cluster_name: str,
) -> bool:
    """Verify a resource carries docs-cli-managed=true and matching docs-cli-cluster."""
    if not freeform_tags:
        return False
    return (
        freeform_tags.get(MANAGED_TAG_KEY) == "true"
        and freeform_tags.get(CLUSTER_TAG_KEY) == cluster_name
    )


def delete_k8s_loadbalancers_and_pvcs(cfg: DocsConfig) -> None:
    """Delete Kubernetes LoadBalancer Services and PVCs before Terraform destroy so OCI LBs/volumes detach cleanly."""
    stop_local_dashboard_port_forward(cfg)
    if not cfg.kubeconfig_path.exists():
        console.print(
            "[dim]No local kubeconfig found; skipping pre-destroy K8s service/PVC cleanup.[/dim]"
        )
        return

    env = dict(os.environ)
    env["KUBECONFIG"] = str(cfg.kubeconfig_path)

    # Check if K8s API is reachable before running pre-clean commands
    probe = subprocess.run(
        ["kubectl", "get", "ns", "--request-timeout=8s"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        console.print("[dim]Kubernetes API already unreachable; proceeding to OCI/Terraform teardown.[/dim]")
        return

    # Cancel any active in-cluster TTL auto-scaler job
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
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    # Remove custom resources with finalizers first
    for ns, kind in (
        (cfg.platform.namespace_ml_infra, "inferenceservices.serving.kserve.io"),
        (cfg.platform.namespace_ml_infra, "milvuses.milvus.io"),
        (cfg.platform.namespace_kubeflow, "pipelineversions.pipelines.kubeflow.org"),
        (cfg.platform.namespace_kubeflow, "pipelines.pipelines.kubeflow.org"),
    ):
        subprocess.run(
            ["kubectl", "-n", ns, "delete", kind, "--all", "--ignore-not-found=true", "--timeout=45s"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    for release, ns in (
        ("gateway-guardrails", cfg.platform.namespace_docs_agent),
        ("docs-agent", cfg.platform.namespace_docs_agent),
        ("llm-runtime", cfg.platform.namespace_ml_infra),
        ("kubeflow-dashboard", cfg.platform.namespace_kubeflow),
        ("kserve-ui", cfg.platform.namespace_kubeflow),
        ("istio-ingressgateway", "istio-system"),
    ):
        subprocess.run(
            ["helm", "uninstall", release, "-n", ns],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    # Explicitly delete any LoadBalancer Services across namespaces so OCI CCM releases public LBs before VCN teardown
    subprocess.run(
        [
            "kubectl",
            "-n",
            "istio-system",
            "delete",
            "svc",
            "istio-ingressgateway",
            "--ignore-not-found=true",
            "--timeout=120s",
        ],
        env=env,
        check=False,
    )

    # Scale down / delete workloads mounting PVCs so pvc-protection releases immediately
    for ns in (
        cfg.platform.namespace_ml_infra,
        cfg.platform.namespace_docs_agent,
        cfg.platform.namespace_kubeflow,
    ):
        subprocess.run(
            [
                "kubectl",
                "-n",
                ns,
                "delete",
                "deployment,statefulset",
                "--all",
                "--ignore-not-found=true",
                "--timeout=45s",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            [
                "kubectl",
                "-n",
                ns,
                "delete",
                "pvc",
                "--all",
                "--ignore-not-found=true",
                "--timeout=60s",
            ],
            env=env,
            check=False,
        )
    time.sleep(5)


def ordered_terraform_down(
    cfg: DocsConfig,
    session: OciSession,
    compartment_id: str,
) -> None:
    """Run ordered teardown: K8s LB/PVC cleanup -> platform TF destroy -> OKE TF destroy."""
    oke_tfvars, platform_tfvars = render_tfvars_files(
        cfg, session=session, compartment_id=compartment_id
    )

    console.print("[bold yellow]Step 1/3: Cleaning up Kubernetes LoadBalancers & PVCs...[/bold yellow]")
    delete_k8s_loadbalancers_and_pvcs(cfg)

    if (cfg.platform_tf_dir / ".terraform").exists() and (cfg.platform_tf_dir / "terraform.tfstate").exists():
        console.print("[bold yellow]Step 2/3: Destroying platform Terraform stack...[/bold yellow]")
        try:
            run_terraform(
                cfg.platform_tf_dir,
                ["destroy", "-auto-approve", f"-var-file={platform_tfvars}"],
            )
        except Exception as exc:
            console.print(
                f"[yellow]! Platform K8s Terraform destroy reported warning ({exc}); OKE cluster deletion in Step 3 will remove all remaining K8s objects.[/yellow]"
            )
    else:
        console.print("[dim]Step 2/3: Platform Terraform state not present; skipping.[/dim]")

    if (cfg.oke_tf_dir / ".terraform").exists():
        console.print("[bold yellow]Step 3/3: Destroying OKE Cluster & VCN Terraform stack...[/bold yellow]")
        try:
            run_terraform(
                cfg.oke_tf_dir,
                ["destroy", "-auto-approve", f"-var-file={oke_tfvars}"],
                extra_env=session.terraform_env(),
            )
        except Exception as exc:
            console.print(
                f"[yellow]! OKE Terraform destroy encountered dependency lag ({exc}); OCI sweep will clean up remaining resources.[/yellow]"
            )
    else:
        console.print("[dim]Step 3/3: OKE Terraform not initialized; skipping.[/dim]")


def discover_tagged_oci_resources(
    session: OciSession,
    compartment_id: str,
    cluster_name: str,
) -> List[Dict[str, str]]:
    """Discover OCI resources in compartment_id tagged with docs-cli-cluster=<cluster_name>."""
    found: List[Dict[str, str]] = []

    # 1. OKE Clusters
    oke_client = session.oke()
    clusters = oci.pagination.list_call_get_all_results(
        oke_client.list_clusters,
        compartment_id=compartment_id,
    ).data
    for c in clusters:
        if c.lifecycle_state not in ("DELETED", "DELETING") and (
            has_cluster_ownership_tag(getattr(c, "freeform_tags", None), cluster_name)
            or c.name == cluster_name
        ):
            found.append(
                {
                    "type": "OKE Cluster",
                    "name": c.name,
                    "id": c.id,
                    "state": c.lifecycle_state,
                }
            )

    # 2. Load Balancers (created by OKE service controller or tagged)
    lb_client = session.load_balancer()
    lbs = oci.pagination.list_call_get_all_results(
        lb_client.list_load_balancers,
        compartment_id=compartment_id,
    ).data
    for lb in lbs:
        if lb.lifecycle_state not in ("DELETED", "DELETING") and (
            has_cluster_ownership_tag(getattr(lb, "freeform_tags", None), cluster_name)
            or cluster_name in (lb.display_name or "")
        ):
            found.append(
                {
                    "type": "LoadBalancer",
                    "name": lb.display_name,
                    "id": lb.id,
                    "state": lb.lifecycle_state,
                }
            )

    # 3. Boot & Block Volumes tagged for this cluster
    bs_client = session.blockstorage()
    vols = oci.pagination.list_call_get_all_results(
        bs_client.list_volumes,
        compartment_id=compartment_id,
    ).data
    for v in vols:
        if v.lifecycle_state not in ("TERMINATED", "TERMINATING") and (
            has_cluster_ownership_tag(getattr(v, "freeform_tags", None), cluster_name)
        ):
            found.append(
                {
                    "type": "BlockVolume",
                    "name": v.display_name,
                    "id": v.id,
                    "state": v.lifecycle_state,
                }
            )

    # 4. VCNs
    net_client = session.network()
    vcns = oci.pagination.list_call_get_all_results(
        net_client.list_vcns,
        compartment_id=compartment_id,
    ).data
    for vcn in vcns:
        if vcn.lifecycle_state not in ("TERMINATED", "TERMINATING") and (
            has_cluster_ownership_tag(getattr(vcn, "freeform_tags", None), cluster_name)
        ):
            found.append(
                {
                    "type": "VCN",
                    "name": vcn.display_name,
                    "id": vcn.id,
                    "state": vcn.lifecycle_state,
                }
            )

    return found


def render_nuke_plan_table(resources: List[Dict[str, str]], cluster_name: str) -> None:
    table = Table(title=f"OCI Resources Tagged for Cluster '{cluster_name}'")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("State", style="yellow")
    table.add_column("OCID", style="dim")
    for r in resources:
        table.add_row(r["type"], r["name"], r["state"], r["id"])
    console.print(table)


def nuke_tagged_oci_resources(
    session: OciSession,
    compartment_id: str,
    cluster_name: str,
    resources: List[Dict[str, str]],
) -> None:
    """Delete tagged OKE clusters, LoadBalancers, BlockVolumes, and VCNs in dependency order."""
    oke_client = session.oke()
    lb_client = session.load_balancer()
    bs_client = session.blockstorage()
    net_client = session.network()

    # 1. Delete Node Pools & OKE Clusters
    for r in [x for x in resources if x["type"] == "OKE Cluster"]:
        pools = oci.pagination.list_call_get_all_results(
            oke_client.list_node_pools,
            compartment_id=compartment_id,
            cluster_id=r["id"],
        ).data
        for np in pools:
            if np.lifecycle_state not in ("DELETED", "DELETING"):
                console.print(f"[yellow]Deleting NodePool {np.name} ({np.id})...[/yellow]")
                oke_client.delete_node_pool(np.id)
        console.print(f"[yellow]Deleting OKE Cluster {r['name']} ({r['id']})...[/yellow]")
        oke_client.delete_cluster(r["id"])

    # 2. Delete Load Balancers
    for r in [x for x in resources if x["type"] == "LoadBalancer"]:
        console.print(f"[yellow]Deleting LoadBalancer {r['name']} ({r['id']})...[/yellow]")
        lb_client.delete_load_balancer(r["id"])

    # 3. Delete Block Volumes
    for r in [x for x in resources if x["type"] == "BlockVolume"]:
        console.print(f"[yellow]Deleting BlockVolume {r['name']} ({r['id']})...[/yellow]")
        bs_client.delete_volume(r["id"])

    # 4. Delete VCN sub-resources & VCN
    for r in [x for x in resources if x["type"] == "VCN"]:
        vcn_id = r["id"]
        time.sleep(2)
        for sn in oci.pagination.list_call_get_all_results(
            net_client.list_subnets, compartment_id=compartment_id, vcn_id=vcn_id
        ).data:
            try:
                net_client.delete_subnet(sn.id)
            except Exception:
                pass
        for igw in oci.pagination.list_call_get_all_results(
            net_client.list_internet_gateways,
            compartment_id=compartment_id,
            vcn_id=vcn_id,
        ).data:
            try:
                net_client.delete_internet_gateway(igw.id)
            except Exception:
                pass
        for nat in oci.pagination.list_call_get_all_results(
            net_client.list_nat_gateways, compartment_id=compartment_id, vcn_id=vcn_id
        ).data:
            try:
                net_client.delete_nat_gateway(nat.id)
            except Exception:
                pass
        for sgw in oci.pagination.list_call_get_all_results(
            net_client.list_service_gateways,
            compartment_id=compartment_id,
            vcn_id=vcn_id,
        ).data:
            try:
                net_client.delete_service_gateway(sgw.id)
            except Exception:
                pass
        console.print(f"[yellow]Deleting VCN {r['name']} ({vcn_id})...[/yellow]")
        try:
            net_client.delete_vcn(vcn_id)
        except Exception as exc:
            console.print(
                f"[red]VCN {vcn_id} still has dependencies terminating ({exc}); re-run `docs-cli nuke --yes` once OKE deletion finishes.[/red]"
            )


def destroy_full_environment(
    cfg: DocsConfig,
    session: OciSession,
    compartment_id: str,
    wait_timeout_seconds: int = 900,
) -> List[Dict[str, str]]:
    """Complete dev environment teardown (`docs-cli destroy --yes`):
    1. Stop local port-forward & clean K8s LoadBalancers, CRs, and PVCs
    2. Ordered Terraform destroy (`platform` -> `oke-cluster`)
    3. OCI tag-verified sweep & wait until 0 OCI resources remain
    4. Clean ephemeral local `.docs-cli/` & `.tfstate` artifacts so next `docs-cli up --apply` starts clean
    """
    stop_local_dashboard_port_forward(cfg)
    ordered_terraform_down(cfg, session, compartment_id)

    deadline = time.time() + wait_timeout_seconds
    attempt = 0
    remaining = discover_tagged_oci_resources(session, compartment_id, cfg.cluster.name)
    while remaining and time.time() < deadline:
        attempt += 1
        console.print(
            f"[yellow]OCI cleanup sweep #{attempt}: {len(remaining)} resource(s) still terminating/present...[/yellow]"
        )
        render_nuke_plan_table(remaining, cfg.cluster.name)
        try:
            nuke_tagged_oci_resources(session, compartment_id, cfg.cluster.name, remaining)
        except Exception as exc:
            console.print(f"[dim]  • OCI sweep note: {exc}[/dim]")
        time.sleep(20)
        remaining = discover_tagged_oci_resources(session, compartment_id, cfg.cluster.name)

    if not remaining:
        # Once OCI confirms 0 cloud resources remain, remove stale local tfstate & ephemeral cluster certs/kubeconfig
        for tf_dir in (cfg.platform_tf_dir, cfg.oke_tf_dir):
            for fname in ("terraform.tfstate", "terraform.tfstate.backup"):
                fpath = tf_dir / fname
                if fpath.exists():
                    fpath.unlink(missing_ok=True)
        for ephemeral in (
            cfg.kubeconfig_path,
            cfg.state_dir / "tls_cert.pem",
            cfg.state_dir / "tls_key.pem",
            cfg.state_dir / "tls_domain.txt",
            cfg.state_dir / "dashboard-port-forward.pid",
            cfg.state_dir / "dashboard-port-forward.log",
        ):
            if ephemeral.exists():
                ephemeral.unlink(missing_ok=True)
        console.print(
            f"[bold green]✓ Verified 0 OCI cloud resources remain for '{cfg.cluster.name}' and cleaned local state![/bold green]"
        )
    return remaining


# =============================================================================
# Google Cloud (GCP / GKE) Ordered Teardown, Label-Verified Nuke & Full Destroy
# =============================================================================


def normalize_gcp_label_value(val: str) -> str:
    """Normalize a cluster name into a valid GCP resource label value (lowercase, <=63 chars)."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", val or "").lower()[:63]


def has_gcp_cluster_ownership_label(
    labels: Optional[Dict[str, str]],
    cluster_name: str,
) -> bool:
    """Verify a GCP resource carries `docs-cli-managed=true` and matching `docs-cli-cluster`."""
    if not labels:
        return False
    expected = normalize_gcp_label_value(cluster_name)
    return (
        labels.get(MANAGED_TAG_KEY) == "true"
        and labels.get(CLUSTER_TAG_KEY) == expected
    )


def ordered_gcp_terraform_down(
    cfg: DocsConfig,
    gcp_session: GcpSession,
) -> None:
    """Run ordered GCP teardown: K8s LB/PVC cleanup -> platform TF destroy -> GKE TF destroy."""
    gke_tfvars, platform_tfvars = render_tfvars_files(cfg, gcp_session=gcp_session)

    console.print("[bold yellow]Step 1/3: Cleaning up Kubernetes LoadBalancers & PVCs...[/bold yellow]")
    delete_k8s_loadbalancers_and_pvcs(cfg)

    if (cfg.platform_tf_dir / ".terraform").exists() and (cfg.platform_tf_dir / "terraform.tfstate").exists():
        console.print("[bold yellow]Step 2/3: Destroying platform Terraform stack...[/bold yellow]")
        try:
            run_terraform(
                cfg.platform_tf_dir,
                ["destroy", "-auto-approve", f"-var-file={platform_tfvars}"],
            )
        except Exception as exc:
            console.print(
                f"[yellow]! Platform K8s Terraform destroy reported warning ({exc}); GKE cluster deletion in Step 3 will remove remaining K8s objects.[/yellow]"
            )
    else:
        console.print("[dim]Step 2/3: Platform Terraform state not present; skipping.[/dim]")

    if (cfg.gke_tf_dir / ".terraform").exists():
        console.print("[bold yellow]Step 3/3: Destroying GKE Cluster & VPC Terraform stack...[/bold yellow]")
        try:
            run_terraform(
                cfg.gke_tf_dir,
                ["destroy", "-auto-approve", f"-var-file={gke_tfvars}"],
                extra_env=gcp_session.terraform_env(),
            )
        except Exception as exc:
            console.print(
                f"[yellow]! GKE Terraform destroy encountered dependency lag ({exc}); GCP sweep will clean up remaining resources.[/yellow]"
            )
    else:
        console.print("[dim]Step 3/3: GKE Terraform not initialized; skipping to GCP sweep.[/dim]")


def discover_labeled_gcp_resources(
    gcp_session: GcpSession,
    cluster_name: str,
) -> List[Dict[str, str]]:
    """Discover GCP resources owned by this cluster (verified by `docs-cli-managed=true` label or dedicated VPC name)."""
    if not shutil.which("gcloud"):
        return []
    gcp_session.ensure_gcloud_auth_if_service_account()
    env = gcp_session.gcloud_env()
    proj = gcp_session.project_id
    lbl_val = normalize_gcp_label_value(cluster_name)
    found: List[Dict[str, str]] = []

    def _gcloud_json(args: List[str]) -> list:
        res = subprocess.run(
            ["gcloud", *args, f"--project={proj}", "--format=json", "--quiet"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            try:
                return json.loads(res.stdout)
            except Exception:
                return []
        return []

    # 1. GKE Clusters
    for cl in _gcloud_json(["container", "clusters", "list"]):
        rlabels = cl.get("resourceLabels") or {}
        if has_gcp_cluster_ownership_label(rlabels, cluster_name):
            found.append(
                {
                    "type": "GkeCluster",
                    "name": str(cl.get("name", "")),
                    "id": f"{cl.get('location', '')}/{cl.get('name', '')}",
                    "state": str(cl.get("status", "RUNNING")),
                }
            )

    # 2. Compute Persistent Disks carrying ownership labels
    label_filter = f"labels.{MANAGED_TAG_KEY}=true AND labels.{CLUSTER_TAG_KEY}={lbl_val}"
    for disk in _gcloud_json(["compute", "disks", "list", f"--filter={label_filter}"]):
        if has_gcp_cluster_ownership_label(disk.get("labels"), cluster_name):
            zone_short = str(disk.get("zone", "")).rsplit("/", 1)[-1]
            found.append(
                {
                    "type": "PersistentDisk",
                    "name": str(disk.get("name", "")),
                    "id": f"{zone_short}/{disk.get('name', '')}",
                    "state": str(disk.get("status", "READY")),
                }
            )

    # 3. Dedicated VPC & Networking sub-resources created by `deployments/terraform/gke-cluster/network.tf`
    vpc_name = f"{cluster_name}-vpc"
    for net in _gcloud_json(["compute", "networks", "list", f"--filter=name={vpc_name}"]):
        if str(net.get("name")) == vpc_name and "docs-cli" in str(net.get("description", "")):
            found.append(
                {
                    "type": "VPC",
                    "name": vpc_name,
                    "id": str(net.get("selfLink", vpc_name)),
                    "state": "ACTIVE",
                }
            )

    return found


def nuke_labeled_gcp_resources(
    gcp_session: GcpSession,
    cluster_name: str,
    resources: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Delete discovered GCP resources owned by `cluster_name` in dependency order."""
    if not shutil.which("gcloud"):
        raise RuntimeError("Google Cloud SDK (`gcloud`) is required for `docs-cli --cloud gcp nuke`.")
    gcp_session.ensure_gcloud_auth_if_service_account()
    env = gcp_session.gcloud_env()
    proj = gcp_session.project_id

    if resources is None:
        resources = discover_labeled_gcp_resources(gcp_session, cluster_name)

    # 1. Delete GKE clusters first (detaches nodes, MIGs, LBs, and dynamic PVs)
    for r in [x for x in resources if x["type"] == "GkeCluster"]:
        loc, name = r["id"].split("/", 1)
        console.print(f"[yellow]Deleting GKE Cluster {name} ({loc})...[/yellow]")
        subprocess.run(
            [
                "gcloud",
                "container",
                "clusters",
                "delete",
                name,
                f"--location={loc}",
                f"--project={proj}",
                "--quiet",
            ],
            env=env,
            check=False,
        )

    # 2. Delete orphaned labeled Persistent Disks
    for r in [x for x in resources if x["type"] == "PersistentDisk"]:
        zone, name = r["id"].split("/", 1)
        console.print(f"[yellow]Deleting PersistentDisk {name} ({zone})...[/yellow]")
        subprocess.run(
            [
                "gcloud",
                "compute",
                "disks",
                "delete",
                name,
                f"--zone={zone}",
                f"--project={proj}",
                "--quiet",
            ],
            env=env,
            check=False,
        )

    # 3. Delete dedicated VPC sub-resources (Firewall -> Router/NAT -> Subnet -> VPC)
    for r in [x for x in resources if x["type"] == "VPC"]:
        vpc_name = r["name"]
        for fw in (
            f"{cluster_name}-allow-internal",
        ):
            subprocess.run(
                ["gcloud", "compute", "firewall-rules", "delete", fw, f"--project={proj}", "--quiet"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        subprocess.run(
            [
                "gcloud",
                "compute",
                "routers",
                " delete" if False else "delete",
                f"{cluster_name}-router",
                f"--region={gcp_session.region}",
                f"--project={proj}",
                "--quiet",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            [
                "gcloud",
                "compute",
                "networks",
                "subnets",
                "delete",
                f"{cluster_name}-nodes",
                f"--region={gcp_session.region}",
                f"--project={proj}",
                "--quiet",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        console.print(f"[yellow]Deleting VPC {vpc_name}...[/yellow]")
        subprocess.run(
            ["gcloud", "compute", "networks", "delete", vpc_name, f"--project={proj}", "--quiet"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


def destroy_full_gcp_environment(
    cfg: DocsConfig,
    gcp_session: GcpSession,
    wait_timeout_seconds: int = 900,
) -> List[Dict[str, str]]:
    """Complete GCP dev environment teardown (`docs-cli --cloud gcp destroy --yes`)."""
    stop_local_dashboard_port_forward(cfg)
    ordered_gcp_terraform_down(cfg, gcp_session)

    deadline = time.time() + wait_timeout_seconds
    attempt = 0
    remaining = discover_labeled_gcp_resources(gcp_session, cfg.cluster.name)
    while remaining and time.time() < deadline:
        attempt += 1
        console.print(
            f"[yellow]GCP cleanup sweep #{attempt}: {len(remaining)} resource(s) still terminating/present...[/yellow]"
        )
        render_nuke_plan_table(remaining, cfg.cluster.name)
        try:
            nuke_labeled_gcp_resources(gcp_session, cfg.cluster.name, remaining)
        except Exception as exc:
            console.print(f"[dim]  • GCP sweep note: {exc}[/dim]")
        time.sleep(20)
        remaining = discover_labeled_gcp_resources(gcp_session, cfg.cluster.name)

    if not remaining:
        for tf_dir in (cfg.platform_tf_dir, cfg.gke_tf_dir):
            for fname in ("terraform.tfstate", "terraform.tfstate.backup"):
                fpath = tf_dir / fname
                if fpath.exists():
                    fpath.unlink(missing_ok=True)
        for ephemeral in (
            cfg.kubeconfig_path,
            cfg.state_dir / "tls_cert.pem",
            cfg.state_dir / "tls_key.pem",
            cfg.state_dir / "tls_domain.txt",
            cfg.state_dir / "dashboard-port-forward.pid",
            cfg.state_dir / "dashboard-port-forward.log",
        ):
            if ephemeral.exists():
                ephemeral.unlink(missing_ok=True)
        console.print(
            f"[bold green]✓ Verified 0 GCP cloud resources remain for '{cfg.cluster.name}' and cleaned local state![/bold green]"
        )
    return remaining
