"""Click CLI entrypoint for `docs-cli` / `docsctl`."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .auth import OciSession
from .config import DocsConfig
from .kube import (
    compile_kfp_pipelines,
    deploy_helm_stack,
    expose_kubeflow_dashboard_port_forward,
    get_ingress_lb_ip,
    render_helm_values_files,
    run_kfp_and_bootstrap_rag,
    sync_all_secrets,
    validate_rendered_chart_contracts,
    wait_for_worker_nodes_ready,
    wrap_compiled_pipeline_as_k8s_native_crs,
)
from .scale import (
    cancel_in_cluster_ttl_job,
    get_in_cluster_ttl_status,
    scale_node_pools,
    schedule_in_cluster_ttl_job,
)
from .netlify import (
    build_agent_endpoint_url,
    build_hugo_site,
    deploy_directory_to_netlify_api,
    inject_hugo_docsy_hooks,
    prepare_demo_bundle,
)
from .nuke import (
    destroy_full_environment,
    discover_tagged_oci_resources,
    nuke_tagged_oci_resources,
    ordered_terraform_down,
    render_nuke_plan_table,
)
from .terraform import (
    ensure_kubeflow_manifests_charts,
    fetch_oke_kubeconfig,
    get_terraform_outputs,
    render_tfvars_files,
    run_terraform,
)
from .verify import print_verification_report, run_verification_suite

console = Console()


def _load_cfg(
    config_path: Optional[str],
    no_gpu: bool = False,
) -> DocsConfig:
    p = Path(config_path) if config_path else None
    gpu_override = False if no_gpu else None
    return DocsConfig.load(path=p, enable_gpu_override=gpu_override)


@click.group()
@click.option(
    "-c",
    "--config",
    "config_path",
    default=None,
    help="Path to docs.toml configuration file.",
)
@click.pass_context
def main(ctx: click.Context, config_path: Optional[str]) -> None:
    """docs-cli (docsctl): OCI OKE + Netlify Dev Cluster CLI for Kubeflow Docs Agent."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@main.command()
@click.option("--no-gpu", is_flag=True, help="Render templates for a CPU-only cluster (external LLM API).")
@click.option("--domain", default=None, help="Override ingress domain when rendering Helm values.")
@click.pass_context
def template(ctx: click.Context, no_gpu: bool, domain: Optional[str]) -> None:
    """Render Terraform (.tfvars.json) and Helm values files into .docs-cli/ without touching cloud resources."""
    cfg = _load_cfg(ctx.obj.get("config_path"), no_gpu=no_gpu)
    if domain:
        cfg.ingress.domain = domain

    session: Optional[OciSession] = None
    comp_id: Optional[str] = cfg.oci.compartment_id or None
    try:
        session = OciSession.from_docs_config(cfg)
    except Exception:
        pass

    oke_tfvars, platform_tfvars = render_tfvars_files(
        cfg, session=session, compartment_id=comp_id
    )
    helm_files = render_helm_values_files(cfg, domain=domain)

    table = Table(title="Rendered Configuration Templates (.docs-cli/)")
    table.add_column("Component", style="cyan")
    table.add_column("File Path", style="green")
    table.add_row("OKE Cluster Terraform", str(oke_tfvars))
    table.add_row("Platform Stack Terraform", str(platform_tfvars))
    for k, v in helm_files.items():
        table.add_row(f"Helm Values ({k})", str(v))
    console.print(table)
    console.print(
        f"[dim]GPU Node Pool Enabled: {cfg.cluster.enable_gpu_pool} | "
        f"Active LLM Profile: {cfg.llm.active_profile if cfg.cluster.enable_gpu_pool else cfg.llm.external_model}[/dim]"
    )


@main.command()
@click.option("--no-gpu", is_flag=True, help="Provision a CPU-only OKE cluster and use an external OpenAI-compatible API.")
@click.option("--skip-oke", is_flag=True, help="Skip OKE cluster Terraform (use existing cluster/kubeconfig).")
@click.option("--skip-platform", is_flag=True, help="Skip platform Terraform (Istio, Knative, KServe, Milvus, KFP).")
@click.option("--skip-pods", is_flag=True, help="Skip Helm chart deployment (llm-runtime, docs-agent, gateway-guardrails).")
@click.option("--with-netlify", is_flag=True, help="Also build and deploy the frontend/website to Netlify.")
@click.option("--ttl", default=None, help="Auto-scale node pools down to 0 after duration (e.g. '2h', '90m', '45m').")
@click.option("--ttl-gpu-only", is_flag=True, help="When --ttl expires, scale only the GPU node pool to 0 (keep CPU nodes up).")
@click.option("--apply", is_flag=True, help="Execute changes (default is dry-run / plan).")
@click.pass_context
def up(
    ctx: click.Context,
    no_gpu: bool,
    skip_oke: bool,
    skip_platform: bool,
    skip_pods: bool,
    with_netlify: bool,
    ttl: Optional[str],
    ttl_gpu_only: bool,
    apply: bool,
) -> None:
    """Spin up the full Kubeflow Docs Agent dev stack on OCI OKE (including KFP ingestion & optional --ttl auto-scale-down)."""
    cfg = _load_cfg(ctx.obj.get("config_path"), no_gpu=no_gpu)
    mode_label = "APPLY" if apply else "DRY-RUN (pass --apply to execute)"
    console.print(
        Panel(
            f"[bold cyan]Cluster:[/bold cyan] {cfg.cluster.name} ({cfg.oci.region})\n"
            f"[bold cyan]Topology:[/bold cyan] CPU ({cfg.cluster.cpu_node_count}x {cfg.cluster.cpu_node_shape}) "
            + (
                f"+ GPU ({cfg.cluster.gpu_node_count}x {cfg.cluster.gpu_node_shape}, profile={cfg.llm.active_profile})"
                if cfg.cluster.enable_gpu_pool
                else f"[yellow]CPU-Only (--no-gpu, external LLM={cfg.llm.external_model})[/yellow]"
            )
            + f"\n[bold cyan]Mode:[/bold cyan] {mode_label}",
            title="Kubeflow Docs Agent — `docs-cli up`",
        )
    )

    session: Optional[OciSession] = None
    comp_id: Optional[str] = cfg.oci.compartment_id or None
    if not skip_oke or apply:
        try:
            session = OciSession.from_docs_config(cfg)
            comp_id = session.resolve_compartment_id(cfg)
        except Exception as exc:
            if apply and not skip_oke:
                raise click.ClickException(f"OCI authentication failed: {exc}")
            console.print(f"[yellow]! OCI credentials not active ({exc}); templating with placeholders.[/yellow]")

    oke_tfvars, platform_tfvars = render_tfvars_files(
        cfg, session=session, compartment_id=comp_id
    )
    render_helm_values_files(cfg)

    if not apply:
        console.print(
            "[green]✓ Rendered Terraform & Helm configurations into .docs-cli/[/green]\n"
            "Re-run with [bold]docs-cli up --apply[/bold] to provision the cluster and deploy pods."
        )
        return

    # Phase 1: OKE Cluster Terraform
    if not skip_oke:
        assert session is not None
        console.print("[bold]Phase 1/4: Provisioning OCI OKE Cluster & VCN...[/bold]")
        if not (cfg.oke_tf_dir / ".terraform").exists():
            run_terraform(cfg.oke_tf_dir, ["init"], extra_env=session.terraform_env())
        run_terraform(
            cfg.oke_tf_dir,
            ["apply", "-auto-approve", f"-var-file={oke_tfvars}"],
            extra_env=session.terraform_env(),
        )
        outputs = get_terraform_outputs(cfg.oke_tf_dir)
        cluster_id = outputs.get("cluster_id")
        if not cluster_id:
            raise click.ClickException("Could not read cluster_id from OKE Terraform outputs.")
        fetch_oke_kubeconfig(cfg, session, cluster_id)
        wait_for_worker_nodes_ready(cfg)

    # Phase 2: Platform Stack Terraform (Istio, Knative, KServe, Milvus, KFP, TEI Embeddings & Reranker)
    if not skip_platform:
        if cfg.platform.use_kubeflow_helm_charts:
            ensure_kubeflow_manifests_charts(cfg)
        console.print("[bold]Phase 2/4: Deploying Platform Stack (Istio, Knative, KServe, Milvus, KFP, TEI)...[/bold]")
        if not (cfg.platform_tf_dir / ".terraform").exists():
            run_terraform(cfg.platform_tf_dir, ["init"])
        run_terraform(
            cfg.platform_tf_dir,
            ["apply", "-auto-approve", f"-var-file={platform_tfvars}"],
        )

    # Phase 3: Secrets & Helm Charts (llm-runtime, docs-agent, gateway-guardrails)
    domain = cfg.resolve_agent_domain()
    if not skip_pods:
        console.print("[bold]Phase 3/4: Syncing Secrets & Deploying Helm Releases...[/bold]")
        domain = deploy_helm_stack(cfg)

    # Phase 4: Optional Netlify deployment
    exposed_page_url = cfg.netlify.site_url or f"https://{domain}"
    if with_netlify:
        console.print("[bold]Phase 4/5: Deploying Website/Frontend to Netlify...[/bold]")
        deployed_url = ctx.invoke(netlify, mode=cfg.netlify.mode, domain=domain, deploy=True)
        if deployed_url:
            exposed_page_url = deployed_url

    # Optional TTL auto-scaler
    if ttl and session and comp_id:
        schedule_in_cluster_ttl_job(
            cfg,
            session=session,
            compartment_id=comp_id,
            duration_str=ttl,
            gpu_only=ttl_gpu_only,
        )

    # Phase 5: Ordered Verification & Exposed Page Report
    console.print("[bold]Phase 5/5: Running Ordered Verification Suite...[/bold]")
    results, all_passed, agent_url, final_page_url = run_verification_suite(
        cfg,
        domain=domain,
        exposed_page_url=exposed_page_url,
    )
    print_verification_report(results, all_passed, agent_url, final_page_url)


@main.command()
@click.option("--cluster-id", default=None, help="Explicit OKE Cluster OCID (auto-detected from Terraform outputs if omitted).")
@click.pass_context
def kubeconfig(ctx: click.Context, cluster_id: Optional[str]) -> None:
    """Fetch OKE kubeconfig into .docs-cli/kubeconfig."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    session = OciSession.from_docs_config(cfg)
    cid = cluster_id or get_terraform_outputs(cfg.oke_tf_dir).get("cluster_id")
    if not cid:
        raise click.ClickException(
            "No cluster_id found in Terraform outputs. Pass --cluster-id <ocid>."
        )
    fetch_oke_kubeconfig(cfg, session, cid)


@main.command()
@click.option("--domain", default=None, help="Domain name for self-signed TLS secret generation.")
@click.pass_context
def secrets(ctx: click.Context, domain: Optional[str]) -> None:
    """Create or update all Kubernetes Secrets from secure environment variables."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    sync_all_secrets(cfg, domain=domain)


@main.command(name="pods")
@click.option("--no-gpu", is_flag=True, help="Deploy without in-cluster GPU vLLM (use external LLM API).")
@click.option("--skip-kagent", is_flag=True, help="Skip installing/upgrading Kagent CRDs and controller.")
@click.pass_context
def pods_cmd(ctx: click.Context, no_gpu: bool, skip_kagent: bool) -> None:
    """Deploy/upgrade the application pods & Helm charts (llm-runtime, docs-agent, gateway-guardrails)."""
    cfg = _load_cfg(ctx.obj.get("config_path"), no_gpu=no_gpu)
    domain = deploy_helm_stack(cfg, skip_kagent=skip_kagent)
    console.print(
        f"[green]✓ Application pods & ingress configured for https://{domain}[/green]"
    )


@main.command()
@click.option(
    "--pipeline",
    "pipelines",
    multiple=True,
    type=click.Choice(["docs", "issues", "code"]),
    default=("docs", "issues", "code"),
    help="Which KFP v4 ingestion pipelines to compile.",
)
@click.option(
    "--k8s-native/--no-k8s-native",
    default=True,
    help="Also emit KFP Kubernetes-native Pipeline & PipelineVersion CRDs (pipelines.kubeflow.org/v2beta1).",
)
@click.option(
    "--apply",
    is_flag=True,
    help="Apply the generated Pipeline & PipelineVersion CRDs directly to the cluster via `kubectl apply`.",
)
@click.option(
    "--run/--no-run",
    "run_job",
    default=True,
    help="When --apply is set, also trigger KFP v2beta1 pipeline runs and bootstrap Milvus RAG collections.",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    pipelines: tuple[str, ...],
    k8s_native: bool,
    apply: bool,
    run_job: bool,
) -> None:
    """Compile KFP v4 RAG ingestion pipelines, apply Kubernetes-native Pipeline CRDs, and trigger ingestion runs."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    compiled = compile_kfp_pipelines(cfg, selected=pipelines)
    for path in compiled:
        console.print(f"[green]✓ Compiled KFP pipeline YAML:[/green] {path}")
        if k8s_native:
            cr_file = wrap_compiled_pipeline_as_k8s_native_crs(
                path,
                namespace=cfg.platform.namespace_kubeflow,
            )
            console.print(
                f"[green]✓ Generated K8s-native Pipeline + PipelineVersion CRDs:[/green] {cr_file}"
            )
            if apply:
                env = dict(os.environ)
                if cfg.kubeconfig_path.exists():
                    env["KUBECONFIG"] = str(cfg.kubeconfig_path)
                res_apply = subprocess.run(
                    ["kubectl", "apply", "-f", str(cr_file)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res_apply.returncode != 0 and "immutable" in res_apply.stderr:
                    subprocess.run(
                        ["kubectl", "replace", "--force", "-f", str(cr_file)],
                        env=env,
                        check=True,
                    )
                elif res_apply.returncode != 0:
                    raise click.ClickException(res_apply.stderr.strip())
                console.print(
                    f"[bold green]✓ Applied K8s-native Pipeline CRDs to namespace '{cfg.platform.namespace_kubeflow}'[/bold green]"
                )
    if apply and run_job:
        run_kfp_and_bootstrap_rag(cfg)


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["website", "demo"]),
    default=None,
    help="'website' injects Docsy hooks into kubeflow/website; 'demo' bundles standalone frontend/.",
)
@click.option("--domain", default=None, help="Docs Agent ingress domain (auto-detected from LB IP if omitted).")
@click.option("--website-path", default=None, help="Path to local kubeflow/website checkout.")
@click.option("--build-hugo", is_flag=True, help="Run `hugo --gc --minify` after injecting hooks in website mode.")
@click.option("--deploy", is_flag=True, help="Upload the built site to Netlify using NETLIFY_AUTH_TOKEN and NETLIFY_SITE_ID.")
@click.pass_context
def netlify(
    ctx: click.Context,
    mode: Optional[str],
    domain: Optional[str],
    website_path: Optional[str],
    build_hugo: bool,
    deploy: bool,
) -> None:
    """Wire kubeflow/website (Hugo + Docsy) or standalone demo frontend to the cluster & deploy to Netlify."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    active_mode = mode or cfg.netlify.mode
    lb_ip = None if domain else get_ingress_lb_ip(cfg, timeout_seconds=5)
    resolved_domain = domain or cfg.resolve_agent_domain(lb_ip=lb_ip)
    agent_url = build_agent_endpoint_url(resolved_domain)

    if active_mode == "website":
        site_dir = Path(website_path) if website_path else None
        result = inject_hugo_docsy_hooks(cfg, resolved_domain, website_dir=site_dir)
        console.print(
            Panel(
                f"[bold green]✓ Injected Docs Agent into kubeflow/website![/bold green]\n"
                f"• Website Repo: [cyan]{result['website_dir']}[/cyan]\n"
                f"• Stylesheet:   [cyan]{result['css']}[/cyan]\n"
                f"• Script:       [cyan]{result['js']}[/cyan]\n"
                f"• Head Hook:    [cyan]{result['head_hook']}[/cyan]\n"
                f"• Body Hook:    [cyan]{result['body_hook']}[/cyan]\n"
                f"• Agent URL:    [cyan]{agent_url}[/cyan]",
                title="Netlify / Hugo + Docsy Integration",
            )
        )
        publish_dir = result["website_dir"] / "public"
        if build_hugo or deploy:
            publish_dir = build_hugo_site(
                result["website_dir"],
                base_url=cfg.netlify.site_url or None,
            )
    else:
        publish_dir = prepare_demo_bundle(cfg, resolved_domain)
        console.print(
            Panel(
                f"[bold green]✓ Built Standalone Demo Bundle![/bold green]\n"
                f"• Bundle Dir: [cyan]{publish_dir}[/cyan]\n"
                f"• Agent URL:  [cyan]{agent_url}[/cyan]",
                title="Netlify Standalone Demo Bundle",
            )
        )

    if deploy:
        resp = deploy_directory_to_netlify_api(
            publish_dir=publish_dir,
            site_id=cfg.netlify.site_id,
            auth_token=cfg.netlify.auth_token,
        )
        deploy_url = resp.get("deploy_ssl_url") or resp.get("ssl_url") or resp.get("url") or ""
        console.print(f"[bold green]✓ Deployed to Netlify:[/bold green] {deploy_url}")
        return deploy_url
    return cfg.netlify.site_url or ""


@main.command()
@click.option("--domain", default=None, help="Explicit Docs Agent ingress domain.")
@click.option("--page-url", default=None, help="Explicit exposed page URL to test.")
@click.option("--helm-tests", is_flag=True, help="Also execute in-cluster `helm test` suites.")
@click.pass_context
def verify(
    ctx: click.Context,
    domain: Optional[str],
    page_url: Optional[str],
    helm_tests: bool,
) -> None:
    """Run the ordered 6-stage verification suite and print passed tests + exposed page link."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    results, all_passed, agent_url, final_page_url = run_verification_suite(
        cfg,
        domain=domain,
        exposed_page_url=page_url,
        run_helm_tests=helm_tests,
    )
    print_verification_report(results, all_passed, agent_url, final_page_url)
    if not all_passed:
        raise SystemExit(1)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current cluster pods, InferenceServices, Kagent Agents, and Ingress IP."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    if not cfg.kubeconfig_path.exists():
        console.print(
            f"[yellow]No kubeconfig found at {cfg.kubeconfig_path}. Run `docs-cli kubeconfig` or `docs-cli up --apply`.[/yellow]"
        )
        return

    env = dict(os.environ)
    env["KUBECONFIG"] = str(cfg.kubeconfig_path)

    for label, cmd in (
        ("Nodes", ["kubectl", "get", "nodes", "-o", "wide"]),
        (
            "ML Infra Pods & InferenceServices",
            ["kubectl", "-n", cfg.platform.namespace_ml_infra, "get", "pods,inferenceservices"],
        ),
        (
            "Docs Agent Pods & Kagent Agents",
            ["kubectl", "-n", cfg.platform.namespace_docs_agent, "get", "pods,agents,modelconfigs"],
        ),
        (
            "Istio Ingress Gateway",
            ["kubectl", "-n", "istio-system", "get", "svc,gateway,virtualservice"],
        ),
    ):
        console.print(f"\n[bold cyan]=== {label} ===[/bold cyan]")
        subprocess.run(cmd, env=env, check=False)


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Verify required CLI tools, local repository paths, and environment variables."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    table = Table(title="docs-cli Environment & Prerequisite Check")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for binary in ("terraform", "kubectl", "helm", "oci", "hugo"):
        found = shutil.which(binary)
        table.add_row(
            f"Binary: {binary}",
            "[green]OK[/green]" if found else "[yellow]MISSING[/yellow]",
            found or "Not in PATH",
        )

    table.add_row(
        "OKE Terraform Module",
        "[green]OK[/green]" if cfg.oke_tf_dir.exists() else "[red]MISSING[/red]",
        str(cfg.oke_tf_dir),
    )
    table.add_row(
        "Platform Terraform Module",
        "[green]OK[/green]" if cfg.platform_tf_dir.exists() else "[red]MISSING[/red]",
        str(cfg.platform_tf_dir),
    )
    table.add_row(
        "Helm Charts Directory",
        "[green]OK[/green]" if cfg.charts_dir.exists() else "[red]MISSING[/red]",
        str(cfg.charts_dir),
    )

    website_dir = (cfg.repo_root / cfg.netlify.website_repo_path).resolve()
    table.add_row(
        "kubeflow/website Repo",
        "[green]OK[/green]" if website_dir.exists() else "[yellow]NOT FOUND[/yellow]",
        str(website_dir),
    )

    has_oci = bool(cfg.oci.tenancy_ocid or (Path.home() / ".oci" / "config").exists())
    table.add_row(
        "OCI Credentials",
        "[green]CONFIGURED[/green]" if has_oci else "[yellow]UNSET[/yellow]",
        "OCI_TENANCY env or ~/.oci/config",
    )
    table.add_row(
        "Netlify Token",
        "[green]CONFIGURED[/green]" if cfg.netlify.auth_token else "[yellow]UNSET[/yellow]",
        "NETLIFY_AUTH_TOKEN",
    )
    console.print(table)


@main.command(name="scale-down")
@click.option("--gpu-only", is_flag=True, help="Scale only the GPU node pool down to 0 (keep CPU nodes running).")
@click.option("--after", default=None, help="Schedule an in-cluster TTL job to scale down after duration (e.g. '2h', '90m').")
@click.option(
    "--release-frontend/--keep-frontend",
    default=True,
    help="Release the demo frontend (scale docs-agent-demo-web to 0 and disable its VirtualService route) during scale-down.",
)
@click.option("--yes", is_flag=True, help="Execute node pool scale-down immediately.")
@click.pass_context
def scale_down_cmd(
    ctx: click.Context,
    gpu_only: bool,
    after: Optional[str],
    release_frontend: bool,
    yes: bool,
) -> None:
    """Hibernate / scale down OKE node pools to 0 nodes (preserving PVCs, Milvus data, and LB IP) or schedule with --after."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    session = OciSession.from_docs_config(cfg)
    comp_id = session.resolve_compartment_id(cfg)

    if after:
        schedule_in_cluster_ttl_job(
            cfg,
            session=session,
            compartment_id=comp_id,
            duration_str=after,
            gpu_only=gpu_only,
        )
        return

    cpu_target = None if gpu_only else 0
    gpu_target = 0
    scale_node_pools(
        cfg,
        session=session,
        compartment_id=comp_id,
        cpu_size=cpu_target,
        gpu_size=gpu_target,
        apply=yes,
        release_frontend=release_frontend,
    )


@main.command(name="scale-up")
@click.option("--cpu", default=None, type=int, help="Target CPU node count (defaults to docs.toml cpu_node_count).")
@click.option("--gpu", default=None, type=int, help="Target GPU node count (defaults to docs.toml gpu_node_count).")
@click.option("--no-gpu", is_flag=True, help="Wake only CPU nodes (leave GPU node pool at 0).")
@click.option("--ttl", default=None, help="Also arm an in-cluster auto-scale-down timer (e.g. '2h', '90m').")
@click.option("--ttl-gpu-only", is_flag=True, help="When --ttl expires, scale only the GPU pool down to 0.")
@click.option("--yes", is_flag=True, help="Execute node pool scale-up.")
@click.pass_context
def scale_up_cmd(
    ctx: click.Context,
    cpu: Optional[int],
    gpu: Optional[int],
    no_gpu: bool,
    ttl: Optional[str],
    ttl_gpu_only: bool,
    yes: bool,
) -> None:
    """Wake / scale up OKE node pools back to active capacity (with optional --ttl auto-scale-down timer)."""
    cfg = _load_cfg(ctx.obj.get("config_path"), no_gpu=no_gpu)
    session = OciSession.from_docs_config(cfg)
    comp_id = session.resolve_compartment_id(cfg)

    cpu_target = cpu if cpu is not None else cfg.cluster.cpu_node_count
    gpu_target = 0 if no_gpu else (gpu if gpu is not None else cfg.cluster.gpu_node_count)

    scale_node_pools(
        cfg,
        session=session,
        compartment_id=comp_id,
        cpu_size=cpu_target,
        gpu_size=gpu_target,
        apply=yes,
        release_frontend=False,
    )
    if yes and ttl:
        schedule_in_cluster_ttl_job(
            cfg,
            session=session,
            compartment_id=comp_id,
            duration_str=ttl,
            gpu_only=ttl_gpu_only,
        )


@main.command(name="scale")
@click.option("--cpu", default=None, type=int, help="Target CPU node pool size.")
@click.option("--gpu", default=None, type=int, help="Target GPU node pool size.")
@click.option("--yes", is_flag=True, help="Execute node pool scaling.")
@click.pass_context
def scale_cmd(ctx: click.Context, cpu: Optional[int], gpu: Optional[int], yes: bool) -> None:
    """Scale OKE CPU and/or GPU node pools to an explicit node count."""
    if cpu is None and gpu is None:
        raise click.ClickException("Specify at least one of --cpu <count> or --gpu <count>.")
    cfg = _load_cfg(ctx.obj.get("config_path"))
    session = OciSession.from_docs_config(cfg)
    comp_id = session.resolve_compartment_id(cfg)
    scale_node_pools(
        cfg,
        session=session,
        compartment_id=comp_id,
        cpu_size=cpu,
        gpu_size=gpu,
        apply=yes,
    )


@main.command(name="ttl")
@click.option("--set", "set_duration", default=None, help="Arm or update the in-cluster auto-scale-down timer (e.g. '2h', '90m').")
@click.option("--gpu-only", is_flag=True, help="Scale only the GPU node pool down to 0 when the timer expires.")
@click.option("--cancel", is_flag=True, help="Cancel the active in-cluster auto-scale-down timer.")
@click.option("--status", "show_status", is_flag=True, help="Show the active auto-scale-down timer status.")
@click.pass_context
def ttl_cmd(
    ctx: click.Context,
    set_duration: Optional[str],
    gpu_only: bool,
    cancel: bool,
    show_status: bool,
) -> None:
    """Manage the in-cluster auto-scale-down TTL Job (`kube-system/docs-agent-autoscale-ttl`)."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    if cancel:
        cancel_in_cluster_ttl_job(cfg)
        return
    if set_duration:
        session = OciSession.from_docs_config(cfg)
        comp_id = session.resolve_compartment_id(cfg)
        schedule_in_cluster_ttl_job(
            cfg,
            session=session,
            compartment_id=comp_id,
            duration_str=set_duration,
            gpu_only=gpu_only,
        )
        return

    info = get_in_cluster_ttl_status(cfg)
    if not info:
        console.print("[yellow]No active in-cluster auto-scale-down timer (`kube-system/docs-agent-autoscale-ttl`).[/yellow]")
        console.print("Arm one anytime with: [bold]docs-cli ttl --set 2h[/bold] or [bold]docs-cli scale-down --after 90m[/bold]")
        return

    mins, secs = divmod(info["remaining_seconds"], 60)
    hrs, mins = divmod(mins, 60)
    console.print(
        Panel(
            f"• [bold]Configured TTL:[/bold]    [cyan]{info['duration']} ({info['ttl_seconds']}s)[/cyan]\n"
            f"• [bold]Time Remaining:[/bold]    [bold green]{hrs}h {mins}m {secs}s[/bold green]\n"
            f"• [bold]Target Scope:[/bold]      {'GPU pool only (size=0)' if info['gpu_only'] else 'All node pools (GPU=0, CPU=0)'}\n"
            f"• [bold]Pod Status:[/bold]        active={info['active']}, succeeded={info['succeeded']}",
            title="Active In-Cluster Auto-Scale Timer",
        )
    )


@main.command()
@click.option("--yes", is_flag=True, help="Confirm ordered Terraform teardown.")
@click.pass_context
def down(ctx: click.Context, yes: bool) -> None:
    """Ordered teardown: remove K8s LoadBalancers & PVCs, then destroy platform and OKE Terraform stacks."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    if not yes:
        console.print(
            f"[yellow]Dry-run:[/yellow] Will tear down K8s LoadBalancers/PVCs, platform Terraform, and OKE cluster '{cfg.cluster.name}'.\n"
            "Re-run with [bold]docs-cli down --yes[/bold] to execute."
        )
        return

    session = OciSession.from_docs_config(cfg)
    comp_id = session.resolve_compartment_id(cfg)
    ordered_terraform_down(cfg, session, comp_id)
    console.print("[bold green]✓ Cluster teardown complete.[/bold green]")


@main.command()
@click.option("--yes", is_flag=True, help="Execute direct OCI API deletion of all tagged cluster resources.")
@click.pass_context
def nuke(ctx: click.Context, yes: bool) -> None:
    """Direct OCI API sweep of any remaining resources tagged for this cluster (`docs-cli-cluster`)."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    session = OciSession.from_docs_config(cfg)
    comp_id = session.resolve_compartment_id(cfg)

    resources = discover_tagged_oci_resources(session, comp_id, cfg.cluster.name)
    if not resources:
        console.print(
            f"[green]✓ No lingering OCI resources found for cluster '{cfg.cluster.name}' in compartment {comp_id}.[/green]"
        )
        return

    render_nuke_plan_table(resources, cfg.cluster.name)
    if not yes:
        console.print(
            "\n[yellow]Dry-run mode.[/yellow] Re-run with [bold]docs-cli nuke --yes[/bold] to permanently delete these OCI resources."
        )
        return

    nuke_tagged_oci_resources(session, comp_id, cfg.cluster.name, resources)
    console.print("[bold green]✓ OCI nuke sweep dispatched.[/bold green]")


@main.command(name="destroy")
@click.option("--yes", is_flag=True, help="Permanently tear down the entire dev environment (K8s LBs/PVCs + Terraform + OCI sweep).")
@click.pass_context
def destroy_cmd(ctx: click.Context, yes: bool) -> None:
    """Tear down the entire dev environment end-to-end (port-forward, K8s LoadBalancers/PVCs, platform & OKE Terraform, OCI sweep, and local state)."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    session = OciSession.from_docs_config(cfg)
    comp_id = session.resolve_compartment_id(cfg)
    resources = discover_tagged_oci_resources(session, comp_id, cfg.cluster.name)
    if resources:
        render_nuke_plan_table(resources, cfg.cluster.name)
    if not yes:
        console.print(
            f"[yellow]Dry-run:[/yellow] Will tear down K8s LoadBalancers/PVCs, platform & OKE Terraform stacks, and verify 0 OCI resources remain for '{cfg.cluster.name}'.\n"
            "Re-run with [bold]docs-cli destroy --yes[/bold] to execute full teardown."
        )
        return

    remaining = destroy_full_environment(cfg, session, comp_id)
    if remaining:
        render_nuke_plan_table(remaining, cfg.cluster.name)
        raise click.ClickException(
            f"{len(remaining)} OCI resource(s) are still terminating; re-run `docs-cli destroy --yes` in a minute."
        )
    console.print(f"[bold green]✓ Environment '{cfg.cluster.name}' completely destroyed (0 cloud resources remaining).[/bold green]")


@main.command(name="update")
@click.option("--pull", is_flag=True, help="Run `git pull --ff-only` before syncing upstream charts and Terraform.")
@click.option(
    "--track",
    type=click.Choice(["main", "prod", "local"], case_sensitive=False),
    default="local",
    help="Resolve latest published GHCR image tag from 'main' commits, Argo CD 'prod' revision, or 'local' values.yaml.",
)
@click.option(
    "--compile-pipelines",
    is_flag=True,
    help="Force recompilation of KFP pipeline Python definitions into K8s-native Pipeline/PipelineVersion CRDs.",
)
@click.option(
    "--check",
    is_flag=True,
    help="CI check mode: fail with exit code 1 if docs.toml or templates have drifted from upstream charts/Terraform.",
)
@click.option(
    "--diagnose",
    is_flag=True,
    help="Emit structured JSON multi-agent, Argo CD app, and Secret contract topology report for agentic migration workflows.",
)
@click.option("--apply", is_flag=True, help="Upgrade active cluster Helm releases & Kubeflow Dashboard and run live `docs-cli verify`.")
@click.option("--skip-verify", is_flag=True, help="Skip live cluster verification after applying updates.")
@click.pass_context
def update_cmd(
    ctx: click.Context,
    pull: bool,
    track: str,
    compile_pipelines: bool,
    check: bool,
    diagnose: bool,
    apply: bool,
    skip_verify: bool,
) -> None:
    """Sync CLI defaults from upstream Helm/Terraform/GHCR, recompile pipelines if changed, run tests, and optionally upgrade + verify the active cluster."""
    import sys

    cfg = _load_cfg(ctx.obj.get("config_path"))

    if pull:
        console.print("[bold cyan]0. Pulling latest repository changes (`git pull --ff-only`)...[/bold cyan]")
        subprocess.run(["git", "-C", str(cfg.repo_root), "pull", "--ff-only"], check=True)

    if track.lower() in ("main", "prod"):
        console.print(f"[bold cyan]Resolving latest verified GHCR image tag (track={track})...[/bold cyan]")
        resolved_tag = cfg.resolve_latest_ghcr_tag(track=track.lower())
        if resolved_tag:
            console.print(f"  [green]✓ Verified published GHCR tag for mcp-kubeflow-docs & session-issuer: {resolved_tag}[/green]")
            cfg.mcp.image_tag = resolved_tag
        else:
            console.print("  [yellow]! Could not reach GHCR API; keeping existing chart tag.[/yellow]")

    console.print("[bold cyan]1. Introspecting upstream Helm charts, Argo CD pins & Terraform variables...[/bold cyan]")
    discovered = cfg.discover_upstream_defaults()
    changes = cfg.sync_with_upstream(update_docs_toml=not check)

    sync_table = Table(title="Upstream Component Source-of-Truth Sync")
    sync_table.add_column("Source Key", style="cyan")
    sync_table.add_column("Discovered Value", style="bold")
    sync_table.add_column("Sync Status", justify="center")

    changed_by_source = {c["source"]: c for c in changes}
    for k, v in sorted(discovered.items()):
        if k in changed_by_source:
            c = changed_by_source[k]
            status_str = f"[bold yellow]{'DRIFT' if check else 'UPDATED'} ({c['old']} -> {c['new']})[/bold yellow]"
        else:
            status_str = "[green]IN SYNC[/green]"
        sync_table.add_row(k, str(v), status_str)
    console.print(sync_table)

    if check and changes:
        raise click.ClickException(
            f"Detected {len(changes)} drifted configuration field(s). Run `docs-cli update` to sync docs.toml."
        )

    # Auto-detect if pipeline .py sources are newer than compiled .yaml files
    chart_pipes_dir = cfg.charts_dir / "docs-agent" / "files" / "pipelines"
    needs_pipe_compile = compile_pipelines
    if not needs_pipe_compile and cfg.pipelines_dir.exists() and chart_pipes_dir.exists():
        py_mtimes = [p.stat().st_mtime for p in cfg.pipelines_dir.glob("*-pipeline.py")]
        yaml_mtimes = [y.stat().st_mtime for y in chart_pipes_dir.glob("*.yaml")]
        if py_mtimes and (not yaml_mtimes or max(py_mtimes) > min(yaml_mtimes)):
            needs_pipe_compile = True

    if needs_pipe_compile:
        console.print("\n[bold cyan]1b. Recompiling KFP pipeline Python definitions into K8s-native CRDs...[/bold cyan]")
        compiled_files = compile_kfp_pipelines(cfg)
        for cf in compiled_files:
            wrap_compiled_pipeline_as_k8s_native_crs(
                cf,
                namespace=cfg.platform.namespace_kubeflow,
                version_name="v4",
            )
        console.print(f"  [green]✓ Recompiled & wrapped {len(compiled_files)} KFP pipeline(s)[/green]")

    # Re-render local Helm & Terraform values
    lb_ip = get_ingress_lb_ip(cfg, timeout_seconds=5) if cfg.kubeconfig_path.exists() else None
    domain = cfg.resolve_agent_domain(lb_ip=lb_ip)
    values_files = render_helm_values_files(cfg, domain=domain)
    render_tfvars_files(cfg, compartment_id=cfg.oci.compartment_id or "ocid1.compartment.oc1..example")
    console.print("[green]✓ Re-rendered .docs-cli/helm/*.values.yaml and .docs-cli/platform.tfvars.json[/green]")

    # 2. Run Helm lint & rendered Secret/Schema contract check across local charts
    console.print("\n[bold cyan]2. Running `helm lint` & rendered contract validation on local charts...[/bold cyan]")
    contract_issues = validate_rendered_chart_contracts(cfg, values_files)
    if diagnose:
        import json as _json

        known_argocd = {
            "docs-agent",
            "gateway-guardrails",
            "llm-runtime",
            "reranker",
            "docs-agent-pr-eval",
            "docs-agent-preview",
            "docs-agent-staging",
            "langfuse",
            "langfuse-retention",
            "llm-judge",
            "otel-collector",
            "preview-namespace",
        }
        argo_apps = cfg.discover_argocd_apps()
        unmapped_apps = [a for a in argo_apps if a["name"] not in known_argocd]
        diag_payload = {
            "enabled_kagent_agents": cfg.discover_enabled_kagent_agents(),
            "argocd_apps": argo_apps,
            "unmapped_argocd_apps": unmapped_apps,
            "drifted_fields": changes,
            "contract_issues": contract_issues,
            "status": "ACTION_REQUIRED" if (contract_issues or unmapped_apps) else "HEALTHY",
        }
        diag_path = cfg.state_dir / "drift-diagnosis.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(_json.dumps(diag_payload, indent=2) + "\n", encoding="utf-8")
        console.print(
            Panel(
                _json.dumps(diag_payload, indent=2),
                title=f"Multi-Agent & Upstream Drift Diagnosis ({diag_path})",
            )
        )
    if contract_issues:
        for issue in contract_issues:
            console.print(f"  [red]✗ Contract drift: {issue}[/red]")
        raise click.ClickException(
            "Upstream chart contract changed (new Secret or broken values schema). Update cli/docs_cli/kube.py."
        )
    console.print("  [green]✓ Rendered chart Secret & values contracts verified[/green]")
    chart_dirs = [
        cfg.charts_dir / "llm-runtime",
        cfg.charts_dir / "docs-agent",
        cfg.charts_dir / "gateway-guardrails",
    ]
    for cdir in chart_dirs:
        if not cdir.exists():
            continue
        extra_args = ["--set", f"domain={domain}"] if cdir.name == "gateway-guardrails" else []
        res = subprocess.run(
            ["helm", "lint", str(cdir), *extra_args],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            console.print(f"  [green]✓ helm lint {cdir.name}: PASSED[/green]")
        else:
            console.print(f"  [red]✗ helm lint {cdir.name}: FAILED[/red]\n{res.stdout}\n{res.stderr}")
            raise click.ClickException(f"helm lint failed for {cdir.name}")

    # 3. Run unit test suite (pytest)
    console.print("\n[bold cyan]3. Running CLI unit test suite (`pytest`)...[/bold cyan]")
    tests_dir = cfg.repo_root / "cli" / "tests"
    pytest_res = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    if pytest_res.returncode == 0:
        console.print(f"  [green]✓ pytest unit tests: PASSED[/green] ({pytest_res.stdout.strip().splitlines()[-1]})")
    else:
        console.print(f"  [red]✗ pytest unit tests: FAILED[/red]\n{pytest_res.stdout}\n{pytest_res.stderr}")
        raise click.ClickException("Unit test suite failed during `docs-cli update`.")

    # 4. Optionally apply to active cluster & run live verification
    if apply:
        if not cfg.kubeconfig_path.exists():
            raise click.ClickException(
                f"No kubeconfig found at {cfg.kubeconfig_path}. Run `docs-cli kubeconfig` first."
            )
        console.print("\n[bold cyan]4. Upgrading active cluster releases & Kubeflow Dashboard...[/bold cyan]")
        domain = deploy_helm_stack(cfg, skip_kagent=True)
        if not skip_verify:
            console.print("\n[bold cyan]5. Running live environment verification (`docs-cli verify`)...[/bold cyan]")
            results, all_passed, agent_url, page_url = run_verification_suite(
                cfg,
                domain=domain,
                exposed_page_url=f"https://chat.{domain}",
            )
            print_verification_report(results, all_passed, agent_url, page_url)
            if not all_passed:
                raise SystemExit(1)


@main.command(name="expose-dashboard")
@click.option("--port", default=8085, type=int, help="Local port for the Kubeflow Central Dashboard (default: 8085).")
@click.option("--address", default="localhost,127.0.0.1", help="Local bind address (default: localhost,127.0.0.1 for IPv4 + IPv6).")
@click.option(
    "--background",
    "-b",
    is_flag=True,
    help="Run the port-forward in the background and save its PID in .docs-cli/dashboard-port-forward.pid.",
)
@click.pass_context
def expose_dashboard_cmd(
    ctx: click.Context,
    port: int,
    address: str,
    background: bool,
) -> None:
    """Port-forward the unified Kubeflow Central Dashboard (Pipelines + KServe Models + Cross-Namespace Selector) to localhost."""
    cfg = _load_cfg(ctx.obj.get("config_path"))
    if not cfg.kubeconfig_path.exists():
        raise click.ClickException(
            f"No kubeconfig found at {cfg.kubeconfig_path}. Run `docs-cli kubeconfig` first."
        )
    expose_kubeflow_dashboard_port_forward(
        cfg,
        local_port=port,
        address=address,
        background=background,
    )


@main.command(name="dashboard")
@click.option("--port", default=8085, type=int, help="Local port for the Kubeflow Central Dashboard (default: 8085).")
@click.option("--address", default="localhost,127.0.0.1", help="Local bind address (default: localhost,127.0.0.1 for IPv4 + IPv6).")
@click.option(
    "--background",
    "-b",
    is_flag=True,
    help="Run the port-forward in the background and save its PID in .docs-cli/dashboard-port-forward.pid.",
)
@click.pass_context
def dashboard_cmd(
    ctx: click.Context,
    port: int,
    address: str,
    background: bool,
) -> None:
    """Alias for `expose-dashboard`: port-forward the Kubeflow Central Dashboard to http://localhost:8085."""
    ctx.invoke(expose_dashboard_cmd, port=port, address=address, background=background)


if __name__ == "__main__":
    main()
