"""Unit tests for docs-cli (config, Terraform templating, Helm values, Netlify hooks, nuke tags)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from docs_cli.cli import main
from docs_cli.config import DocsConfig
from docs_cli.kube import (
    build_docs_agent_values,
    build_gateway_guardrails_values,
    build_llm_runtime_values,
    generate_rsa_keypair_pem,
    generate_self_signed_tls_pem,
    wrap_compiled_pipeline_as_k8s_native_crs,
)
from docs_cli.netlify import (
    MARKER_BEGIN,
    MARKER_END,
    build_agent_endpoint_url,
    inject_hugo_docsy_hooks,
    prepare_demo_bundle,
)
from docs_cli.nuke import has_cluster_ownership_tag
from docs_cli.scale import parse_duration_seconds
from docs_cli.terraform import build_oke_tfvars, build_platform_tfvars


def test_config_loads_defaults_and_env_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OCI_REGION", "us-ashburn-1")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..testcomp")
    monkeypatch.setenv("NETLIFY_SITE_URL", "https://deploy-preview-99--kubeflow.netlify.app/")
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")

    repo_root = Path(__file__).resolve().parents[2]
    cfg = DocsConfig.load(path=repo_root / "docs.toml")

    assert cfg.oci.region == "us-ashburn-1"
    assert cfg.oci.compartment_id == "ocid1.compartment.oc1..testcomp"
    assert cfg.cluster.enable_gpu_pool is True
    assert cfg.llm.hf_token == "hf_test_secret"

    cors = cfg.resolve_cors_origins()
    assert cors[0] == "https://deploy-preview-99--kubeflow.netlify.app"
    assert "https://www.kubeflow.org" in cors


def test_oke_and_platform_tfvars_gpu_and_no_gpu(monkeypatch) -> None:
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..xyz")
    repo_root = Path(__file__).resolve().parents[2]

    cfg_gpu = DocsConfig.load(path=repo_root / "docs.toml", enable_gpu_override=True)
    oke_gpu = build_oke_tfvars(cfg_gpu)
    assert oke_gpu["enable_gpu_pool"] is True
    assert oke_gpu["gpu_node_shape"] == "VM.GPU.A10.1"
    assert oke_gpu["compartment_ocid"] == "ocid1.compartment.oc1..xyz"

    cfg_cpu = DocsConfig.load(path=repo_root / "docs.toml", enable_gpu_override=False)
    oke_cpu = build_oke_tfvars(cfg_cpu)
    assert oke_cpu["enable_gpu_pool"] is False

    platform_vars = build_platform_tfvars(cfg_gpu)
    assert platform_vars["enable_istio_ingressgateway"] is True
    assert platform_vars["namespace_ml_infra"] == "ml-infra"
    assert platform_vars["namespace_docs_agent"] == "docs-agent"


def test_helm_values_generation_gpu_vs_external_llm(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_123")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-1")
    repo_root = Path(__file__).resolve().parents[2]

    cfg_gpu = DocsConfig.load(path=repo_root / "docs.toml", enable_gpu_override=True)
    llm_vals = build_llm_runtime_values(cfg_gpu)
    assert llm_vals["model"] == "gemma"
    assert llm_vals["hfToken"]["existingSecret"] == "huggingface-secret"

    docs_vals_gpu = build_docs_agent_values(cfg_gpu)
    assert "models" not in docs_vals_gpu["kagent"]
    assert len(docs_vals_gpu["mcp"]["extraEnv"]) == 2

    cfg_cpu = DocsConfig.load(path=repo_root / "docs.toml", enable_gpu_override=False)
    docs_vals_cpu = build_docs_agent_values(cfg_cpu)
    assert "models" in docs_vals_cpu["kagent"]
    assert docs_vals_cpu["kagent"]["models"][0]["model"] == cfg_cpu.llm.external_model

    gw_vals = build_gateway_guardrails_values(cfg_gpu, domain="203.0.113.10.nip.io")
    assert gw_vals["domain"] == "203.0.113.10.nip.io"
    assert gw_vals["gateway"]["tls"]["acme"]["enabled"] is False
    assert gw_vals["sessionAuth"]["enabled"] is True


def test_crypto_generators() -> None:
    priv_pem, pub_pem = generate_rsa_keypair_pem()
    assert "BEGIN PRIVATE KEY" in priv_pem
    assert "BEGIN PUBLIC KEY" in pub_pem

    cert_pem, key_pem = generate_self_signed_tls_pem("203.0.113.10.nip.io")
    assert "BEGIN CERTIFICATE" in cert_pem
    assert "PRIVATE KEY" in key_pem


def test_netlify_hugo_docsy_injection_is_idempotent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = DocsConfig.load(path=repo_root / "docs.toml")

    fake_website = tmp_path / "kubeflow-website"
    (fake_website / "layouts" / "partials" / "hooks").mkdir(parents=True)
    (fake_website / "static").mkdir(parents=True)

    # Pre-populate head-end.html with existing custom HTML to verify preservation
    head_hook = fake_website / "layouts" / "partials" / "hooks" / "head-end.html"
    head_hook.write_text("<!-- Custom Analytics -->\n", encoding="utf-8")

    res1 = inject_hugo_docsy_hooks(
        cfg, domain="agent.kubeflow-dev.org", website_dir=fake_website
    )
    assert res1["css"].exists()
    assert res1["js"].exists()

    head_content_1 = res1["head_hook"].read_text(encoding="utf-8")
    body_content_1 = res1["body_hook"].read_text(encoding="utf-8")
    assert "<!-- Custom Analytics -->" in head_content_1
    assert '/docs_styles/chatbot.css' in head_content_1
    assert (
        "https://agent.kubeflow-dev.org/a2a/docs-agent/kubeflow-docs-agent"
        in body_content_1
    )

    # Second injection with a new domain should replace between markers without duplicating
    inject_hugo_docsy_hooks(
        cfg, domain="new-agent.kubeflow-dev.org", website_dir=fake_website
    )
    body_content_2 = res1["body_hook"].read_text(encoding="utf-8")
    assert body_content_2.count(MARKER_BEGIN) == 1
    assert body_content_2.count(MARKER_END) == 1
    assert "new-agent.kubeflow-dev.org" in body_content_2
    assert "https://agent.kubeflow-dev.org" not in body_content_2


def test_netlify_standalone_demo_bundle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = DocsConfig.load(path=repo_root / "docs.toml")

    bundle_dir = prepare_demo_bundle(
        cfg, domain="198.51.100.42.nip.io", out_dir=tmp_path / "demo-dist"
    )
    assert (bundle_dir / "index.html").exists()
    assert (bundle_dir / "docs_scripts" / "chatbot.js").exists()
    assert (bundle_dir / "docs_styles" / "chatbot.css").exists()
    assert (bundle_dir / "netlify.toml").exists()

    html = (bundle_dir / "index.html").read_text(encoding="utf-8")
    assert (
        build_agent_endpoint_url("198.51.100.42.nip.io")
        in html
    )


def test_kfp_k8s_native_crd_wrapper(tmp_path: Path) -> None:
    import yaml

    fake_compiled = tmp_path / "github_rag_pipeline.yaml"
    fake_compiled.write_text(
        """pipelineInfo:
  name: github-rag
  description: RAG pipeline for processing GitHub documentation
root:
  dag:
    tasks: {}
schemaVersion: 2.1.0
sdkVersion: kfp-2.18.0
---
platforms:
  kubernetes:
    deploymentSpec:
      executors: {}
""",
        encoding="utf-8",
    )

    out_file = wrap_compiled_pipeline_as_k8s_native_crs(
        fake_compiled,
        namespace="kubeflow",
        version_name="v4",
        out_dir=tmp_path / "k8s-native",
    )
    assert out_file.exists()
    docs = list(yaml.safe_load_all(out_file.read_text(encoding="utf-8")))
    assert len(docs) == 2
    assert docs[0]["apiVersion"] == "pipelines.kubeflow.org/v2beta1"
    assert docs[0]["kind"] == "Pipeline"
    assert docs[0]["metadata"]["name"] == "github-rag"
    assert docs[1]["apiVersion"] == "pipelines.kubeflow.org/v2beta1"
    assert docs[1]["kind"] == "PipelineVersion"
    assert docs[1]["spec"]["pipelineName"] == "github-rag"
    assert docs[1]["spec"]["versionName"] == "v4"
    assert "pipelineSpec" in docs[1]["spec"]
    assert "platformSpec" in docs[1]["spec"]


def test_nuke_tag_ownership_verification() -> None:
    cluster = "kubeflow-docs-agent-dev"
    assert (
        has_cluster_ownership_tag(
            {"docs-cli-managed": "true", "docs-cli-cluster": cluster},
            cluster,
        )
        is True
    )
    assert (
        has_cluster_ownership_tag(
            {"docs-cli-managed": "true", "docs-cli-cluster": "other-cluster"},
            cluster,
        )
        is False
    )
    assert has_cluster_ownership_tag({}, cluster) is False
    assert has_cluster_ownership_tag(None, cluster) is False


def test_cli_template_and_doctor_commands(monkeypatch, tmp_path: Path) -> None:
    isolated_state = tmp_path / ".docs-cli"
    monkeypatch.setattr(DocsConfig, "state_dir", property(lambda self: isolated_state))
    monkeypatch.setattr(
        DocsConfig,
        "kubeconfig_path",
        property(lambda self: isolated_state / "kubeconfig"),
    )
    monkeypatch.setattr("docs_cli.cli.get_ingress_lb_ip", lambda cfg, timeout_seconds=5: None)
    runner = CliRunner()
    res_help = runner.invoke(main, ["--help"])
    assert res_help.exit_code == 0
    assert "template" in res_help.output
    assert "up" in res_help.output
    assert "netlify" in res_help.output
    assert "nuke" in res_help.output

    res_tmpl = runner.invoke(
        main, ["template", "--no-gpu", "--domain", "test.agent.local"]
    )
    assert res_tmpl.exit_code == 0

    oke_json = json.loads(
        (isolated_state / "oke.tfvars.json").read_text(encoding="utf-8")
    )
    assert oke_json["enable_gpu_pool"] is False
    runner.invoke(main, ["template"])

    from unittest.mock import patch
    import subprocess

    with patch(
        "docs_cli.verify._run_quiet",
        return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
    ):
        res_verify = runner.invoke(
            main,
            ["verify", "--domain", "test.agent.local", "--page-url", "https://preview.kubeflow.netlify.app"],
        )
    assert "Ordered Deployment Verification Report" in res_verify.output
    assert "https://preview.kubeflow.netlify.app" in res_verify.output


def test_parse_duration_seconds_and_scale_cli_commands() -> None:
    assert parse_duration_seconds("2h") == 7200
    assert parse_duration_seconds("90m") == 5400
    assert parse_duration_seconds("1h30m") == 5400
    assert parse_duration_seconds("45") == 2700
    assert parse_duration_seconds("3600s") == 3600

    runner = CliRunner()
    res_help = runner.invoke(main, ["--help"])
    assert res_help.exit_code == 0
    assert "scale-down" in res_help.output
    assert "scale-up" in res_help.output
    assert "scale" in res_help.output
    assert "ttl" in res_help.output
    assert "update" in res_help.output
    assert "expose-dashboard" in res_help.output
    assert "dashboard" in res_help.output


def test_upstream_sync_and_isolated_frontend_tls_sans() -> None:
    from cryptography import x509

    cfg = DocsConfig.load()
    discovered = cfg.discover_upstream_defaults()
    assert "mcp.image_repository" in discovered
    assert "mcp.collection_docs" in discovered
    assert "tf.milvus_version" in discovered

    domain = "129.153.227.208.nip.io"
    cert_pem, _ = generate_self_signed_tls_pem(domain)
    cert = x509.load_pem_x509_certificate(cert_pem.encode("utf-8"))
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    dns_names = san.get_values_for_type(x509.DNSName)
    assert domain in dns_names
    assert f"*.{domain}" in dns_names
    assert f"chat.{domain}" in dns_names
    assert f"kubeflow.{domain}" in dns_names

    gw_released = build_gateway_guardrails_values(cfg, domain, demo_web_enabled=False)
    assert gw_released["demoWeb"]["enabled"] is False
    origins = [o["exact"] for o in gw_released["routing"]["cors"]["allowOrigins"]]
    assert f"https://chat.{domain}" in origins


def test_destroy_command_and_dynamic_multi_agent_discovery(monkeypatch, tmp_path: Path) -> None:
    import subprocess

    isolated_state = tmp_path / ".docs-cli"
    monkeypatch.setattr(DocsConfig, "state_dir", property(lambda self: isolated_state))
    monkeypatch.setattr(
        DocsConfig,
        "kubeconfig_path",
        property(lambda self: isolated_state / "kubeconfig"),
    )

    cfg = DocsConfig.load()
    agents = cfg.discover_enabled_kagent_agents()
    assert "kubeflow-docs-agent" in agents
    assert "kubeflow-docs-agent-think" in agents
    assert "kubeflow-debug-agent" in agents

    argo_apps = cfg.discover_argocd_apps()
    app_names = {a["name"] for a in argo_apps}
    assert "docs-agent" in app_names
    assert "llm-runtime" in app_names
    assert "gateway-guardrails" in app_names

    discovered = cfg.discover_upstream_defaults()
    assert "kagent.enabled_agents" in discovered
    assert "argocd.apps" in discovered

    runner = CliRunner()
    res_help = runner.invoke(main, ["--help"])
    assert res_help.exit_code == 0
    assert "destroy" in res_help.output

    res_destroy_help = runner.invoke(main, ["destroy", "--help"])
    assert res_destroy_help.exit_code == 0
    assert "--yes" in res_destroy_help.output

    # 1. Healthy expected-architecture check passes and emits HEALTHY JSON
    monkeypatch.setattr("docs_cli.cli.validate_rendered_chart_contracts", lambda c, v: [])
    monkeypatch.setattr(
        "docs_cli.cli.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="1 passed", stderr=""),
    )
    res_healthy = runner.invoke(main, ["update", "--check", "--diagnose"])
    assert res_healthy.exit_code == 0
    diag_path = isolated_state / "drift-diagnosis.json"
    diag_healthy = json.loads(diag_path.read_text(encoding="utf-8"))
    assert diag_healthy["status"] == "HEALTHY"
    assert "kubeflow-docs-agent-think" in diag_healthy["enabled_kagent_agents"]

    # 2. Simulate upstream adding an unmapped new Argo CD microservice app (e.g. semantic-router) -> fails gate
    monkeypatch.setattr(
        DocsConfig,
        "discover_argocd_apps",
        lambda self: [
            *argo_apps,
            {"name": "semantic-router", "path": "docs-agent-mcp/charts/semantic-router", "targetRevision": "main"},
        ],
    )
    res_unmapped = runner.invoke(main, ["update", "--check", "--diagnose"])
    assert res_unmapped.exit_code != 0
    diag_unmapped = json.loads(diag_path.read_text(encoding="utf-8"))
    assert diag_unmapped["status"] == "ACTION_REQUIRED"
    assert "semantic-router" in diag_unmapped["unmapped_argocd_apps"]
