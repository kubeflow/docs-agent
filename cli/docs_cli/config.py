"""Configuration loader for docs-cli (docs.toml + secure environment variables)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


def load_dotenv_file(dotenv_path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE .env file without clobbering existing os.environ."""
    loaded: Dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        loaded[key] = val
        if key not in os.environ and val:
            os.environ[key] = val
    return loaded


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class OciConfig:
    profile: str = "DEFAULT"
    region: str = "us-ashburn-1"
    compartment_name: str = "kubeflow"
    compartment_id: str = ""
    tenancy_ocid: str = ""
    user_ocid: str = ""
    fingerprint: str = ""
    private_key_path: str = ""
    private_key_passphrase: str = ""


@dataclass
class ClusterConfig:
    name: str = "kubeflow-docs-agent-dev"
    kubernetes_version: str = "v1.36.4"
    vcn_cidr: str = "10.0.0.0/16"
    cpu_node_shape: str = "VM.Standard.E5.Flex"
    cpu_node_count: int = 2
    cpu_node_ocpus: int = 4
    cpu_node_memory_gb: int = 64
    cpu_boot_volume_gb: int = 100
    enable_gpu_pool: bool = True
    gpu_node_shape: str = "VM.GPU.A10.1"
    gpu_node_count: int = 1
    gpu_node_ocpus: int = 15
    gpu_node_memory_gb: int = 240
    gpu_boot_volume_gb: int = 200
    gpu_node_image_id: str = ""


@dataclass
class PlatformConfig:
    namespace_ml_infra: str = "ml-infra"
    namespace_docs_agent: str = "docs-agent"
    namespace_kubeflow: str = "kubeflow"
    cert_manager_version: str = "v1.20.2"
    istio_version: str = "1.23.0"
    knative_version: str = "1.22.0"
    kserve_version: str = "v0.17.0"
    kfp_version: str = "2.15.0"
    milvus_version: str = "v2.6.22"
    use_kubeflow_helm_charts: bool = True
    kubeflow_pipelines_scenario: str = "platform-k8s-native"
    enable_kubeflow_full_platform_helm: bool = False
    embeddings_model_id: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model_id: str = "BAAI/bge-reranker-base"


@dataclass
class LlmConfig:
    active_profile: str = "gemma4"
    external_base_url: str = "https://api.openai.com/v1"
    external_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    hf_token: str = ""


@dataclass
class McpConfig:
    image_repository: str = "ghcr.io/kubeflow/mcp-kubeflow-docs"
    image_tag: str = "145409a"
    collection_docs: str = "kubeflow_docs"
    collection_issues: str = "issues_rag"
    collection_code: str = "code_rag"
    reranker_enabled: bool = True
    milvus_password: str = ""
    github_token: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


@dataclass
class IngressConfig:
    domain: str = ""
    tls_mode: str = "self-signed"
    acme_email: str = ""
    enable_jwt_guardrails: bool = True
    enable_oauth2_proxy: bool = False
    lb_min_bandwidth_mbps: int = 10
    lb_max_bandwidth_mbps: int = 100


@dataclass
class NetlifyConfig:
    mode: str = "website"
    website_repo_path: str = "../kubeflow-website/website"
    extra_cors_origins: List[str] = field(
        default_factory=lambda: [
            "https://www.kubeflow.org",
            "http://localhost:1313",
            "http://localhost:8080",
        ]
    )
    auth_token: str = ""
    site_id: str = ""
    site_url: str = ""


@dataclass
class DocsConfig:
    oci: OciConfig
    cluster: ClusterConfig
    platform: PlatformConfig
    llm: LlmConfig
    mcp: McpConfig
    ingress: IngressConfig
    netlify: NetlifyConfig
    repo_root: Path

    @property
    def state_dir(self) -> Path:
        return self.repo_root / ".docs-cli"

    @property
    def kubeconfig_path(self) -> Path:
        env_kc = os.environ.get("DOCS_KUBECONFIG")
        if env_kc:
            return Path(env_kc).expanduser().resolve()
        return self.state_dir / "kubeconfig"

    @property
    def oke_tf_dir(self) -> Path:
        return self.repo_root / "deployments" / "terraform" / "oke-cluster"

    @property
    def platform_tf_dir(self) -> Path:
        return self.repo_root / "docs-agent-mcp" / "terraform"

    @property
    def charts_dir(self) -> Path:
        return self.repo_root / "docs-agent-mcp" / "charts"

    @property
    def pipelines_dir(self) -> Path:
        return self.repo_root / "docs-agent-mcp" / "pipelines"

    @property
    def frontend_dir(self) -> Path:
        return self.repo_root / "frontend"

    def resolve_cors_origins(self) -> List[str]:
        """Combine extra_cors_origins with NETLIFY_SITE_URL without duplicates."""
        origins: List[str] = []
        if self.netlify.site_url:
            clean = self.netlify.site_url.rstrip("/")
            if clean not in origins:
                origins.append(clean)
        for origin in self.netlify.extra_cors_origins:
            clean = origin.rstrip("/")
            if clean and clean not in origins:
                origins.append(clean)
        return origins

    def resolve_agent_domain(self, lb_ip: Optional[str] = None) -> str:
        """Return explicit domain or construct <lb_ip>.nip.io fallback."""
        if self.ingress.domain:
            return self.ingress.domain.strip()
        if lb_ip:
            return f"{lb_ip.strip()}.nip.io"
        return "agent.docs-agent.local"

    def ensure_milvus_password(self) -> str:
        """Return MILVUS_PASSWORD or persist a generated one in .docs-cli/milvus_password."""
        if self.mcp.milvus_password:
            return self.mcp.milvus_password
        self.state_dir.mkdir(parents=True, exist_ok=True)
        pw_file = self.state_dir / "milvus_password"
        if pw_file.exists():
            self.mcp.milvus_password = pw_file.read_text(encoding="utf-8").strip()
            return self.mcp.milvus_password
        generated = secrets.token_urlsafe(24)
        pw_file.write_text(generated, encoding="utf-8")
        pw_file.chmod(0o600)
        self.mcp.milvus_password = generated
        return generated

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        enable_gpu_override: Optional[bool] = None,
    ) -> "DocsConfig":
        if path is None:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "docs.toml"
        else:
            path = path.resolve()
            repo_root = path.parent

        load_dotenv_file(repo_root / ".env")

        raw: dict = {}
        if path.exists():
            with path.open("rb") as f:
                raw = tomllib.load(f)

        oci_raw = dict(raw.get("oci", {}))
        cluster_raw = dict(raw.get("cluster", {}))
        platform_raw = dict(raw.get("platform", {}))
        llm_raw = dict(raw.get("llm", {}))
        mcp_raw = dict(raw.get("mcp", {}))
        ingress_raw = dict(raw.get("ingress", {}))
        netlify_raw = dict(raw.get("netlify", {}))

        # Apply environment variable overrides
        oci_cfg = OciConfig(
            profile=os.environ.get("OCI_PROFILE") or oci_raw.get("profile", "DEFAULT"),
            region=os.environ.get("OCI_REGION") or oci_raw.get("region", "us-chicago-1"),
            compartment_name=os.environ.get("OCI_COMPARTMENT_NAME")
            or oci_raw.get("compartment_name", "kubeflow-docs-agent-dev"),
            compartment_id=os.environ.get("OCI_COMPARTMENT_ID")
            or oci_raw.get("compartment_id", ""),
            tenancy_ocid=os.environ.get("OCI_TENANCY", ""),
            user_ocid=os.environ.get("OCI_USER", ""),
            fingerprint=os.environ.get("OCI_FINGERPRINT", ""),
            private_key_path=os.environ.get("OCI_KEY_FILE", ""),
            private_key_passphrase=os.environ.get("OCI_KEY_PASSPHRASE", ""),
        )

        default_gpu = bool(cluster_raw.get("enable_gpu_pool", True))
        enable_gpu = _parse_bool(os.environ.get("DOCS_ENABLE_GPU"), default_gpu)
        if enable_gpu_override is not None:
            enable_gpu = enable_gpu_override

        cluster_cfg = ClusterConfig(
            name=os.environ.get("DOCS_CLUSTER_NAME")
            or cluster_raw.get("name", "kubeflow-docs-agent-dev"),
            kubernetes_version=os.environ.get("DOCS_K8S_VERSION")
            or cluster_raw.get("kubernetes_version", "v1.31.1"),
            vcn_cidr=cluster_raw.get("vcn_cidr", "10.0.0.0/16"),
            cpu_node_shape=cluster_raw.get("cpu_node_shape", "VM.Standard.E5.Flex"),
            cpu_node_count=int(cluster_raw.get("cpu_node_count", 2)),
            cpu_node_ocpus=int(cluster_raw.get("cpu_node_ocpus", 4)),
            cpu_node_memory_gb=int(cluster_raw.get("cpu_node_memory_gb", 64)),
            cpu_boot_volume_gb=int(cluster_raw.get("cpu_boot_volume_gb", 100)),
            enable_gpu_pool=enable_gpu,
            gpu_node_shape=cluster_raw.get("gpu_node_shape", "VM.GPU.A10.1"),
            gpu_node_count=int(cluster_raw.get("gpu_node_count", 1)),
            gpu_node_ocpus=int(cluster_raw.get("gpu_node_ocpus", 15)),
            gpu_node_memory_gb=int(cluster_raw.get("gpu_node_memory_gb", 240)),
            gpu_boot_volume_gb=int(cluster_raw.get("gpu_boot_volume_gb", 200)),
            gpu_node_image_id=os.environ.get("OCI_GPU_NODE_IMAGE_ID")
            or cluster_raw.get("gpu_node_image_id", ""),
        )

        platform_cfg = PlatformConfig(**platform_raw)

        llm_cfg = LlmConfig(
            active_profile=os.environ.get("DOCS_LLM_PROFILE")
            or llm_raw.get("active_profile", "gemma4"),
            external_base_url=os.environ.get("LLM_BASE_URL")
            or llm_raw.get("external_base_url", "https://api.openai.com/v1"),
            external_model=os.environ.get("LLM_MODEL")
            or llm_raw.get("external_model", "gpt-4o-mini"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            hf_token=os.environ.get("HF_TOKEN", ""),
        )

        mcp_cfg = McpConfig(
            image_repository=mcp_raw.get(
                "image_repository", "ghcr.io/kubeflow/mcp-kubeflow-docs"
            ),
            image_tag=os.environ.get("DOCS_MCP_IMAGE_TAG")
            or mcp_raw.get("image_tag", "145409a"),
            collection_docs=mcp_raw.get("collection_docs", "kubeflow_docs"),
            collection_issues=mcp_raw.get("collection_issues", "issues_rag"),
            collection_code=mcp_raw.get("collection_code", "code_rag"),
            reranker_enabled=bool(mcp_raw.get("reranker_enabled", True)),
            milvus_password=os.environ.get("MILVUS_PASSWORD", ""),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        )

        ingress_cfg = IngressConfig(
            domain=os.environ.get("DOCS_AGENT_DOMAIN")
            or ingress_raw.get("domain", ""),
            tls_mode=os.environ.get("DOCS_AGENT_TLS_MODE")
            or ingress_raw.get("tls_mode", "self-signed"),
            acme_email=os.environ.get("DOCS_AGENT_ACME_EMAIL")
            or ingress_raw.get("acme_email", ""),
            enable_jwt_guardrails=bool(
                ingress_raw.get("enable_jwt_guardrails", True)
            ),
            enable_oauth2_proxy=bool(ingress_raw.get("enable_oauth2_proxy", False)),
            lb_min_bandwidth_mbps=int(ingress_raw.get("lb_min_bandwidth_mbps", 10)),
            lb_max_bandwidth_mbps=int(ingress_raw.get("lb_max_bandwidth_mbps", 100)),
        )

        default_website_path = "../kubeflow-website/website"
        for candidate in ("../kubeflow-website/website", "../../kubeflow-website/website"):
            if (repo_root / candidate).resolve().exists():
                default_website_path = candidate
                break

        netlify_cfg = NetlifyConfig(
            mode=os.environ.get("NETLIFY_DEPLOY_MODE")
            or netlify_raw.get("mode", "website"),
            website_repo_path=os.environ.get("KUBEFLOW_WEBSITE_PATH")
            or (
                default_website_path
                if netlify_raw.get("website_repo_path") == "../kubeflow-website/website"
                else netlify_raw.get("website_repo_path", default_website_path)
            ),
            extra_cors_origins=list(
                netlify_raw.get(
                    "extra_cors_origins",
                    [
                        "https://www.kubeflow.org",
                        "http://localhost:1313",
                        "http://localhost:8080",
                    ],
                )
            ),
            auth_token=os.environ.get("NETLIFY_AUTH_TOKEN", ""),
            site_id=os.environ.get("NETLIFY_SITE_ID", ""),
            site_url=os.environ.get("NETLIFY_SITE_URL", ""),
        )

        return cls(
            oci=oci_cfg,
            cluster=cluster_cfg,
            platform=platform_cfg,
            llm=llm_cfg,
            mcp=mcp_cfg,
            ingress=ingress_cfg,
            netlify=netlify_cfg,
            repo_root=repo_root,
        )

    def discover_upstream_defaults(self) -> Dict[str, Any]:
        """Introspect Helm values.yaml and Terraform variables.tf in the repo to prevent CLI drift."""
        import re
        import yaml

        discovered: Dict[str, Any] = {}

        # 1. docs-agent chart values.yaml
        da_values_path = self.charts_dir / "docs-agent" / "values.yaml"
        if da_values_path.exists():
            da = yaml.safe_load(da_values_path.read_text(encoding="utf-8")) or {}
            mcp_sec = da.get("mcp", {})
            img_sec = mcp_sec.get("image", {})
            cfg_sec = mcp_sec.get("config", {})
            pipe_sec = da.get("pipelines", {})
            if img_sec.get("repository"):
                discovered["mcp.image_repository"] = str(img_sec["repository"])
            if img_sec.get("tag"):
                discovered["mcp.image_tag"] = str(img_sec["tag"])
            if cfg_sec.get("docsCollection"):
                discovered["mcp.collection_docs"] = str(cfg_sec["docsCollection"])
            if cfg_sec.get("issuesCollection"):
                discovered["mcp.collection_issues"] = str(cfg_sec["issuesCollection"])
            if cfg_sec.get("codeCollection"):
                discovered["mcp.collection_code"] = str(cfg_sec["codeCollection"])
            if pipe_sec.get("versionName"):
                discovered["pipelines.version_name"] = str(pipe_sec["versionName"])
            reg_img = (pipe_sec.get("registerJob") or {}).get("image")
            if reg_img:
                discovered["pipelines.register_job_image"] = str(reg_img)
            agents_sec = (da.get("kagent") or {}).get("agents") or {}
            enabled_agents = [
                str(spec.get("name"))
                for _role, spec in agents_sec.items()
                if isinstance(spec, dict) and spec.get("enabled", True) and spec.get("name")
            ]
            if enabled_agents:
                discovered["kagent.enabled_agents"] = ",".join(enabled_agents)

        # 2. gateway-guardrails chart values.yaml
        gw_values_path = self.charts_dir / "gateway-guardrails" / "values.yaml"
        if gw_values_path.exists():
            gw = yaml.safe_load(gw_values_path.read_text(encoding="utf-8")) or {}
            sa = gw.get("sessionAuth", {})
            if sa.get("image"):
                discovered["gateway.session_issuer_image"] = str(sa["image"])

        # 3. llm-runtime chart values.yaml
        llm_values_path = self.charts_dir / "llm-runtime" / "values.yaml"
        if llm_values_path.exists():
            lr = yaml.safe_load(llm_values_path.read_text(encoding="utf-8")) or {}
            if lr.get("model"):
                discovered["llm.chart_default_model"] = str(lr["model"])
            catalog = lr.get("catalog") or {}
            if isinstance(catalog, dict) and catalog:
                discovered["llm.catalog_profiles"] = ",".join(sorted(catalog.keys()))
            gemma_id = (lr.get("gemma") or (catalog.get("gemma") or {})).get("modelId")
            if gemma_id:
                discovered["llm.gemma_model_id"] = str(gemma_id)

        # 4. Terraform variables.tf (platform + OKE cluster)
        tf_var_block = re.compile(
            r'variable\s+"(?P<name>[a-zA-Z0-9_]+)"\s*\{(?P<body>[^}]*)\}',
            re.MULTILINE | re.DOTALL,
        )
        tf_default_re = re.compile(r'default\s*=\s*"(?P<val>[^"]+)"')
        for tf_file in (
            self.platform_tf_dir / "variables.tf",
            self.oke_tf_dir / "variables.tf",
        ):
            if not tf_file.exists():
                continue
            text = tf_file.read_text(encoding="utf-8")
            for match in tf_var_block.finditer(text):
                vname = match.group("name")
                dmatch = tf_default_re.search(match.group("body"))
                if dmatch:
                    discovered[f"tf.{vname}"] = dmatch.group("val")

        # 5. Argo CD production pin & app inventory (deploy/argocd/apps/*.yaml)
        argo_apps_dir = self.repo_root / "deploy" / "argocd" / "apps"
        if argo_apps_dir.exists():
            app_names = sorted(p.stem for p in argo_apps_dir.glob("*.yaml"))
            if app_names:
                discovered["argocd.apps"] = ",".join(app_names)
        argo_da_path = argo_apps_dir / "docs-agent.yaml"
        if argo_da_path.exists():
            argo_text = argo_da_path.read_text(encoding="utf-8")
            pmatch = re.search(
                r"targetRevision:\s*([0-9a-f]{7,40})\s*#\s*prod-revision", argo_text
            )
            if pmatch:
                discovered["argocd.prod_revision"] = pmatch.group(1)[:7]

        return discovered

    def discover_enabled_kagent_agents(self) -> List[str]:
        """Dynamically discover all enabled Kagent Agent CR names from docs-agent/values.yaml."""
        import yaml

        da_values_path = self.charts_dir / "docs-agent" / "values.yaml"
        if da_values_path.exists():
            da = yaml.safe_load(da_values_path.read_text(encoding="utf-8")) or {}
            agents_sec = (da.get("kagent") or {}).get("agents") or {}
            names = [
                str(spec.get("name"))
                for _role, spec in agents_sec.items()
                if isinstance(spec, dict) and spec.get("enabled", True) and spec.get("name")
            ]
            if names:
                return names
        return [
            "kubeflow-docs-agent",
            "kubeflow-docs-agent-think",
            "kubeflow-debug-agent",
        ]

    def discover_argocd_apps(self) -> List[Dict[str, str]]:
        """Return inventory of upstream Argo CD Application manifests in deploy/argocd/apps/."""
        import yaml

        apps: List[Dict[str, str]] = []
        argo_apps_dir = self.repo_root / "deploy" / "argocd" / "apps"
        if not argo_apps_dir.exists():
            return apps
        for p in sorted(argo_apps_dir.glob("*.yaml")):
            try:
                doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                src = (doc.get("spec") or {}).get("source") or {}
                apps.append(
                    {
                        "name": str((doc.get("metadata") or {}).get("name") or p.stem),
                        "path": str(src.get("path") or ""),
                        "targetRevision": str(src.get("targetRevision") or ""),
                    }
                )
            except Exception:
                apps.append({"name": p.stem, "path": "", "targetRevision": ""})
        return apps

    def resolve_latest_ghcr_tag(self, track: str = "main") -> Optional[str]:
        """Walk recent commits on main (or prod-revision) and return the newest tag published on GHCR for both MCP and session-issuer."""
        import json
        import subprocess
        import urllib.request

        discovered = self.discover_upstream_defaults()
        candidates: List[str] = []

        if track == "prod" and discovered.get("argocd.prod_revision"):
            candidates.append(str(discovered["argocd.prod_revision"]))

        git_res = subprocess.run(
            ["git", "-C", str(self.repo_root), "log", "-n", "25", "--format=%h"],
            capture_output=True,
            text=True,
            check=False,
        )
        if git_res.returncode == 0:
            for line in git_res.stdout.splitlines():
                sha = line.strip()[:7]
                if sha and sha not in candidates:
                    candidates.append(sha)

        if discovered.get("argocd.prod_revision"):
            pr = str(discovered["argocd.prod_revision"])
            if pr not in candidates:
                candidates.append(pr)
        if self.mcp.image_tag and self.mcp.image_tag not in candidates:
            candidates.append(self.mcp.image_tag)

        def _ghcr_tag_exists(repo_name: str, tag: str) -> bool:
            try:
                tok_url = f"https://ghcr.io/token?scope=repository:kubeflow/{repo_name}:pull"
                with urllib.request.urlopen(tok_url, timeout=8) as tresp:
                    token = json.loads(tresp.read().decode("utf-8")).get("token", "")
                if not token:
                    return False
                man_url = f"https://ghcr.io/v2/kubeflow/{repo_name}/manifests/{tag}"
                req = urllib.request.Request(
                    man_url,
                    method="HEAD",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json",
                    },
                )
                with urllib.request.urlopen(req, timeout=8) as mresp:
                    return mresp.status == 200
            except Exception:
                return False

        for candidate in candidates:
            if _ghcr_tag_exists("mcp-kubeflow-docs", candidate) and _ghcr_tag_exists(
                "kubeflow-session-issuer", candidate
            ):
                return candidate
        return None

    def sync_with_upstream(self, update_docs_toml: bool = True) -> List[Dict[str, str]]:
        """Sync in-memory DocsConfig (and optionally docs.toml) with upstream Helm/Terraform defaults."""
        discovered = self.discover_upstream_defaults()
        changes: List[Dict[str, str]] = []

        mapping = [
            ("mcp.image_repository", self.mcp, "image_repository"),
            ("mcp.image_tag", self.mcp, "image_tag"),
            ("mcp.collection_docs", self.mcp, "collection_docs"),
            ("mcp.collection_issues", self.mcp, "collection_issues"),
            ("mcp.collection_code", self.mcp, "collection_code"),
            ("tf.cert_manager_version", self.platform, "cert_manager_version"),
            ("tf.istio_version", self.platform, "istio_version"),
            ("tf.knative_version", self.platform, "knative_version"),
            ("tf.kserve_version", self.platform, "kserve_version"),
            ("tf.kfp_version", self.platform, "kfp_version"),
            ("tf.milvus_version", self.platform, "milvus_version"),
            ("tf.embeddings_model_id", self.platform, "embeddings_model_id"),
            ("tf.reranker_model_id", self.platform, "reranker_model_id"),
            ("tf.kubernetes_version", self.cluster, "kubernetes_version"),
        ]

        toml_path = self.repo_root / "docs.toml"
        toml_text = toml_path.read_text(encoding="utf-8") if toml_path.exists() else ""

        for key, obj, attr in mapping:
            if key not in discovered:
                continue
            upstream_val = str(discovered[key])
            current_val = str(getattr(obj, attr))
            if current_val != upstream_val:
                changes.append(
                    {
                        "field": f"{attr}",
                        "source": key,
                        "old": current_val,
                        "new": upstream_val,
                    }
                )
                setattr(obj, attr, upstream_val)
                if update_docs_toml and toml_text:
                    import re

                    pattern = re.compile(
                        rf'^(\s*{re.escape(attr)}\s*=\s*)"{re.escape(current_val)}"',
                        re.MULTILINE,
                    )
                    toml_text = pattern.sub(rf'\1"{upstream_val}"', toml_text)

        if update_docs_toml and changes and toml_path.exists():
            toml_path.write_text(toml_text, encoding="utf-8")

        return changes
