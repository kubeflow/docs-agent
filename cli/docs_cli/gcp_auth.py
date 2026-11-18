"""Google Cloud (GCP / GKE) authentication, SDK discovery, API bootstrap, and quota helpers for docs-cli."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from .config import DocsConfig

console = Console()

REQUIRED_GCP_APIS: List[str] = [
    "serviceusage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "aiplatform.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
]


def ensure_gcloud_on_path() -> Optional[str]:
    """Ensure `gcloud` and `gke-gcloud-auth-plugin` are on `os.environ['PATH']`, discovering common macOS/Linux SDK locations."""
    os.environ.setdefault("USE_GKE_GCLOUD_AUTH_PLUGIN", "True")
    existing = shutil.which("gcloud")
    if existing:
        return existing

    home = Path.home()
    candidates: List[Path] = []

    # 1. Inspect shell rc files for `.../google-cloud-sdk.../path.zsh.inc` or `path.bash.inc`
    path_inc_re = re.compile(r"""['"]([^'"]*google-cloud-sdk[^'"]*)/path\.[a-z]+\.inc['"]""")
    for rc_name in (".zshrc", ".zprofile", ".bashrc", ".bash_profile"):
        rc_path = home / rc_name
        try:
            if rc_path.exists():
                for match in path_inc_re.finditer(rc_path.read_text(encoding="utf-8", errors="ignore")):
                    candidates.append(Path(match.group(1)) / "bin")
        except Exception:
            pass

    # 2. Check standard Homebrew, user home, and Downloads install roots
    for pattern in (
        "/opt/homebrew/share/google-cloud-sdk/bin",
        "/opt/homebrew/Caskroom/google-cloud-sdk/*/google-cloud-sdk/bin",
        "/usr/local/share/google-cloud-sdk/bin",
        "/usr/local/Caskroom/google-cloud-sdk/*/google-cloud-sdk/bin",
        str(home / "google-cloud-sdk" / "bin"),
        str(home / "Downloads" / "google-cloud-sdk*" / "bin"),
        "/snap/bin",
    ):
        try:
            for matched in sorted(glob.glob(pattern), reverse=True):
                candidates.append(Path(matched))
        except Exception:
            pass

    for bin_dir in candidates:
        try:
            gcloud_bin = bin_dir / "gcloud"
            if gcloud_bin.exists() and os.access(gcloud_bin, os.X_OK):
                cur_path = os.environ.get("PATH", "")
                if str(bin_dir) not in cur_path.split(os.pathsep):
                    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{cur_path}"
                return str(gcloud_bin)
        except Exception:
            continue

    return None


@dataclass
class GcpSession:
    project_id: str
    region: str
    zone: str
    credentials_file: Optional[str] = None

    @classmethod
    def from_docs_config(cls, cfg: DocsConfig) -> "GcpSession":
        """Resolve GCP project, region/zone, and credentials from env vars, Service Account JSON, or `gcloud`."""
        ensure_gcloud_on_path()

        creds_raw = (
            cfg.gcp.credentials_file
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("GCP_CREDENTIALS_FILE")
            or ""
        ).strip()
        creds_path: Optional[str] = None
        sa_project_id = ""
        if creds_raw:
            expanded = Path(creds_raw).expanduser()
            if not expanded.exists():
                raise FileNotFoundError(
                    f"GCP credentials file not found at {expanded}. Check GOOGLE_APPLICATION_CREDENTIALS."
                )
            creds_path = str(expanded.resolve())
            try:
                sa_doc = json.loads(expanded.read_text(encoding="utf-8"))
                sa_project_id = str(sa_doc.get("project_id") or "")
            except Exception:
                pass

        project_id = (
            cfg.gcp.project_id
            or os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("CLOUDSDK_CORE_PROJECT")
            or sa_project_id
            or ""
        ).strip()

        if not project_id and shutil.which("gcloud"):
            res = subprocess.run(
                ["gcloud", "config", "get-value", "project", "--quiet"],
                capture_output=True,
                text=True,
                check=False,
            )
            val = res.stdout.strip()
            if res.returncode == 0 and val and val != "(unset)":
                project_id = val

        if not project_id:
            adc_file = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            if adc_file.exists():
                try:
                    adc_doc = json.loads(adc_file.read_text(encoding="utf-8"))
                    project_id = str(adc_doc.get("quota_project_id") or "")
                except Exception:
                    pass

        if not project_id:
            raise RuntimeError(
                "Could not determine GCP Project ID. Set GCP_PROJECT_ID (or [gcp].project_id in docs.toml) "
                "or run `gcloud config set project <PROJECT_ID>`."
            )

        region = (
            os.environ.get("GCP_REGION")
            or os.environ.get("CLOUDSDK_COMPUTE_REGION")
            or cfg.gcp.region
            or "us-central1"
        ).strip()
        zone = (
            os.environ.get("GCP_ZONE")
            or os.environ.get("CLOUDSDK_COMPUTE_ZONE")
            or cfg.gcp.zone
            or f"{region}-a"
        ).strip()

        return cls(
            project_id=project_id,
            region=region,
            zone=zone,
            credentials_file=creds_path,
        )

    def mint_gcloud_access_token(self) -> Optional[str]:
        """Mint a fresh 60-minute OAuth2 access token from active `gcloud` CLI session (`--force-auth-refresh`)."""
        ensure_gcloud_on_path()
        if not shutil.which("gcloud"):
            return None
        for cmd in (
            [
                "gcloud",
                "config",
                "config-helper",
                "--force-auth-refresh",
                "--format=value(credential.access_token)",
                "--quiet",
            ],
            ["gcloud", "auth", "print-access-token", "--quiet"],
        ):
            res = subprocess.run(
                cmd,
                env=self.gcloud_env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            tok = res.stdout.strip()
            if res.returncode == 0 and tok:
                return tok
        return None

    def is_adc_valid(self) -> bool:
        """Check whether Application Default Credentials can currently mint a valid token."""
        ensure_gcloud_on_path()
        if not shutil.which("gcloud"):
            return False
        res = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token", "--quiet"],
            env=self.gcloud_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        return res.returncode == 0 and bool(res.stdout.strip())

    def terraform_env(self) -> Dict[str, str]:
        """Return environment variables for the HashiCorp Google Terraform provider.

        Automatically falls back to `gcloud auth print-access-token` (`GOOGLE_OAUTH_ACCESS_TOKEN`)
        when Application Default Credentials (`gcloud auth application-default login`) are unset or expired.
        """
        ensure_gcloud_on_path()
        env: Dict[str, str] = {
            "GOOGLE_PROJECT": self.project_id,
            "GOOGLE_BILLING_PROJECT": self.project_id,
            "USER_PROJECT_OVERRIDE": "true",
            "GOOGLE_REGION": self.region,
            "TF_VAR_project_id": self.project_id,
            "TF_VAR_region": self.region,
            "TF_VAR_zone": self.zone,
        }
        if self.credentials_file:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
            env["TF_VAR_credentials_file"] = self.credentials_file
        elif not os.environ.get("PYTEST_CURRENT_TEST"):
            if not self.is_adc_valid():
                tok = self.mint_gcloud_access_token()
                if tok:
                    env["GOOGLE_OAUTH_ACCESS_TOKEN"] = tok
        return env

    def gcloud_env(self) -> Dict[str, str]:
        """Return environment variables for `gcloud` CLI invocations."""
        ensure_gcloud_on_path()
        env = dict(os.environ)
        env["CLOUDSDK_CORE_PROJECT"] = self.project_id
        env["USE_GKE_GCLOUD_AUTH_PLUGIN"] = "True"
        if self.credentials_file:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
        return env

    def ensure_gcloud_auth_if_service_account(self) -> None:
        """Activate service account in `gcloud` if an explicit service_account JSON key file is configured."""
        ensure_gcloud_on_path()
        if not self.credentials_file or not shutil.which("gcloud"):
            return
        try:
            doc = json.loads(Path(self.credentials_file).read_text(encoding="utf-8"))
            if doc.get("type") == "service_account":
                subprocess.run(
                    [
                        "gcloud",
                        "auth",
                        "activate-service-account",
                        f"--key-file={self.credentials_file}",
                        f"--project={self.project_id}",
                        "--quiet",
                    ],
                    env=self.gcloud_env(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
        except Exception:
            pass


def bootstrap_gcp_project_apis(gcp_session: GcpSession) -> None:
    """Enable all required Google Cloud APIs on a (potentially brand-new) GCP project and wait for Compute readiness."""
    ensure_gcloud_on_path()
    if not shutil.which("gcloud"):
        return

    gcp_session.ensure_gcloud_auth_if_service_account()
    env = gcp_session.gcloud_env()
    proj = gcp_session.project_id

    console.print(
        f"[bold cyan]Enabling required Google Cloud APIs on project '{proj}' ({len(REQUIRED_GCP_APIS)} services)...[/bold cyan]"
    )
    # Enable foundational APIs first so Service Usage & Resource Manager are ready
    subprocess.run(
        [
            "gcloud",
            "services",
            "enable",
            "serviceusage.googleapis.com",
            "cloudresourcemanager.googleapis.com",
            f"--project={proj}",
            "--quiet",
        ],
        env=env,
        check=True,
    )
    # Enable full GKE + Compute + IAM + Observability + Storage API set
    subprocess.run(
        [
            "gcloud",
            "services",
            "enable",
            *REQUIRED_GCP_APIS,
            f"--project={proj}",
            "--quiet",
        ],
        env=env,
        check=True,
    )

    # Wait until Compute Engine API & default service agents finish initializing on brand-new projects
    deadline = time.time() + 120
    while time.time() < deadline:
        probe = subprocess.run(
            [
                "gcloud",
                "compute",
                "regions",
                "describe",
                gcp_session.region,
                f"--project={proj}",
                "--format=value(status)",
                "--quiet",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and "UP" in probe.stdout:
            console.print(
                f"[green]✓ Enabled all {len(REQUIRED_GCP_APIS)} GCP APIs and verified Compute Engine readiness in {gcp_session.region}[/green]"
            )
            return
        time.sleep(5)


def inspect_gcp_gpu_quotas(gcp_session: GcpSession) -> Dict[str, float]:
    """Return available GPU quotas (`metric -> remaining_limit`) in `gcp_session.region` plus global `GPUS_ALL_REGIONS`."""
    ensure_gcloud_on_path()
    quotas: Dict[str, float] = {}
    if not shutil.which("gcloud"):
        return quotas
    env = gcp_session.gcloud_env()
    proj = gcp_session.project_id

    for cmd in (
        ["gcloud", "compute", "project-info", "describe", f"--project={proj}", "--format=json", "--quiet"],
        ["gcloud", "compute", "regions", "describe", gcp_session.region, f"--project={proj}", "--format=json", "--quiet"],
    ):
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            try:
                data: Dict[str, Any] = json.loads(res.stdout)
                for q in data.get("quotas", []):
                    metric = str(q.get("metric", ""))
                    if any(
                        tok in metric
                        for tok in ("GPU", "A100", "L4", "T4", "CPU", "IN_USE_ADDRESSES", "SSD_TOTAL_GB", "DISKS_TOTAL_GB")
                    ):
                        limit = float(q.get("limit", 0.0))
                        usage = float(q.get("usage", 0.0))
                        quotas[metric] = max(0.0, limit - usage)
            except Exception:
                pass
    return quotas
