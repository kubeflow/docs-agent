"""Google Cloud (GCP / GKE) authentication, project discovery, and zone helper for docs-cli."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import DocsConfig


@dataclass
class GcpSession:
    project_id: str
    region: str
    zone: str
    credentials_file: Optional[str] = None

    @classmethod
    def from_docs_config(cls, cfg: DocsConfig) -> "GcpSession":
        """Resolve GCP project, region/zone, and credentials from env vars, Service Account JSON, or `gcloud` ADC."""
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

    def terraform_env(self) -> Dict[str, str]:
        """Return environment variables for the HashiCorp Google Terraform provider."""
        env = {
            "GOOGLE_PROJECT": self.project_id,
            "GOOGLE_REGION": self.region,
            "TF_VAR_project_id": self.project_id,
            "TF_VAR_region": self.region,
            "TF_VAR_zone": self.zone,
        }
        if self.credentials_file:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
            env["TF_VAR_credentials_file"] = self.credentials_file
        return env

    def gcloud_env(self) -> Dict[str, str]:
        """Return environment variables for `gcloud` CLI invocations."""
        env = dict(os.environ)
        env["CLOUDSDK_CORE_PROJECT"] = self.project_id
        env["USE_GKE_GCLOUD_AUTH_PLUGIN"] = "True"
        if self.credentials_file:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
        return env

    def ensure_gcloud_auth_if_service_account(self) -> None:
        """Activate service account in `gcloud` if an explicit service_account JSON key file is configured."""
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
