"""OCI authentication and client factory for docs-cli (env vars + ~/.oci/config)."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DocsConfig


@dataclass
class OciSession:
    config: dict
    tenancy_id: str
    user_id: str
    region: str
    fingerprint: str
    key_file: str
    passphrase: Optional[str]

    @classmethod
    def from_docs_config(cls, cfg: DocsConfig) -> "OciSession":
        """Load OCI credentials from environment variables first, falling back to ~/.oci/config."""
        env_tenancy = cfg.oci.tenancy_ocid or os.environ.get("OCI_TENANCY", "")
        if env_tenancy:
            raw = {
                "tenancy": env_tenancy,
                "user": cfg.oci.user_ocid or os.environ.get("OCI_USER", ""),
                "fingerprint": cfg.oci.fingerprint
                or os.environ.get("OCI_FINGERPRINT", ""),
                "key_file": os.path.expanduser(
                    cfg.oci.private_key_path or os.environ.get("OCI_KEY_FILE", "")
                ),
                "passphrase": cfg.oci.private_key_passphrase
                or os.environ.get("OCI_KEY_PASSPHRASE"),
                "region": cfg.oci.region
                or os.environ.get("OCI_REGION", "us-chicago-1"),
            }
        else:
            config_file = os.environ.get(
                "OCI_CONFIG_FILE", str(Path.home() / ".oci" / "config")
            )
            profile = cfg.oci.profile or os.environ.get("OCI_PROFILE", "DEFAULT")
            try:
                import oci

                raw = oci.config.from_file(file_location=config_file, profile_name=profile)
            except ImportError:
                cp = configparser.ConfigParser()
                cp.read(os.path.expanduser(config_file), encoding="utf-8")
                if profile not in cp:
                    raise RuntimeError(f"OCI profile '{profile}' not found in {config_file}")
                sec = cp[profile]
                raw = {
                    "tenancy": sec.get("tenancy", "").strip(),
                    "user": sec.get("user", "").strip(),
                    "fingerprint": sec.get("fingerprint", "").strip(),
                    "key_file": os.path.expanduser(sec.get("key_file", "").strip()),
                    "passphrase": sec.get("pass_phrase"),
                    "region": sec.get("region", "us-ashburn-1").strip(),
                }
            if cfg.oci.region:
                raw["region"] = cfg.oci.region

        try:
            import oci

            oci.config.validate_config(raw)
        except ImportError:
            for req_key in ("tenancy", "user", "fingerprint", "key_file", "region"):
                if not raw.get(req_key):
                    raise ValueError(f"Missing required OCI config field '{req_key}'")

        return cls(
            config=raw,
            tenancy_id=raw["tenancy"],
            user_id=raw["user"],
            region=raw["region"],
            fingerprint=raw["fingerprint"],
            key_file=os.path.expanduser(raw["key_file"]),
            passphrase=raw.get("passphrase"),
        )

    def _for_region(self, region: Optional[str] = None) -> dict:
        c = dict(self.config)
        if region:
            c["region"] = region
        return c

    def identity(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.identity.IdentityClient(self._for_region(region))

    def compute(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.core.ComputeClient(self._for_region(region))

    def network(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.core.VirtualNetworkClient(self._for_region(region))

    def blockstorage(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.core.BlockstorageClient(self._for_region(region))

    def oke(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.container_engine.ContainerEngineClient(self._for_region(region))

    def load_balancer(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.load_balancer.LoadBalancerClient(self._for_region(region))

    def network_load_balancer(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.network_load_balancer.NetworkLoadBalancerClient(
            self._for_region(region)
        )

    def search(self, region: Optional[str] = None) -> Any:
        import oci

        return oci.resource_search.ResourceSearchClient(self._for_region(region))

    def resolve_compartment_id(self, cfg: DocsConfig) -> str:
        """Resolve target compartment OCID from OCI_COMPARTMENT_ID, native `oci` CLI, or SDK."""
        import json
        import shutil
        import subprocess

        if cfg.oci.compartment_id:
            return cfg.oci.compartment_id

        if shutil.which("oci"):
            cmd = [
                "oci",
                "iam",
                "compartment",
                "list",
                "--compartment-id",
                self.tenancy_id,
                "--compartment-id-in-subtree",
                "true",
                "--access-level",
                "ANY",
                "--all",
            ]
            if cfg.oci.profile and not cfg.oci.tenancy_ocid:
                cmd.extend(["--profile", cfg.oci.profile])
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout).get("data", [])
                for comp in data:
                    if (
                        comp.get("name") == cfg.oci.compartment_name
                        and comp.get("lifecycle-state") == "ACTIVE"
                    ):
                        return comp["id"]

        ident = self.identity()
        compartments = oci.pagination.list_call_get_all_results(
            ident.list_compartments,
            self.tenancy_id,
            compartment_id_in_subtree=True,
            access_level="ANY",
        ).data
        for comp in compartments:
            if (
                comp.name == cfg.oci.compartment_name
                and comp.lifecycle_state == "ACTIVE"
            ):
                return comp.id

        raise RuntimeError(
            f"Could not find an ACTIVE compartment named '{cfg.oci.compartment_name}' "
            f"under tenancy {self.tenancy_id}. Set OCI_COMPARTMENT_ID or create the compartment."
        )

    def terraform_env(self) -> Dict[str, str]:
        """Return TF_VAR_* environment variables for OCI provider authentication."""
        env = {
            "TF_VAR_tenancy_ocid": self.tenancy_id,
            "TF_VAR_user_ocid": self.user_id,
            "TF_VAR_fingerprint": self.fingerprint,
            "TF_VAR_private_key_path": self.key_file,
            "TF_VAR_region": self.region,
        }
        if self.passphrase:
            env["TF_VAR_private_key_password"] = self.passphrase
        return env
