terraform {
  required_version = ">= 1.5.0"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 6.0.0"
    }
  }
}

# Supports both:
# 1. Secure environment variables (TF_VAR_tenancy_ocid, TF_VAR_user_ocid,
#    TF_VAR_fingerprint, TF_VAR_private_key_path, or OCI_* mapped by docs-cli)
# 2. Standard ~/.oci/config profile when the explicit variables are null.
provider "oci" {
  region               = var.region
  tenancy_ocid         = var.tenancy_ocid
  user_ocid            = var.user_ocid
  fingerprint          = var.fingerprint
  private_key_path     = var.private_key_path
  private_key_password = var.private_key_password
  config_file_profile  = var.tenancy_ocid == null ? var.oci_profile : null
}
