terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
  }
}

# Supports both:
# 1. Google Cloud Application Default Credentials (`gcloud auth application-default login`
#    or Workload Identity Federation in GitHub Actions) when `credentials_file` is null/empty
# 2. Explicit Service Account JSON key (`GOOGLE_APPLICATION_CREDENTIALS` or `credentials_file`).
provider "google" {
  project     = var.project_id
  region      = var.region
  credentials = var.credentials_file != null && var.credentials_file != "" ? file(pathexpand(var.credentials_file)) : null
}
