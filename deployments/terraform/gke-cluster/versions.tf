terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
  }
}

# Supports:
# 1. Active `gcloud` CLI session (`GOOGLE_OAUTH_ACCESS_TOKEN` minted automatically by `docs-cli`)
# 2. Application Default Credentials (`gcloud auth application-default login` / Workload Identity)
# 3. Explicit Service Account JSON key (`GOOGLE_APPLICATION_CREDENTIALS` or `credentials_file`).
# `user_project_override = true` ensures newly created projects use their own enabled APIs & quota.
provider "google" {
  project               = var.project_id
  billing_project       = var.project_id
  user_project_override = true
  region                = var.region
  credentials           = var.credentials_file != null && var.credentials_file != "" ? file(pathexpand(var.credentials_file)) : null
}
