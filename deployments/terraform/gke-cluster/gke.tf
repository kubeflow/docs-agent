# =============================================================================
# deployments/terraform/gke-cluster/gke.tf
# Provisions a VPC-native GKE Standard cluster, CPU node pool, and optional
# NVIDIA L4 GPU node pool (with automatic GKE NVIDIA driver installation).
# =============================================================================

data "google_container_engine_versions" "zone" {
  location       = var.zone
  version_prefix = var.kubernetes_version != "" ? trimprefix(var.kubernetes_version, "v") : null

  depends_on = [google_project_service.container]
}

locals {
  # Only pin min_master_version when explicitly set to a GKE-available prefix;
  # otherwise let GKE's REGULAR release channel pick the latest stable version.
  resolved_k8s_version = (
    var.kubernetes_version != "" && length(data.google_container_engine_versions.zone.valid_master_versions) > 0
    ? data.google_container_engine_versions.zone.latest_master_version
    : null
  )
  effective_gpu_zones = length(var.gpu_zones) > 0 ? var.gpu_zones : [var.zone]
}

resource "google_container_cluster" "this" {
  name     = local.name
  location = var.zone

  # Manage CPU and GPU node pools as independent resources
  remove_default_node_pool = true
  initial_node_count       = 1

  # Allow clean automated teardown via `docs-cli destroy --yes`
  deletion_protection = false

  network         = google_compute_network.this.id
  subnetwork      = google_compute_subnetwork.nodes.id
  networking_mode = "VPC_NATIVE"

  min_master_version = local.resolved_k8s_version

  release_channel {
    channel = "REGULAR"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = var.api_allowed_cidr
      display_name = "docs-cli-allowed"
    }
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  resource_labels = local.common_labels

  depends_on = [
    google_project_service.container,
    google_compute_router_nat.this,
  ]
}

# --- CPU Node Pool -----------------------------------------------------------

resource "google_container_node_pool" "cpu" {
  name       = "${local.name}-cpu-pool"
  location   = var.zone
  cluster    = google_container_cluster.this.name
  node_count = var.cpu_node_count

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.cpu_machine_type
    disk_size_gb = var.cpu_disk_size_gb
    disk_type    = var.cpu_disk_type
    image_type   = "COS_CONTAINERD"

    labels = merge(local.common_labels, {
      "docs-agent-pool" = "cpu"
    })

    resource_labels = local.common_labels

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

# --- GPU Node Pool (Optional: enabled by default, skipped with --no-gpu) -----

resource "google_container_node_pool" "gpu" {
  count          = var.enable_gpu_pool ? 1 : 0
  name           = "${local.name}-gpu-pool"
  location       = var.zone
  node_locations = local.effective_gpu_zones
  cluster        = google_container_cluster.this.name
  node_count     = var.gpu_node_count

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.gpu_machine_type
    disk_size_gb = var.gpu_disk_size_gb
    disk_type    = var.gpu_disk_type
    image_type   = "COS_CONTAINERD"

    guest_accelerator {
      type  = var.gpu_accelerator_type
      count = var.gpu_accelerator_count
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    # Matches `llm-runtime` chart's `placement.nodeSelector: {"nvidia.com/gpu": "true"}`
    labels = merge(local.common_labels, {
      "docs-agent-pool" = "gpu"
      "nvidia.com/gpu"  = "true"
    })

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    resource_labels = local.common_labels

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}
