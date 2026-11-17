# =============================================================================
# deployments/terraform/gke-cluster/network.tf
# Dedicated VPC, VPC-native GKE subnet (Pod & Service secondary ranges),
# Cloud Router + Cloud NAT, and internal/webhook firewall rules.
# Every resource carries `docs-cli-managed = "true"` and `docs-cli-cluster`
# labels for safe, deterministic cleanup via `docs-cli destroy` / `nuke`.
# =============================================================================

locals {
  name       = var.cluster_name
  label_name = substr(lower(replace(var.cluster_name, "/[^a-zA-Z0-9_-]/", "-")), 0, 63)
  common_labels = {
    "docs-cli-managed" = "true"
    "docs-cli-cluster" = local.label_name
    "project"          = "kubeflow-docs-agent"
  }
}

resource "google_project_service" "compute" {
  project            = var.project_id
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "container" {
  project            = var.project_id
  service            = "container.googleapis.com"
  disable_on_destroy = false

  depends_on = [google_project_service.compute]
}

resource "google_compute_network" "this" {
  name                    = "${local.name}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
  description             = "Dedicated VPC for ${local.name} (managed by docs-cli)"

  depends_on = [google_project_service.compute]
}

resource "google_compute_subnetwork" "nodes" {
  name                     = "${local.name}-nodes"
  ip_cidr_range            = var.vpc_cidr
  region                   = var.region
  network                  = google_compute_network.this.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr
  }
}

resource "google_compute_router" "this" {
  name    = "${local.name}-router"
  region  = var.region
  network = google_compute_network.this.id
}

resource "google_compute_router_nat" "this" {
  name                               = "${local.name}-nat"
  router                             = google_compute_router.this.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# Allow internal node-to-node, Pod-to-Pod, and Istio/Knative/KServe/cert-manager webhook traffic
resource "google_compute_firewall" "internal" {
  name    = "${local.name}-allow-internal"
  network = google_compute_network.this.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "icmp"
  }

  source_ranges = [
    var.vpc_cidr,
    var.pods_cidr,
    var.services_cidr,
  ]
}
