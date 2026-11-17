output "cluster_id" {
  description = "Full resource ID of the GKE cluster."
  value       = google_container_cluster.this.id
}

output "cluster_name" {
  description = "Name of the GKE cluster."
  value       = google_container_cluster.this.name
}

output "project_id" {
  description = "GCP Project ID holding the cluster."
  value       = var.project_id
}

output "region" {
  description = "GCP region of the VPC and subnet."
  value       = var.region
}

output "location" {
  description = "GKE cluster location (zone)."
  value       = google_container_cluster.this.location
}

output "kubernetes_version" {
  description = "Kubernetes control-plane version deployed."
  value       = google_container_cluster.this.master_version
}

output "vpc_id" {
  description = "ID of the dedicated GKE VPC network."
  value       = google_compute_network.this.id
}

output "cpu_pool_id" {
  description = "ID of the CPU node pool."
  value       = google_container_node_pool.cpu.id
}

output "gpu_pool_id" {
  description = "ID of the GPU node pool (null when enable_gpu_pool = false)."
  value       = var.enable_gpu_pool ? google_container_node_pool.gpu[0].id : null
}

output "gpu_enabled" {
  description = "Whether the GPU node pool is enabled."
  value       = var.enable_gpu_pool
}

output "kubeconfig_command" {
  description = "Command to generate kubeconfig via gcloud CLI."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.this.name} --location ${google_container_cluster.this.location} --project ${var.project_id}"
}
