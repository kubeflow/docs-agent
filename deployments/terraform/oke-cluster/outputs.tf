output "cluster_id" {
  description = "OCID of the OKE cluster."
  value       = oci_containerengine_cluster.this.id
}

output "cluster_name" {
  description = "Name of the OKE cluster."
  value       = oci_containerengine_cluster.this.name
}

output "compartment_ocid" {
  description = "Compartment OCID holding the cluster."
  value       = var.compartment_ocid
}

output "region" {
  description = "OCI region of the cluster."
  value       = var.region
}

output "kubernetes_version" {
  description = "Kubernetes version deployed."
  value       = local.k8s_version
}

output "vcn_id" {
  description = "OCID of the cluster VCN."
  value       = oci_core_vcn.this.id
}

output "cpu_pool_id" {
  description = "OCID of the CPU node pool."
  value       = oci_containerengine_node_pool.cpu.id
}

output "gpu_pool_id" {
  description = "OCID of the GPU node pool (null when enable_gpu_pool = false)."
  value       = var.enable_gpu_pool ? oci_containerengine_node_pool.gpu[0].id : null
}

output "gpu_enabled" {
  description = "Whether the GPU node pool is enabled."
  value       = var.enable_gpu_pool
}

output "kubeconfig_command" {
  description = "Command to generate kubeconfig via OCI CLI."
  value       = "oci ce cluster create-kubeconfig --cluster-id ${oci_containerengine_cluster.this.id} --file $HOME/.kube/config --region ${var.region} --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT --profile ${var.oci_profile}"
}
