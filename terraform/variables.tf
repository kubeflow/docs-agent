variable "tenancy_ocid" {}
variable "user_ocid" {}
variable "fingerprint" {}
variable "private_key_path" {}
variable "region" {}
variable "compartment_id" {}

output "oke_cluster_id" {
  value = oci_containerengine_cluster.k8s_cluster.id
}
