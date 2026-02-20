resource "oci_core_vcn" "k8s_vcn" {
  cidr_block     = "10.0.0.0/16"
  compartment_id = var.compartment_id
  display_name   = "k8s_vcn"
}

resource "oci_core_subnet" "k8s_subnet" {
  cidr_block     = "10.0.1.0/24"
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.k8s_vcn.id
  display_name   = "k8s_subnet"
}

resource "oci_containerengine_cluster" "k8s_cluster" {
  compartment_id     = var.compartment_id
  kubernetes_version = "v1.27.2"
  name               = "kubeflow-gsoc-cluster"
  vcn_id             = oci_core_vcn.k8s_vcn.id

  options {
    service_lb_subnet_ids = [oci_core_subnet.k8s_subnet.id]
  }
}

resource "oci_containerengine_node_pool" "k8s_node_pool" {
  cluster_id         = oci_containerengine_cluster.k8s_cluster.id
  compartment_id     = var.compartment_id
  kubernetes_version = "v1.27.2"
  name               = "k8s_node_pool"
  node_shape         = "VM.Standard.E4.Flex"

  node_config_details {
    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.k8s_subnet.id
    }
    size = 3
  }

  node_shape_config {
    memory_in_gbs = 64
    ocpus         = 4
  }
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_id
}
