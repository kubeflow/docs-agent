# ---------------------------------------------------------------------------
# OKE Cluster Module — Oracle Kubernetes Engine
# ---------------------------------------------------------------------------
#
# Creates a VCN, subnets, networking resources (Internet Gateway, NAT
# Gateway, security lists, route tables), and an OKE cluster with CPU
# and optional GPU node pools for running the docs-agent stack.

terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 5.0"
    }
  }
}

# -- Variables --------------------------------------------------------------

variable "compartment_id" {
  type = string
}

variable "cluster_name" {
  type    = string
  default = "docs-agent-cluster"
}

variable "kubernetes_version" {
  type    = string
  default = "v1.30.0"
}

variable "vcn_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "region" {
  type = string
}

variable "cpu_node_shape" {
  type    = string
  default = "VM.Standard.E4.Flex"
}

variable "cpu_node_count" {
  type    = number
  default = 3
}

variable "gpu_node_shape" {
  type    = string
  default = "VM.GPU.A10.1"
}

variable "gpu_node_count" {
  type    = number
  default = 1
}

# -- VCN -------------------------------------------------------------------

resource "oci_core_vcn" "this" {
  compartment_id = var.compartment_id
  display_name   = "${var.cluster_name}-vcn"
  cidr_blocks    = [var.vcn_cidr]
}

# -- Internet Gateway (public subnets) ------------------------------------

resource "oci_core_internet_gateway" "this" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.cluster_name}-igw"
  enabled        = true
}

# -- NAT Gateway (private subnets) ----------------------------------------

resource "oci_core_nat_gateway" "this" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.cluster_name}-natgw"
}

# -- Route Tables ----------------------------------------------------------

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.cluster_name}-public-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.this.id
  }
}

resource "oci_core_route_table" "private" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.cluster_name}-private-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_nat_gateway.this.id
  }
}

# -- Security Lists --------------------------------------------------------

resource "oci_core_security_list" "api" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.cluster_name}-api-seclist"

  # Allow inbound HTTPS to the K8s API server
  ingress_security_rules {
    protocol = "6" # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 6443
      max = 6443
    }
  }

  # Allow all egress
  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

resource "oci_core_security_list" "workers" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${var.cluster_name}-worker-seclist"

  # Allow all traffic within the VCN
  ingress_security_rules {
    protocol = "all"
    source   = var.vcn_cidr
  }

  # Allow all egress (workers need to pull images, talk to API, etc.)
  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

# -- Subnets ---------------------------------------------------------------

resource "oci_core_subnet" "api" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.this.id
  display_name               = "${var.cluster_name}-api-subnet"
  cidr_block                 = cidrsubnet(var.vcn_cidr, 8, 0)
  prohibit_public_ip_on_vnic = false # public subnet for API endpoint
  route_table_id             = oci_core_route_table.public.id
  security_list_ids          = [oci_core_security_list.api.id]
}

resource "oci_core_subnet" "workers" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.this.id
  display_name               = "${var.cluster_name}-worker-subnet"
  cidr_block                 = cidrsubnet(var.vcn_cidr, 8, 1)
  prohibit_public_ip_on_vnic = true # private subnet for worker nodes
  route_table_id             = oci_core_route_table.private.id
  security_list_ids          = [oci_core_security_list.workers.id]
}

resource "oci_core_subnet" "lb" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.this.id
  display_name               = "${var.cluster_name}-lb-subnet"
  cidr_block                 = cidrsubnet(var.vcn_cidr, 8, 2)
  prohibit_public_ip_on_vnic = false # public subnet for load balancers
  route_table_id             = oci_core_route_table.public.id
  security_list_ids          = [oci_core_security_list.api.id]
}

# -- OKE Cluster -----------------------------------------------------------

resource "oci_containerengine_cluster" "this" {
  compartment_id     = var.compartment_id
  kubernetes_version = var.kubernetes_version
  name               = var.cluster_name
  vcn_id             = oci_core_vcn.this.id

  endpoint_config {
    is_public_ip_enabled = true
    subnet_id            = oci_core_subnet.api.id
  }

  options {
    service_lb_subnet_ids = [oci_core_subnet.lb.id]
  }
}

# -- CPU Node Pool ---------------------------------------------------------

resource "oci_containerengine_node_pool" "cpu" {
  compartment_id     = var.compartment_id
  cluster_id         = oci_containerengine_cluster.this.id
  kubernetes_version = var.kubernetes_version
  name               = "${var.cluster_name}-cpu-pool"

  node_shape = var.cpu_node_shape

  node_shape_config {
    ocpus         = 4
    memory_in_gbs = 32
  }

  node_config_details {
    size = var.cpu_node_count

    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.workers.id
    }
  }
}

# -- GPU Node Pool (optional) ----------------------------------------------

resource "oci_containerengine_node_pool" "gpu" {
  count = var.gpu_node_count > 0 ? 1 : 0

  compartment_id     = var.compartment_id
  cluster_id         = oci_containerengine_cluster.this.id
  kubernetes_version = var.kubernetes_version
  name               = "${var.cluster_name}-gpu-pool"
  node_shape         = var.gpu_node_shape

  # GPU shapes are fixed (not Flex), but node_shape_config is still
  # required by the OCI provider to acknowledge the shape's resources.
  node_shape_config {
    ocpus         = 16
    memory_in_gbs = 240
  }

  node_config_details {
    size = var.gpu_node_count

    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.workers.id
    }
  }
}

# -- Data sources ----------------------------------------------------------

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_id
}

# -- Outputs ---------------------------------------------------------------

output "cluster_id" {
  value = oci_containerengine_cluster.this.id
}

output "cluster_endpoint" {
  value = oci_containerengine_cluster.this.endpoints[0].public_endpoint
}

output "cluster_ca_certificate" {
  description = "Base64-decoded CA certificate for the OKE cluster."
  value       = base64decode(oci_containerengine_cluster.this.metadata[0].cluster_ca_certificate)
  sensitive   = true
}
