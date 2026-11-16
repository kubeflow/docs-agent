# =============================================================================
# deployments/terraform/oke-cluster/network.tf
# VCN layout for kubeflow/docs-agent on OKE:
#   10.0.0.0/28   api   (public)  - Kubernetes API endpoint (6443)
#   10.0.10.0/24  nodes (private) - CPU & GPU workers, egress via NAT + Service GW
#   10.0.20.0/24  lb    (public)  - Istio ingress LoadBalancer (80, 443)
# =============================================================================

locals {
  name       = trimspace(var.cluster_name)
  vcn_cidr   = "10.0.0.0/16"
  api_cidr   = "10.0.0.0/28"
  nodes_cidr = "10.0.10.0/24"
  lb_cidr    = "10.0.20.0/24"

  # Ownership tags stamped on every resource so `docs-cli nuke` can verify and
  # sweep orphaned resources safely without touching foreign resources.
  common_tags = {
    "docs-cli-managed" = "true"
    "docs-cli-cluster" = local.name
  }
}

data "oci_core_services" "all" {
  filter {
    name   = "name"
    values = ["All .* Services In Oracle Services Network"]
    regex  = true
  }
}

resource "oci_core_vcn" "this" {
  compartment_id = var.compartment_ocid
  display_name   = "${local.name}-vcn"
  cidr_blocks    = [local.vcn_cidr]
  dns_label      = substr(replace(lower(local.name), "/[^a-z0-9]/", ""), 0, 15)
  freeform_tags  = local.common_tags
}

resource "oci_core_internet_gateway" "this" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-igw"
  freeform_tags  = local.common_tags
}

resource "oci_core_nat_gateway" "this" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-nat"
  freeform_tags  = local.common_tags
}

resource "oci_core_service_gateway" "this" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-sgw"
  freeform_tags  = local.common_tags
  services {
    service_id = data.oci_core_services.all.services[0].id
  }
}

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-public-rt"
  freeform_tags  = local.common_tags
  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.this.id
  }
}

resource "oci_core_route_table" "private" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-private-rt"
  freeform_tags  = local.common_tags
  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_nat_gateway.this.id
  }
  route_rules {
    destination       = data.oci_core_services.all.services[0].cidr_block
    destination_type  = "SERVICE_CIDR_BLOCK"
    network_entity_id = oci_core_service_gateway.this.id
  }
}

# --- Security Lists ----------------------------------------------------------

resource "oci_core_security_list" "api" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-api-sl"
  freeform_tags  = local.common_tags

  ingress_security_rules {
    description = "kubectl to Kubernetes API"
    protocol    = "6"
    source      = var.api_allowed_cidr
    tcp_options {
      min = 6443
      max = 6443
    }
  }
  ingress_security_rules {
    description = "workers to Kubernetes API"
    protocol    = "6"
    source      = local.nodes_cidr
    tcp_options {
      min = 6443
      max = 6443
    }
  }
  ingress_security_rules {
    description = "workers to OKE control plane"
    protocol    = "6"
    source      = local.nodes_cidr
    tcp_options {
      min = 12250
      max = 12250
    }
  }
  ingress_security_rules {
    description = "path MTU discovery"
    protocol    = "1"
    source      = local.nodes_cidr
    icmp_options {
      type = 3
      code = 4
    }
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

resource "oci_core_security_list" "nodes" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-nodes-sl"
  freeform_tags  = local.common_tags

  ingress_security_rules {
    description = "node-to-node and pod-to-pod traffic"
    protocol    = "all"
    source      = local.nodes_cidr
  }
  ingress_security_rules {
    description = "control plane to workers (webhooks, logs, exec)"
    protocol    = "6"
    source      = local.api_cidr
  }
  ingress_security_rules {
    description = "load balancers to worker NodePorts and health checks"
    protocol    = "6"
    source      = local.lb_cidr
  }
  ingress_security_rules {
    description = "path MTU discovery"
    protocol    = "1"
    source      = "0.0.0.0/0"
    icmp_options {
      type = 3
      code = 4
    }
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

resource "oci_core_security_list" "lb" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.this.id
  display_name   = "${local.name}-lb-sl"
  freeform_tags  = local.common_tags

  dynamic "ingress_security_rules" {
    for_each = [80, 443]
    content {
      description = "public HTTP/HTTPS traffic to Istio ingress load balancer"
      protocol    = "6"
      source      = "0.0.0.0/0"
      tcp_options {
        min = ingress_security_rules.value
        max = ingress_security_rules.value
      }
    }
  }

  egress_security_rules {
    description = "load balancers to worker NodePorts"
    protocol    = "6"
    destination = local.nodes_cidr
  }
}

# --- Subnets -----------------------------------------------------------------

resource "oci_core_subnet" "api" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.this.id
  display_name      = "${local.name}-api"
  cidr_block        = local.api_cidr
  dns_label         = "api"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.api.id]
  freeform_tags     = local.common_tags
}

resource "oci_core_subnet" "nodes" {
  compartment_id             = var.compartment_ocid
  vcn_id                     = oci_core_vcn.this.id
  display_name               = "${local.name}-nodes"
  cidr_block                 = local.nodes_cidr
  dns_label                  = "nodes"
  prohibit_public_ip_on_vnic = true
  route_table_id             = oci_core_route_table.private.id
  security_list_ids          = [oci_core_security_list.nodes.id]
  freeform_tags              = local.common_tags
}

resource "oci_core_subnet" "lb" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.this.id
  display_name      = "${local.name}-lb"
  cidr_block        = local.lb_cidr
  dns_label         = "lb"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.lb.id]
  freeform_tags     = local.common_tags
}
