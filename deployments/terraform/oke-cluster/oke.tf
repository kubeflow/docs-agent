# =============================================================================
# deployments/terraform/oke-cluster/oke.tf
# Provisions the OKE Enhanced cluster, CPU node pool, and optional GPU node pool.
# =============================================================================

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_ocid
}

data "oci_containerengine_cluster_option" "all" {
  cluster_option_id = "all"
}

data "oci_containerengine_node_pool_option" "all" {
  node_pool_option_id = "all"
  compartment_id      = var.compartment_ocid
}

locals {
  k8s_version = var.kubernetes_version != "" ? var.kubernetes_version : element(
    data.oci_containerengine_cluster_option.all.kubernetes_versions,
    length(data.oci_containerengine_cluster_option.all.kubernetes_versions) - 1
  )
  k8s_ver_clean = trimprefix(local.k8s_version, "v")

  # Match x86_64 Oracle Linux 8 OKE images for the target Kubernetes version.
  cpu_image_regex = "^Oracle-Linux-8\\.[0-9]+-20[0-9]{2}\\..*-OKE-${local.k8s_ver_clean}-"
  cpu_node_images = [
    for s in data.oci_containerengine_node_pool_option.all.sources :
    s.image_id if length(regexall(local.cpu_image_regex, s.source_name)) > 0
    && length(regexall("GPU|aarch64", s.source_name)) == 0
  ]

  # Match x86_64 Oracle Linux 8 GPU OKE images (preloaded with NVIDIA drivers).
  gpu_image_regex = "^Oracle-Linux-8\\.[0-9]+-Gen2-GPU-20[0-9]{2}\\..*-OKE-${local.k8s_ver_clean}-"
  gpu_node_images = [
    for s in data.oci_containerengine_node_pool_option.all.sources :
    s.image_id if length(regexall(local.gpu_image_regex, s.source_name)) > 0
    && length(regexall("aarch64", s.source_name)) == 0
  ]

  # Cloud-init script that:
  # 1. Loads kernel modules required by Istio sidecars and gateways
  # 2. Expands the root filesystem to fill the configured boot volume size (100G/250G)
  # 3. Executes the standard OKE node bootstrap script
  worker_cloud_init = base64encode(<<-EOT
    #cloud-config
    runcmd:
      - "modprobe br_netfilter || true"
      - "modprobe nf_nat || true"
      - "modprobe xt_REDIRECT || true"
      - "modprobe xt_owner || true"
      - "modprobe iptable_nat || true"
      - "modprobe iptable_mangle || true"
      - "modprobe iptable_filter || true"
      - "/usr/libexec/oci-growfs -y || true"
      - 'curl --fail -H "Authorization: Bearer Oracle" -L0 http://169.254.169.254/opc/v2/instance/metadata/oke_init_script | base64 --decode >/var/run/oke-init.sh'
      - "bash /var/run/oke-init.sh"
  EOT
  )

  gpu_is_flex = endswith(var.gpu_node_shape, ".Flex")
}

resource "oci_containerengine_cluster" "this" {
  compartment_id     = var.compartment_ocid
  name               = local.name
  kubernetes_version = local.k8s_version
  vcn_id             = oci_core_vcn.this.id
  type               = "ENHANCED_CLUSTER"
  freeform_tags      = local.common_tags

  cluster_pod_network_options {
    cni_type = "FLANNEL_OVERLAY"
  }

  endpoint_config {
    subnet_id            = oci_core_subnet.api.id
    is_public_ip_enabled = true
  }

  options {
    service_lb_subnet_ids = [oci_core_subnet.lb.id]
    kubernetes_network_config {
      pods_cidr     = "10.244.0.0/16"
      services_cidr = "10.96.0.0/16"
    }
    persistent_volume_config {
      freeform_tags = local.common_tags
    }
    service_lb_config {
      freeform_tags = local.common_tags
    }
  }
}

# --- CPU Node Pool -----------------------------------------------------------

resource "oci_containerengine_node_pool" "cpu" {
  compartment_id     = var.compartment_ocid
  cluster_id         = oci_containerengine_cluster.this.id
  name               = "${local.name}-cpu-pool"
  kubernetes_version = local.k8s_version
  node_shape         = var.cpu_node_shape
  ssh_public_key     = var.ssh_public_key != "" ? var.ssh_public_key : null
  freeform_tags      = local.common_tags

  node_shape_config {
    ocpus         = var.cpu_node_ocpus
    memory_in_gbs = var.cpu_node_memory_gbs
  }

  node_source_details {
    source_type             = "IMAGE"
    image_id                = local.cpu_node_images[0]
    boot_volume_size_in_gbs = var.cpu_boot_volume_gbs
  }

  node_config_details {
    size          = var.cpu_node_count
    freeform_tags = local.common_tags

    dynamic "placement_configs" {
      for_each = data.oci_identity_availability_domains.ads.availability_domains
      content {
        availability_domain = placement_configs.value.name
        subnet_id           = oci_core_subnet.nodes.id
      }
    }
  }

  initial_node_labels {
    key   = "docs-agent-pool"
    value = "cpu"
  }

  lifecycle {
    precondition {
      condition     = length(local.cpu_node_images) > 0
      error_message = "No x86_64 Oracle Linux 8 OKE CPU image found for Kubernetes ${local.k8s_version}. Set kubernetes_version to a supported release."
    }
  }
}

# --- GPU Node Pool (Optional: enabled by default, skipped with --no-gpu) -----

resource "oci_containerengine_node_pool" "gpu" {
  count              = var.enable_gpu_pool ? 1 : 0
  compartment_id     = var.compartment_ocid
  cluster_id         = oci_containerengine_cluster.this.id
  name               = "${local.name}-gpu-pool"
  kubernetes_version = local.k8s_version
  node_shape         = var.gpu_node_shape
  ssh_public_key     = var.ssh_public_key != "" ? var.ssh_public_key : null
  freeform_tags      = local.common_tags

  dynamic "node_shape_config" {
    for_each = local.gpu_is_flex ? [1] : []
    content {
      ocpus         = var.gpu_node_ocpus
      memory_in_gbs = var.gpu_node_memory_gbs
    }
  }

  node_source_details {
    source_type             = "IMAGE"
    image_id                = local.gpu_node_images[0]
    boot_volume_size_in_gbs = var.gpu_boot_volume_gbs
  }

  node_config_details {
    size          = var.gpu_node_count
    freeform_tags = local.common_tags

    dynamic "placement_configs" {
      for_each = length(var.gpu_availability_domains) > 0 ? var.gpu_availability_domains : [for ad in data.oci_identity_availability_domains.ads.availability_domains : ad.name]
      content {
        availability_domain = placement_configs.value
        subnet_id           = oci_core_subnet.nodes.id
      }
    }
  }

  initial_node_labels {
    key   = "nvidia.com/gpu"
    value = "true"
  }

  initial_node_labels {
    key   = "docs-agent-pool"
    value = "gpu"
  }

  lifecycle {
    precondition {
      condition     = length(local.gpu_node_images) > 0
      error_message = "No x86_64 Oracle Linux 8 OKE GPU image found for Kubernetes ${local.k8s_version}. Set kubernetes_version to a supported release or use --no-gpu."
    }
  }
}
