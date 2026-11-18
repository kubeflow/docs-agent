# GCP GKE Cluster Terraform (`deployments/terraform/gke-cluster`)

Provisions a dedicated VPC-native Google Kubernetes Engine (GKE) Standard cluster for `kubeflow/docs-agent`:

- **Dedicated VPC & Subnet** (`network.tf`) with secondary IP ranges for GKE VPC-native Pods (`10.48.0.0/14`) and Services (`10.52.0.0/20`), Cloud Router + Cloud NAT, and internal/webhook firewall rules.
- **GKE Standard Cluster** (`gke.tf`) on the `REGULAR` release channel with `deletion_protection = false` and Workload Identity enabled.
- **CPU Node Pool** (`n2-highmem-4` default: `2x` nodes = 8 vCPU, 64 GB RAM total, `pd-standard` boot disks preserving new-project `CPUS_ALL_REGIONS=12` & `250 GB SSD_TOTAL_GB` quotas) labeled `docs-agent-pool=cpu`.
- **Optional GPU Node Pool** (`g2-standard-12` default: `1x nvidia-l4` 24 GB VRAM) labeled `docs-agent-pool=gpu` and `nvidia.com/gpu=true`, with GKE automatic NVIDIA driver installation (`gpu_driver_version = "DEFAULT"`).
- **Ownership Labels** (`docs-cli-managed = "true"`, `docs-cli-cluster = <cluster_name>`) on all resources for deterministic teardown via `docs-cli destroy --cloud gcp --yes`.

## Recommended Usage via `docs-cli`

```bash
./bin/docs-cli up --cloud gcp --apply --ttl 6h
./bin/docs-cli verify --cloud gcp
./bin/docs-cli destroy --cloud gcp --yes
```
