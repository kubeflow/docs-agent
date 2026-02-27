# Kubeflow Docs-Agent — OCI Deployment (Terraform)

This directory contains Terraform modules for deploying the Kubeflow
docs-agent stack on **Oracle Cloud Infrastructure (OCI)**.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     OCI Tenancy                      │
│  ┌───────────────────────────────────────────────┐  │
│  │            OKE Cluster (Kubernetes)            │  │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────────┐  │  │
│  │  │  Milvus │ │  KServe  │ │  docs-agent   │  │  │
│  │  │ (vector │ │ (LLM     │ │ (WebSocket +  │  │  │
│  │  │   DB)   │ │  serving)│ │  HTTPS API)   │  │  │
│  │  └─────────┘ └──────────┘ └───────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Modules

| Module       | Description                                      |
|------------- |--------------------------------------------------|
| `oke`        | OKE cluster with CPU + GPU node pools            |
| `milvus`     | Milvus vector DB via Helm chart                  |
| `kserve`     | KServe + Knative + cert-manager for LLM serving  |
| `docs-agent` | docs-agent Deployment, Service, Ingress          |

## Quick start

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your OCI credentials

terraform init
terraform plan
terraform apply
```

## Prerequisites

- Terraform >= 1.5
- OCI CLI configured (`oci setup config`)
- An OCI compartment with sufficient quotas (OKE, GPU shapes)
- A container registry with the docs-agent image built and pushed
