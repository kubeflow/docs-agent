# OCI OKE Cluster Terraform Module (`docs-agent`)

Provisions a dedicated Oracle Cloud Infrastructure (OCI) Container Engine for Kubernetes (OKE) cluster for `kubeflow/docs-agent` with:

- **VCN (`10.0.0.0/16`)**:
  - Public API Endpoint Subnet (`10.0.0.0/28`)
  - Public Load Balancer Subnet (`10.0.20.0/24`) for `istio-ingressgateway`
  - Private Worker Node Subnet (`10.0.10.0/24`) behind a NAT Gateway + Service Gateway
- **CPU Worker Node Pool (`<cluster_name>-cpu-pool`)**:
  - `VM.Standard.E4.Flex` (default: 2 nodes × 4 OCPUs / 64 GB RAM, 100 GB boot volume)
  - Hosts Milvus, KFP, KServe, Knative, Istio, TEI embeddings (`embeddings-service`), TEI reranker (`reranker-service`), Kagent controller/UI, `mcp-server`, `session-issuer`, and `feedback-service`
- **Optional GPU Worker Node Pool (`<cluster_name>-gpu-pool`)**:
  - Controlled by `enable_gpu_pool` (default `true`, or `--no-gpu` in `docs-cli`)
  - `VM.GPU.A10.1` (default: 1 node, 200 GB boot volume for model weights)
  - Tainted with `nvidia.com/gpu=present:NoSchedule` and labeled `workload=gpu-inference` for the `llm-runtime` vLLM chart (Gemma 4 / Qwen)
- **Resource Tagging**:
  - Every resource carries `freeform_tags = { "docs-cli-managed" = "true", "docs-cli-cluster" = var.cluster_name }` so `docs-cli nuke` can verify ownership and safely sweep orphaned cloud resources.

## Usage with `docs-cli` (Recommended)

```bash
# Render .docs-cli/oke.tfvars.json from docs.toml + environment variables
docs-cli template

# Provision OKE + platform stack + Helm charts
docs-cli up --apply

# Or CPU-only (external LLM API):
docs-cli up --no-gpu --apply
```

## Standalone Terraform Usage

```bash
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```
