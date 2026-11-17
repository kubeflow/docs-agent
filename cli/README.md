# `docs-cli` (`docsctl`) — Multi-Cloud (OCI OKE & GCP GKE) + Netlify CLI

`docs-cli` (also aliased as `docsctl`) is the full-lifecycle development & operations CLI for spinning up, scaling, updating, verifying, and tearing down `kubeflow/docs-agent` clusters on **Oracle Cloud Infrastructure (OCI OKE)** and **Google Cloud Platform (GCP GKE)**, and wiring them to **Netlify** (either the official `kubeflow/website` Hugo + Docsy site or the standalone `frontend/` demo site).

---

## 0. How Repo-Hosted Terraform & Helm Work With `docs-cli`

All infrastructure-as-code lives **outside the Python CLI** in versioned repository directories so Terraform and Helm remain single sources of truth:

| Layer | Repository Path | Responsibility |
| :--- | :--- | :--- |
| **OCI Cluster Terraform** | `deployments/terraform/oke-cluster/` | OCI VCN, Subnets, Security Lists, OKE Enhanced Cluster, CPU (`VM.Standard.E5.Flex`) & GPU (`VM.GPU.A10.1`) Node Pools |
| **GCP Cluster Terraform** | `deployments/terraform/gke-cluster/` | GCP VPC, VPC-native Subnet (Pod/Service secondary CIDRs), Cloud Router/NAT, GKE Standard Cluster, CPU (`n2-highmem-8`) & GPU (`g2-standard-12` / `nvidia-l4` with automatic GKE driver installer) Node Pools |
| **Shared Platform Terraform** | `docs-agent-mcp/terraform/` | Cloud-agnostic Kubernetes stack (`cert-manager`, `Istio`, `Knative`, `KServe`, `Milvus + SeaweedFS`, `Kubeflow Pipelines`, `TEI Embeddings & Reranker`, `Kagent`) |
| **Application Helm Charts** | `docs-agent-mcp/charts/` | `llm-runtime` (`gemma`/`qwen` vLLM catalog), `docs-agent` (MCP + multi-agent `kagent.agents.*`), and `gateway-guardrails` (RS256 JWT + Envoy Lua + CORS) |
| **Kubeflow Upstream Charts** | `deployments/charts/kubeflow-manifests/` | Automatically cloned/checked out at pinned upstream revision by `docs-cli` when `use_kubeflow_helm_charts = true` |

### How `docs-cli` Pulls & Stays Synced With Repo Terraform/Helm
1. **`./bin/docs-cli update --pull`**: Pulls latest `main` commits, introspects `variables.tf` across `oke-cluster`, `gke-cluster`, and `docs-agent-mcp/terraform` plus `values.yaml` across all Helm charts, updates `docs.toml`, recompiles KFP v4 pipelines if modified, and re-renders `.docs-cli/{oke,gke}.tfvars.json`, `.docs-cli/platform.tfvars.json`, and `.docs-cli/helm/*.values.yaml`.
2. **Expected-Architecture CI Gate (`./bin/docs-cli update --check --diagnose`)**: Runs on every PR, `main` push, and release (`.github/workflows/tests.yml`, `oke-cicd.yaml`, `e2e-cli-release-gate.yaml`). Fails CI immediately if a commit adds an unmapped Argo CD app, unmapped Kubernetes `Secret` reference, or drifted chart/Terraform default.

---

## 1. Cloud Topology Comparison (OCI OKE vs. GCP GKE)

Select the target cloud via `--cloud oci` (default) or `--cloud gcp`, `DOCS_CLOUD=gcp` in `.env`, or `[cloud] provider = "gcp"` in `docs.toml`.

| Capability | OCI OKE (`--cloud oci`) | GCP GKE (`--cloud gcp`) |
| :--- | :--- | :--- |
| **Auth Methods** | `OCI_*` env vars or `~/.oci/config` | `gcloud auth application-default login` (ADC), Workload Identity, or `GOOGLE_APPLICATION_CREDENTIALS` JSON |
| **CPU Worker Pool** | `2x VM.Standard.E5.Flex` (4 OCPU / 8 vCPU, 64 GB RAM) | `2x n2-highmem-8` (8 vCPU, 64 GB RAM, `COS_CONTAINERD`) |
| **GPU Worker Pool** | `1x VM.GPU.A10.1` (NVIDIA A10 24 GB VRAM) | `1x g2-standard-12` (`1x nvidia-l4` 24 GB VRAM, auto-installed GKE GPU driver) |
| **GPU Capacity Discovery** | OCI `ComputeCapacityReport` API pins `AVAILABLE` ADs | `gcloud compute accelerator-types list` discovers `nvidia-l4` zones in region |
| **Model Cache PVC (50Gi)** | `oci-bv` CSI Block Volume | `standard-rwo` Compute Engine Persistent Disk CSI |
| **Ingress LoadBalancer** | OCI Flexible LoadBalancer (`10–100 Mbps`) | Google Cloud External Passthrough Network LoadBalancer (`cloud.google.com/l4-rbs`) |
| **Hibernate / TTL Auto-Scale** | OCI ContainerEngine signed REST API (`size=0`) | GKE Container Engine `:setSize` REST API / `gcloud container clusters resize` |
| **Destroy / Nuke Ownership** | Freeform tags `docs-cli-managed=true` + `docs-cli-cluster` | GCP resource labels `docs-cli-managed=true` + `docs-cli-cluster` |

---

## 2. Quick Start: OCI OKE (`--cloud oci`)

```bash
# 1. Configure secure environment variables (never committed to Git)
cp .env.example .env
# Optional: set OCI_COMPARTMENT_ID, HF_TOKEN, GITHUB_TOKEN, NETLIFY_AUTH_TOKEN, NETLIFY_SITE_ID in .env

# 2. Pre-flight check & render templates
./bin/docs-cli doctor --cloud oci
./bin/docs-cli template --cloud oci

# 3. Cold-start provision (OKE + Platform + Multi-Agent + KFP v4 ingestion + 6h auto-scale TTL)
./bin/docs-cli up --cloud oci --apply --ttl 6h

# Or CPU-only mode with an external OpenAI-compatible API:
./bin/docs-cli up --cloud oci --no-gpu --apply
```

---

## 3. Quick Start: Google Cloud GKE (`--cloud gcp`)

### Prerequisites on GCP
1. Install Google Cloud CLI (`gcloud`) and the GKE auth plugin:
   ```bash
   gcloud components install gke-gcloud-auth-plugin
   ```
2. Authenticate using **either** Application Default Credentials (interactive dev) **or** a Service Account JSON key (headless CI):
   ```bash
   # Option A: Interactive developer login
   gcloud auth login
   gcloud auth application-default login
   export GCP_PROJECT_ID="your-gcp-project-id"
   export GCP_REGION="us-central1"
   export GCP_ZONE="us-central1-a"

   # Option B: Service Account JSON key
   export GCP_PROJECT_ID="your-gcp-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```
   *(Note: Ensure your GCP project has `NVIDIA_L4_GPUS >= 1` quota in the target region, or pass `--no-gpu` to run CPU-only with an external LLM API.)*

### Provision, Operate & Tear Down on GCP GKE
```bash
# 1. Verify tools (`gcloud`, `gke-gcloud-auth-plugin`, `terraform`, `kubectl`, `helm`) & render GKE templates
./bin/docs-cli doctor --cloud gcp
./bin/docs-cli template --cloud gcp

# 2. Spin up full GKE cluster (VPC + Cloud NAT + 2x n2-highmem-8 + 1x g2-standard-12 L4 GPU),
#    Platform Terraform, standard-rwo HF cache PVC, Multi-Agent Kagent CRDs, KFP v4 runs, and 6h TTL:
./bin/docs-cli up --cloud gcp --apply --ttl 6h

# Or CPU-only GKE cluster (--no-gpu):
./bin/docs-cli up --cloud gcp --no-gpu --apply

# 3. Expose the unified Kubeflow Central Dashboard + KFP Runs + KServe Models + /chat/ locally:
./bin/docs-cli expose-dashboard -b --port 8085

# 4. Run the 6-stage, 35+ check verification suite against GKE:
./bin/docs-cli verify --cloud gcp

# 5. Hibernate & wake GKE node pools (preserving Persistent Disks, Milvus vectors & LB IP):
./bin/docs-cli scale-down --cloud gcp --yes
./bin/docs-cli scale-up --cloud gcp --ttl 4h --yes
./bin/docs-cli ttl --cloud gcp --status

# 6. Full teardown down to 0 GCP cloud resources:
./bin/docs-cli destroy --cloud gcp --yes
```

---

## 4. Unified Local Kubeflow Dashboard & Local Chat (`docs-cli expose-dashboard`)

Works identically against both OKE and GKE clusters via `.docs-cli/kubeconfig`:

```bash
# Expose locally on IPv4 + IPv6 (auto-reconnecting):
./bin/docs-cli expose-dashboard --port 8085

# Or run in the background (PID saved to .docs-cli/dashboard-port-forward.pid):
./bin/docs-cli expose-dashboard -b --port 8085
```

* **Local Routes (`http://localhost:8085`)**:
  - **Central Dashboard (`http://localhost:8085/`)**: Cross-namespace visibility across `kubeflow`, `ml-infra`, and `docs-agent`, with top-level sidebar tabs for `Experiments (KFP)`, `Runs (KFP)`, `Pipelines (KFP)`, `KServe Endpoints (Models)`, and `Flo Docs Agent (Chat)`.
  - **KFP Experiments & Runs (`http://localhost:8085/_/pipeline/#/experiments`)**: Automatically injects multi-user `namespace=kubeflow` parameters and displays `docs-agent-rag-ingestion` and its 3 `Succeeded` runs.
  - **KServe Endpoints (`http://localhost:8085/_/kserve-endpoints/`)**: Streams real-time `text/event-stream` SSE updates for all 3 `Ready` models in `ml-infra` (`embeddings-service`, `llm`, `reranker-service`).
  - **Zero-TLS-Prompt Local Flo Chat (`http://localhost:8085/chat/`)**: Proxies `/chat/`, `/api/session`, `/api/feedback`, and `/a2a/*` on `localhost` so you can test all personas (`kubeflow-docs-agent`, `kubeflow-docs-agent-think`, `kubeflow-debug-agent`) without self-signed browser certificate warnings.

---

## 5. Ordered Verification Suite (`docs-cli verify`)

```bash
./bin/docs-cli verify               # OCI OKE (default)
./bin/docs-cli verify --cloud gcp   # GCP GKE
```

Validates all **6 deployment stages**:
1. **Native CLIs**: `oci` (or `gcloud` + `gke-gcloud-auth-plugin` on GCP), `terraform`, `kubectl`, `helm`, `openssl`, `curl`.
2. **Cluster & Nodes**: Kubeconfig, worker nodes `Ready`, `nvidia.com/gpu` capacity.
3. **Platform & ML Infra**: Istio LoadBalancer IP, KFP `platform-k8s-native` Helm chart + CRDs, MinIO-free SeaweedFS in `ml-infra` & `kubeflow`, TEI `embeddings-service` & `reranker-service` `Ready=True`.
4. **Secrets, GPU LLM & Multi-Agent CRs**: All managed K8s Secrets, GPU `InferenceService/llm` (`Ready=True`), and dynamically discovered Kagent `Agent` CRDs (`kubeflow-docs-agent`, `kubeflow-docs-agent-think`, `kubeflow-debug-agent`).
5. **Edge & A2A Security**: Port 80 → 443 redirect (`301`), RS256 `POST /api/session` + JWKS verification, unauthenticated A2A block (`403`), tampered JWT rejection (`401`), `kagent-ui` admin API & `%2f` path-traversal isolation (`404`/`403`), and strict CORS origin enforcement.
6. **Frontends & Dashboard**: Isolated public HTTPS chatbot (`https://chat.<LB_IP>.nip.io`), local Central Dashboard (`200`), KFP Pipelines/Experiments/Runs (`3 pipelines, 1 experiment, 3 runs`), KServe Models REST + SSE stream (`3 models`), and cross-namespace dropdown (`kubeflow, ml-infra, docs-agent`).

---

## 6. Cost Controls: Hibernate, Wake & Auto-Scale TTL (`scale-down`, `scale-up`, `ttl`)

```bash
# Inspect active in-cluster auto-scale-down timer
./bin/docs-cli ttl --status

# Arm or update an in-cluster TTL job (works on both OCI OKE and GCP GKE)
./bin/docs-cli ttl --set 6h
./bin/docs-cli ttl --cloud gcp --set 6h

# Hibernate immediately (scales CPU=0, GPU=0 and releases demo frontend while preserving PVCs & LB IP)
./bin/docs-cli scale-down --yes
./bin/docs-cli scale-down --cloud gcp --yes

# Hibernate GPU node pool only (keep CPU nodes running)
./bin/docs-cli scale-down --gpu-only --yes

# Wake hibernated node pools and arm a 4-hour TTL timer
./bin/docs-cli scale-up --ttl 4h --yes
./bin/docs-cli scale-up --cloud gcp --ttl 4h --yes
```

---

## 7. Full Environment Teardown (`docs-cli destroy` & `nuke`)

```bash
# Dry-run: inspect all tagged/labeled cloud resources for this cluster
./bin/docs-cli destroy
./bin/docs-cli destroy --cloud gcp

# Execute complete teardown and verify 0 cloud resources remain:
./bin/docs-cli destroy --yes
./bin/docs-cli destroy --cloud gcp --yes

# Independent post-destroy sweep via OCI SDK or GCP gcloud labels:
./bin/docs-cli nuke
./bin/docs-cli nuke --cloud gcp
```

* **Teardown Order (Both Clouds)**:
  1. Stops local port-forwards and cancels `kube-system/docs-agent-autoscale-ttl`.
  2. Deletes `InferenceService`, `Milvus`, and `PipelineVersion` CRs, uninstalls Helm releases, deletes `Service/istio-ingressgateway` (detaching the cloud LoadBalancer), and removes workloads + PVCs across `ml-infra`, `docs-agent`, and `kubeflow` (detaching all CSI volumes/disks).
  3. Runs `terraform destroy` on `docs-agent-mcp/terraform` then `deployments/terraform/{oke,gke}-cluster`.
  4. Sweeps cloud APIs (verified by `docs-cli-managed=true` and `docs-cli-cluster=<name>`) until **0** cloud resources remain, then cleans ephemeral `.docs-cli/` state (`kubeconfig`, `tls_*`, `.tfstate`).

---

## 8. Multi-Agent Architecture & Expected-Architecture CI Gate

### Role-Keyed Multi-Agent Topology
- **`kubeflow-docs-agent`** (`docs` role on `flo-llm`): Grounded documentation & GitHub issue Q&A (`search_kubeflow_docs`, `search_github_issues`).
- **`kubeflow-docs-agent-think`** (`docsThink` role on `flo-llm-think` with `reasoningEffort: low`): Deeper multi-step reasoning mode toggled from the chat UI.
- **`kubeflow-debug-agent`** (`debug` role on `flo-llm`): Code, YAML, and SDK troubleshooting (`search_kubeflow_code`, `search_github_issues`, `search_kubeflow_docs`).

### Keeping `docs-cli` Locked to Upstream Evolution
- **Additive changes auto-adapt**: `discover_enabled_kagent_agents()` dynamically reads `kagent.agents.*` from `docs-agent/values.yaml` and `catalog.*` from `llm-runtime/values.yaml`.
- **Structural changes fail CI fast**: `.github/workflows/tests.yml`, `oke-cicd.yaml`, and `e2e-cli-release-gate.yaml` run `pytest` + `docs-cli update --check --diagnose` (plus `terraform validate` across `oke-cluster`, `gke-cluster`, and `docs-agent-mcp/terraform` on release/dispatch) and publish `.docs-cli/drift-diagnosis.json` to the GitHub Actions Summary.

---

## 9. CLI Commands Reference

| Command | Description |
| :--- | :--- |
| `docs-cli doctor [--cloud oci\|gcp]` | Check installed binaries (`terraform`, `kubectl`, `helm`, `oci`/`gcloud`, `hugo`), paths, and credentials |
| `docs-cli template [--cloud oci\|gcp] [--no-gpu]` | Render `.docs-cli/{oke,gke}.tfvars.json`, `.docs-cli/platform.tfvars.json`, and `.docs-cli/helm/*.values.yaml` |
| `docs-cli up [--cloud oci\|gcp] [--no-gpu] [--ttl 6h] [--apply]` | Full cold-start OKE/GKE + Platform + Multi-Agent + KFP ingestion bring-up & verification |
| `docs-cli kubeconfig [--cloud oci\|gcp]` | Fetch OKE or GKE credentials into `.docs-cli/kubeconfig` |
| `docs-cli expose-dashboard [-b] [--port 8085]` | Auto-reconnecting IPv4+IPv6 local port-forward for Kubeflow Central Dashboard, Pipelines, KServe Models & `/chat/` |
| `docs-cli dashboard [-b] [--port 8085]` | Alias for `expose-dashboard` |
| `docs-cli update [--pull] [--check] [--diagnose] [--apply]` | Sync upstream charts/Terraform/GHCR, validate contracts, run `pytest`, and upgrade live cluster |
| `docs-cli verify [--cloud oci\|gcp] [--helm-tests]` | Run the 6-stage infrastructure, security, multi-agent, and dashboard verification suite |
| `docs-cli ingest [--apply] [--run]` | Compile KFP v4 Python pipelines into Kubernetes-native `Pipeline`/`PipelineVersion` CRDs and trigger runs |
| `docs-cli scale-down [--cloud oci\|gcp] [--gpu-only] [--after 2h] [--yes]` | Hibernate OKE/GKE node pools to `0` (and release demo frontend) while preserving PVCs & LB IP |
| `docs-cli scale-up [--cloud oci\|gcp] [--ttl 6h] [--yes]` | Wake hibernated OKE/GKE node pools and restore workloads + optional TTL timer |
| `docs-cli ttl [--cloud oci\|gcp] [--status \| --set 6h \| --cancel]` | Inspect, arm, or cancel the in-cluster auto-scale-down Kubernetes Job |
| `docs-cli netlify [--mode website\|demo] [--deploy]` | Inject Hugo/Docsy hooks or build standalone bundle and deploy to Netlify |
| `docs-cli status` | Inspect worker nodes, KServe `InferenceServices`, Kagent `Agents`, and Istio Ingress IP |
| `docs-cli destroy [--cloud oci\|gcp] [--yes]` | Full environment teardown (port-forward, K8s LBs/PVCs, Terraform stacks, zero-orphan cloud sweep) |
| `docs-cli nuke [--cloud oci\|gcp] [--yes]` | Direct OCI/GCP inspection & sweep of resources tagged/labeled `docs-cli-cluster = <cluster_name>` |
