# `docs-cli` (`docsctl`) — Kubeflow Docs Agent OCI OKE + Netlify CLI

`docs-cli` (also aliased as `docsctl`) is the full-lifecycle development & operations CLI for spinning up, scaling, updating, verifying, and tearing down `kubeflow/docs-agent` clusters on **Oracle Cloud Infrastructure (OCI) Container Engine for Kubernetes (OKE)** and wiring them to **Netlify** (either the official `kubeflow/website` Hugo + Docsy site or the standalone `frontend/` demo site).

Every workflow documented below was validated end-to-end across a complete **Destroy → Cold-Start Provision → 35-Check Verification → Multi-Agent A2A Stream → Live Browser E2E** cycle.

---

## 1. Quick Start & Pre-Flight Validation

```bash
# 1. Configure secure environment variables (never committed to Git)
cp .env.example .env
# Optional: set OCI_COMPARTMENT_ID, HF_TOKEN, GITHUB_TOKEN, NETLIFY_AUTH_TOKEN, NETLIFY_SITE_ID in .env
# (OCI auth automatically uses ~/.oci/config if OCI_TENANCY is unset)

# 2. Check installed CLIs (terraform, kubectl, helm, oci, hugo), local paths, and credentials
./bin/docs-cli doctor

# 3. Preview rendered Terraform (.tfvars.json) and Helm values (.docs-cli/helm/*.values.yaml)
./bin/docs-cli template

# 4. Validate upstream sync, rendered K8s Secret contracts, Helm lint, and multi-agent topology
./bin/docs-cli update --check --diagnose
```

---

## 2. Cold-Start Provisioning (`docs-cli up`)

```bash
# Spin up the full OKE cluster (2x CPU + 1x A10 GPU), Platform stack, Multi-Agent Kagent CRDs,
# KFP v4 ingestion runs, Kubeflow Central Dashboard, and a 6-hour auto-scale-down TTL timer:
./bin/docs-cli up --apply --ttl 6h

# Or provision a CPU-only cluster using an external OpenAI-compatible LLM API:
./bin/docs-cli up --no-gpu --apply
```

### Validated Cold-Start Protections Built Into `docs-cli up`
1. **Automatic OCI GPU Capacity-Report AD Pinning**: Queries `ComputeClient.create_compute_capacity_report` across all Availability Domains in the target region and pins `gpu_availability_domains` in `oke.tfvars.json` exclusively to ADs reporting `AVAILABLE` capacity for `VM.GPU.A10.1` (preventing single-AD `Out of host capacity` launch failures).
2. **Ordered Readiness & Webhook Guards**:
   - Waits for OKE worker nodes to join Kubernetes and report `Ready` (and deploys the `oci-growfs` boot-volume expansion DaemonSet) before starting Platform Terraform.
   - Waits for `kserve-controller-manager` webhook readiness before applying `InferenceService/embeddings-service` and `InferenceService/reranker-service`.
   - Waits for Milvus (`milvus-milvus-standalone`), SeaweedFS S3 (MinIO-free), TEI Embeddings/Reranker, KFP (`platform-k8s-native`), and GPU vLLM (`google/gemma-4-E4B-it`) `Ready` conditions before executing verification.
3. **Automated KFP v4 Ingestion & Milvus Hybrid Bootstrap**:
   - Registers Kubernetes-native `Pipeline` & `PipelineVersion` (`pipelines.kubeflow.org/v2beta1`) CRDs (`github-rag`, `issues-rag`, `code-rag`), creates the `docs-agent-rag-ingestion` Experiment, triggers all 3 pipeline runs (`ingest-github-rag-v4`, `ingest-issues-rag-v4`, `ingest-code-rag-v4`), and seeds the Milvus v4 hybrid collections (`kubeflow_docs`, `issues_rag`, `code_rag`).

---

## 3. Unified Local Kubeflow Dashboard & Local Chat (`docs-cli expose-dashboard`)

```bash
# Expose the Kubeflow Central Dashboard locally on IPv4 + IPv6 (auto-reconnecting):
./bin/docs-cli expose-dashboard --port 8085

# Or run in the background (PID saved to .docs-cli/dashboard-port-forward.pid):
./bin/docs-cli expose-dashboard -b --port 8085
```

* **Validated Local Routes (`http://localhost:8085`)**:
  - **Central Dashboard (`http://localhost:8085/`)**: Cross-namespace visibility across `kubeflow`, `ml-infra`, and `docs-agent`, with top-level sidebar tabs for `Experiments (KFP)`, `Runs (KFP)`, `Pipelines (KFP)`, `KServe Endpoints (Models)`, and `Flo Docs Agent (Chat)`.
  - **KFP Experiments & Runs (`http://localhost:8085/_/pipeline/#/experiments`)**: Automatically injects multi-user `namespace=kubeflow` parameters and displays `docs-agent-rag-ingestion` and its 3 `Succeeded` runs.
  - **KServe Endpoints (`http://localhost:8085/_/kserve-endpoints/`)**: Streams real-time `text/event-stream` SSE updates for all 3 `Ready` models in `ml-infra` (`embeddings-service`, `llm`, `reranker-service`).
  - **Zero-TLS-Prompt Local Flo Chat (`http://localhost:8085/chat/`)**: Proxies `/chat/`, `/api/session`, `/api/feedback`, and `/a2a/*` on `localhost` so you can test all personas (`kubeflow-docs-agent`, `kubeflow-docs-agent-think`, `kubeflow-debug-agent`) without self-signed browser certificate warnings.

---

## 4. Ordered 35-Check Verification Suite (`docs-cli verify`)

```bash
./bin/docs-cli verify
```

Validates all **6 deployment stages (35 checks)**:
1. **Native CLIs**: `oci`, `terraform`, `kubectl`, `helm`, `openssl`, `curl`.
2. **OKE Cluster**: Kubeconfig, `3/3` worker nodes `Ready`, `nvidia.com/gpu` capacity.
3. **Platform & ML Infra**: Istio LoadBalancer IP, KFP `platform-k8s-native` Helm chart + CRDs, MinIO-free SeaweedFS in `ml-infra` & `kubeflow`, TEI `embeddings-service` & `reranker-service` `Ready=True`.
4. **Secrets, GPU LLM & Multi-Agent CRs**: All managed K8s Secrets, GPU `InferenceService/llm` (`Ready=True`), and dynamically discovered Kagent `Agent` CRDs (`kubeflow-docs-agent`, `kubeflow-docs-agent-think`, `kubeflow-debug-agent`).
5. **Edge & A2A Security**: Port 80 → 443 redirect (`301`), RS256 `POST /api/session` + JWKS verification, unauthenticated A2A block (`403`), tampered JWT rejection (`401`), `kagent-ui` admin API & `%2f` path-traversal isolation (`404`/`403`), and strict CORS origin enforcement.
6. **Frontends & Dashboard**: Isolated public HTTPS chatbot (`https://chat.<LB_IP>.nip.io`), local Central Dashboard (`200`), KFP Pipelines/Experiments/Runs (`3 pipelines, 1 experiment, 3 runs`), KServe Models REST + SSE stream (`3 models`), and cross-namespace dropdown (`kubeflow, ml-infra, docs-agent`).

---

## 5. Cost Controls: Hibernate, Wake & Auto-Scale TTL (`scale-down`, `scale-up`, `ttl`)

```bash
# Inspect active in-cluster auto-scale-down timer
./bin/docs-cli ttl --status

# Arm or update an in-cluster TTL job to hibernate node pools to 0 after 6 hours
./bin/docs-cli ttl --set 6h

# Hibernate immediately (scales CPU=0, GPU=0 and releases the demo frontend while preserving PVCs & LB IP)
./bin/docs-cli scale-down --yes

# Hibernate GPU node pool only (keep CPU nodes running)
./bin/docs-cli scale-down --gpu-only --yes

# Wake hibernated node pools and arm a 4-hour TTL timer
./bin/docs-cli scale-up --ttl 4h --yes
```

---

## 6. Full Environment Teardown (`docs-cli destroy`)

```bash
# Dry-run: inspect all OKE, LoadBalancer, BlockVolume, and VCN resources tagged for this cluster
./bin/docs-cli destroy

# Execute complete teardown and verify 0 OCI cloud resources remain:
./bin/docs-cli destroy --yes

# Independent post-destroy verification via OCI SDK:
./bin/docs-cli nuke
```

* **Validated Teardown Order**:
  1. Stops local port-forwards and cancels `kube-system/docs-agent-autoscale-ttl`.
  2. Deletes `InferenceService`, `Milvus`, and `PipelineVersion` CRs, uninstalls Helm releases, deletes `Service/istio-ingressgateway` (detaching the OCI LoadBalancer), and removes workloads + PVCs across `ml-infra`, `docs-agent`, and `kubeflow` (detaching all CSI Block Volumes).
  3. Runs `terraform destroy` on `docs-agent-mcp/terraform` then `deployments/terraform/oke-cluster`.
  4. Polls and sweeps OCI APIs until **0** OKE clusters, LoadBalancers, Block Volumes, and VCNs remain, then cleans ephemeral `.docs-cli/` state (`kubeconfig`, `tls_*`, `.tfstate`).

---

## 7. Multi-Agent Architecture & Handling Larger Upstream Features

### How `kubeflow/docs-agent` Uses Multi-Agent Orchestration
Recent upstream commits (`#261`, `#262`, `#268`, `#274`) establish a Kubernetes-native multi-agent architecture:
- **Role-Keyed Kagent `Agent` CRDs (`kagent.agents.*`)**:
  - `kubeflow-docs-agent` (`docs` role on `flo-llm`): Fast grounded documentation & GitHub issue Q&A (`search_kubeflow_docs`, `search_github_issues`).
  - `kubeflow-docs-agent-think` (`docsThink` role on `flo-llm-think` with `reasoningEffort: low`): Deeper multi-step reasoning mode toggled from the chat UI.
  - `kubeflow-debug-agent` (`debug` role on `flo-llm`): Code, YAML, and SDK troubleshooting (`search_kubeflow_code`, `search_github_issues`, `search_kubeflow_docs`).
- **Model-Neutral Catalog (`llm-runtime`)**: Serves the active catalog model (`gemma` or `qwen`) behind a single stable alias (`flo-llm` at `http://llm-stable.ml-infra.svc.cluster.local/v1`).
- **Operational Agentic Jobs**: Pre-merge/PostSync LLM-judge evaluation (`eval-job.yaml`), nightly thumbs-down feedback triage (`triage-cronjob.yaml`), and scheduled KFP backfill/incremental ingestion.

### Do We Need Agentic Workflows or Does `docs-cli` Handle Upstream Evolution?
`docs-cli` uses a **Hybrid Deterministic Core + Contract Gate** pattern:
1. **90% of Evolution — Handled Automatically by Deterministic Introspection (`./bin/docs-cli update`)**:
   - Dynamically discovers all enabled `kagent.agents.*` from `docs-agent/values.yaml` (so adding a 4th or 5th agent—such as a Semantic Router or Supervisor agent—is automatically deployed and verified by `docs-cli` with zero code edits).
   - Syncs `llm-runtime` catalog profiles, GHCR image tags (`--track main|prod`), Terraform variable defaults, and KFP v4 pipeline CRDs automatically.
2. **10% Structural Shifts — Caught by Contract Gate (`./bin/docs-cli update --check --diagnose`)**:
   - Cloud provisioning and teardown (`up`, `scale-down`, `destroy`) remain **100% deterministic and LLM-free** for safety and repeatability.
   - Whenever an upstream commit introduces a brand-new Argo CD app in `deploy/argocd/apps/*.yaml`, a new required Kubernetes `Secret` reference (`secretKeyRef`), or a new CRD group, `docs-cli update --check --diagnose` fails fast in CI and writes `.docs-cli/drift-diagnosis.json` (`enabled_kagent_agents`, `unmapped_argocd_apps`, `contract_issues`, `drifted_fields`) so a developer or CI coding agent can generate a targeted PR updating `docs-cli`.

---

## 8. Netlify & `kubeflow/website` Integration (`docs-cli netlify`)

```bash
# Build standalone demo bundle (with same-origin & Netlify /a2a/* proxy rules in netlify.toml)
./bin/docs-cli netlify --mode demo

# Or inject Docsy head/body hooks + chatbot assets into a kubeflow/website checkout and deploy:
./bin/docs-cli netlify --mode website --website-path ../kubeflow-website/website --build-hugo --deploy
```

---

## 9. CLI Commands Reference

| Command | Description |
| :--- | :--- |
| `docs-cli doctor` | Check installed binaries (`terraform`, `kubectl`, `helm`, `oci`, `hugo`), paths, and credentials |
| `docs-cli template [--no-gpu]` | Render `.docs-cli/oke.tfvars.json`, `.docs-cli/platform.tfvars.json`, and `.docs-cli/helm/*.values.yaml` |
| `docs-cli up [--no-gpu] [--ttl 6h] [--apply]` | Full cold-start OKE + Platform + Multi-Agent + KFP ingestion bring-up & 35-check verification |
| `docs-cli expose-dashboard [-b] [--port 8085]` | Auto-reconnecting IPv4+IPv6 local port-forward for Kubeflow Central Dashboard, Pipelines, KServe Models & `/chat/` |
| `docs-cli dashboard [-b] [--port 8085]` | Alias for `expose-dashboard` |
| `docs-cli update [--pull] [--check] [--diagnose] [--apply]` | Sync upstream charts/Terraform/GHCR, validate contracts, run `pytest`, and upgrade live cluster |
| `docs-cli verify [--helm-tests]` | Run the 6-stage, 35-check infrastructure, security, multi-agent, and dashboard verification suite |
| `docs-cli ingest [--apply] [--run]` | Compile KFP v4 Python pipelines into Kubernetes-native `Pipeline`/`PipelineVersion` CRDs and trigger runs |
| `docs-cli scale-down [--gpu-only] [--after 2h] [--yes]` | Hibernate OKE node pools to `0` (and release demo frontend) while preserving PVCs & LB IP |
| `docs-cli scale-up [--ttl 6h] [--yes]` | Wake hibernated OKE node pools and restore workloads + optional TTL timer |
| `docs-cli ttl [--status \| --set 6h \| --cancel]` | Inspect, arm, or cancel the in-cluster auto-scale-down Kubernetes Job |
| `docs-cli netlify [--mode website\|demo] [--deploy]` | Inject Hugo/Docsy hooks or build standalone bundle and deploy to Netlify |
| `docs-cli status` | Inspect OKE nodes, KServe `InferenceServices`, Kagent `Agents`, and Istio Ingress IP |
| `docs-cli destroy [--yes]` | Full environment teardown (port-forward, K8s LBs/PVCs, Terraform stacks, OCI zero-orphan sweep) |
| `docs-cli nuke [--yes]` | Direct OCI API inspection/sweep of any resources tagged `docs-cli-cluster = <cluster_name>` |
