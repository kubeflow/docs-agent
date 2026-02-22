# GSoC 2026 Project Specification: Agentic RAG on Kubeflow

## Executive Summary
This project evolves `kubeflow/docs-agent` from a baseline retrieval script into an enterprise-grade, Multi-Agent Retrieval-Augmented Generation (RAG) Reference Architecture native to Kubeflow. The goal is to provide the community with a hardened, dogfooded architecture that bridges Kubernetes abstraction with advanced Large Language Model (LLM) reasoning, capable of intelligently querying both Kubeflow documentation and release code. 

To ensure maximum community adoption, we are scoping three modular architectural blueprints while standardizing on a primary "live" deployment.

---

## 1. Architectural Blueprints (The "Art of the Possible")
We are designing the system across four decoupled layers so that users can adopt the components that fit their target environments:
1. **Frontend:** User-facing conversational interface.
2. **Middle Agent/Router:** The orchestrator that reasons about the query and invokes tools.
3. **Ingestion Pipeline (KFP):** Kubeflow Pipelines responsible for continuous ETL, chunking, and embedding.
4. **Backend Vector Database:** Scalable ANN storage (e.g., pgvector or Milvus).

### Blueprint Options
While we will provide documentation and scaffolding for all three, **Architecture B (Kagent)** will be the primary deployment we dogfood and support live.

*   **Architecture A: Self-hosted Custom Agent (Served via KServe)**
    *   *Concept:* The agent is a monolithic Python container (e.g., FastAPI + LangChain/LlamaIndex) exposed as a standardized KServe `InferenceService`.
    *   *Pros:* highly portable, relies heavily on Kubeflow's trusted KServe/Knative backbone for scale-to-zero capabilities. 
    *   *Cons:* Stateless by nature; complex conversational memory management requires externalized Redis/state stores.
*   **Architecture B: "Managed" Agent (via Kagent) - *Primary Live Target***
    *   *Concept:* Leverages Kubernetes-native agent management (`Kagent`). Agents are declared as Kubernetes Custom Resource Definitions (CRDs).
    *   *Pros:* Aligns closely with GitOps and Kubernetes-native lifecycle management. Native observability via K8s control plane.
    *   *Cons:* Tightly coupled to the Kagent ecosystem; potentially restricts the use of highly customized cyclical reasoning loops.
*   **Architecture C: Advanced Agent Framework (LangGraph via KServe)**
    *   *Concept:* A stateful, graph-based deterministic agent reasoning loop (e.g., LangGraph) that handles complex multi-tool routing (e.g., deciding whether to query the Docs Vector DB vs. the Code AST Vector DB).
    *   *Pros:* Capable of deep "Plan-Act-Observe" reasoning loops, robust human-in-the-loop state pausing, and cyclic error correction. 
    *   *Cons:* Highest complexity; requires robust tracing to debug infinite loops.

---

## 2. Repository Structure
The repository is fundamentally decoupled to separate infrastructure from agent logic and pipeline execution. 

```text
kubeflow/docs-agent/
├── frontend/                     # Conversational UI (React/Streamlit/Gradio)
│   ├── app/                      
│   └── Dockerfile                # Thin container for the frontend
├── agent/                        # Middleware & Routing Logic
│   ├── core/                     # LangGraph cyclic graphs & state managers
│   ├── tools/                    # MCP (Model Context Protocol) implementations
│   ├── kserve/                   # InferenceService & ServingRuntime manifests
│   └── tests/                    # Agent evaluation frameworks (RAGAS/rag-eval)
├── pipelines/                    # KFP Ingestion Pipelines
│   ├── docs_ingestion/           # Scrapes, chunks, and embeds MKDocs/Markdown
│   ├── code_ingestion/           # AST parsing for release code indexing
│   └── shared/                   # Shared embedding sub-components
├── backend/                      # Vector DB Configurations
│   ├── schemas/                  # Vector schemas and index parity docs
│   └── manifests/                # StatefulSet/Operator configs for DB setup
├── deployments/                  # Multi-environment Infrastructure as Code
│   ├── terraform/                # OCI/Cloud provider provisioning (GCP/AWS)
│   └── helm/                     
│       └── docs-agent/           # Helm chart for idempotent K8s deployment
└── .github/                      
    └── workflows/                # CI/CD, Container builds, and Code Scanning
```

---

## 3. Phased Contributor Work Breakdown

### Phase 1: Ingestion & Foundation (Weeks 1-4)
**Focus: Data Pipeline and Vector Persistence**
*   **KFP Ingestion:** Create robust Kubeflow Pipelines to crawl, heavily chunk, and embed the Kubeflow documentation.
*   **Code Ingestion:** Implement a parallel KFP pipeline to parse standard Kubeflow release code repositories, capturing Abstract Syntax Trees (AST) and context.
*   **Backend:** Stand up the Vector Database with declarative schemas. 
*   **Deliverable:** A functional database loaded with fresh embeddings that update on a schedule via KFP.

### Phase 2: Core Agent & Routing (Weeks 5-8)
**Focus: The "Brain" and Tool Calling**
*   **Router Implementation:** Build the LangGraph semantic router to detect if a user is asking a conceptual question (route to docs) or a debugging/technical question (route to code).
*   **Tool Integration:** Implement MCP (Model Context Protocol) bridging the agent capability to internal search APIs securely. 
*   **Multi-Arch Support:** Write the baseline KServe manifests and Kagent CRD specifications as blueprints.
*   **Deliverable:** An exposed, queryable API endpoint capable of executing stateful RAG with correct contextual generation.

### Phase 3: Deployment, Security & UX (Weeks 9-12)
**Focus: Hardening, Feedback, and Release**
*   **Infrastructure as Code:** Finalize Terraform and Helm charts. Validate idempotent installations.
*   **Guardrails & Tracing:** Integrate validation models to sanitize outputs.
*   **Frontend & Feedback Loop:** Wire up the UI, ensuring all inputs and outputs are explicitly logged, and implement the feedback loop to construct a golden dataset for continuous evaluation.
*   **Deliverable:** The final dogfooded deployment on the community cluster.

---

## 4. Hardened System Considerations (The Checklist)

To ensure this serves as an enterprise reference architecture, the stack is fortified against the following constraints:

### Baseline Requirements
1.  **Security Best Practices:** Agent-to-tool communication is strictly standardized using the **Model Context Protocol (MCP)**. This limits arbitrary remote code execution by confining tool calls to predefined schemas. Traffic between the agent, tools, and the Vector DB is secured via **Istio sidecars** enforcing mTLS policies.
2.  **OWASP LLM Top 10 Mitigations:**
    *   *Prompt Injection:* Use segregated system prompts and structural boundary markers for user input. Inputs run through a lightweight classifier before reaching the LLM.
    *   *Excessive Agency:* The agent's service account operates with the principle of least privilege. MCP tools have zero write access to cluster state—they are strictly `Read-Only` (or isolated to issue-templating tools).
    *   *Data/Model Poisoning:* KFP ingestion pipelines enforce cryptographic signature checks on the source repositories to ensure no tampered code enters the vector DB.
3.  **Guardrails:** Implement a fast, local guardrail model (e.g., Llama-Guard or explicit heuristics) deployed natively in KServe that intercepts LLM outputs to strip toxic/violent language and ensure Kubeflow brand safety.
4.  **Idempotent Deployments:** All Helm charts and Terraform modules utilize remote state and declarative drift detection. Deployments are designed to run safely in automated git-sync modes (e.g., ArgoCD) without mutating persistent data destructively. 
5.  **RetryLogic:** The agent implements exponential backoff with jitter for Vector DB retrievals and LLM API timeouts. If tools strictly fail, the agent is configured to transparently degrade, informing the user that "Live code context is currently unreachable."
6.  **Feedback Loops & Golden Datasets:** The frontend UI must require a feedback loop (e.g., thumbs up/down mechanism) and ensure all inputs and outputs are logged. When a response evaluates poorly, a server-side webhook collects the prompt, the traced vector context retrieved, and the agent's faulty response. This aggregated data constructs a "golden dataset" used to manually fix outputs or automatically fine-tune the LLM pipeline (e.g., adjusting retrieval logic, optimizing model temperatures, or shifting weights). These efforts integrate nicely with Katib, as called out in the [Optimizing RAG Pipelines with Katib](https://blog.kubeflow.org/katib/rag/) blog post, which can support these hyperparameter tuning efforts over the RAG architecture.
7.  **CI/CD & Code Scanning:** Custom agent code is scanned on PR via SonarQube/Trivy for vulnerabilities. Telemetry validation is mocked in CI to ensure OpenTelemetry spans correctly propagate.
8.  **Authentication, Scaling, & Parallelism:** The frontend is secured via standard OAuth2 proxy logic. The backend agent relies heavily on **KServe’s scale-to-zero capabilities** to optimize costs during inactive windows while allowing burst scaling (parallel inference pods) during high community traffic.

### Additional Enterprise-Grade Considerations
9.  **Observability & Telemetry Tracing (OpenTelemetry):** End-to-end distributed tracing is mandatory. The framework will emit OpenTelemetry (OTel) spans tracking the exact lifecycle of a prompt: Frontend Request -> Agent Routing -> Vector DB Query Latency -> LLM Generation Time. This enables mentors and operators to graph bottleneck latency and debug complex hallucination chains.
10. **Data Isolation & Multi-Tenancy (RBAC for Indices):** We must physically or logically isolate Vector Collections. The "Documentation" collection and the "Release Code" collection will have separated RBAC and indexing to prevent "noisy neighbor" semantic overlap. If we expand to multi-tenant capabilities (e.g., separate Kubeflow Sub-projects), namespace-level vector isolation prevents cross-contamination.
11. **Cost Controls & Adaptive Token Quotas:** LLM calls can easily exhaust budgets if subjected to abuse (DoS attacks). We will configure Envoy/Istio rate limiting on the KServe ingress to throttle rapid user queries, alongside an internal token-counting middleware (e.g., Litellm) to set a hard API budget limit per day.
