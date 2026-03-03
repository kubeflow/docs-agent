# GSoC 2026 Project Specification: Agentic RAG on Kubeflow

## Executive Summary
This project evolves `kubeflow/docs-agent` from a baseline retrieval script into an enterprise-grade, Multi-Agent Retrieval-Augmented Generation (RAG) Reference Architecture native to Kubeflow. The goal is to provide the community with a hardened, dogfooded architecture that bridges Kubernetes abstraction with advanced Large Language Model (LLM) reasoning, capable of intelligently querying both Kubeflow documentation and release code. 

To ensure maximum community adoption, we are scoping three modular architectural blueprints while standardizing on a primary "live" deployment.

---

## 1. Architectural Blueprints (The "Art of the Possible")
We are designing the system across four decoupled layers so that users can adopt the components that fit their target environments:
1. **Frontend:** User-facing conversational interface.
2. **Middle Agent/Router:** The orchestrator (or an advanced collaborative system of a dedicated 'code agent' and 'docs agent') that reasons about the query and invokes tools.
3. **Ingestion Pipeline (KFP):** Kubeflow Pipelines responsible for continuous ETL, chunking, and embedding.
4. **Backend Vector Database:** Scalable ANN storage (e.g., pgvector or Milvus).

### Interaction Modes & The "Thin Context" MCP Flow
We are designing for two primary interaction modes: **Frontend Chat Mode** and **Developer IDE Mode**, each tailored to a specific audience and purpose.

*   **Frontend Chat Mode (Website):** The website's chat interface integrates with our MCP to provide best-effort responses with references to documents. It is highly specialized for Official Documentation and community details. It is **not** meant to be development-focused and does **not** rely on the code agent.
*   **Developer IDE Mode:** The developer's IDE integrates with our MCP to primarily use the **code agent**. This flow is heavily repository and code-focused. It understands branches for specific versions, can parse GitHub issues for questions, and directly reads the codebase. This is explicitly designed for deep development support.

We can offload complex "supervisor" orchestration to the developer's IDE agent (e.g., Claude, Cursor) or utilize a simple supervisor for our frontend. We will also provide documented guidance on system prompts so developers can configure their IDEs to collaborate optimally with our MCP.

**The Tool Handshake Flow (IDE Example):**
1. **The Trigger (The Developer Asks a Question):** A developer is working in their IDE (Cursor, Copilot, or Claude) or chatting on the web frontend. They ask a syntax-heavy API question (e.g., "How do I configure the mutating webhook for notebooks?").
2. **The Handshake (The MCP Intercept):** Instead of the user's expensive LLM trying to guess the answer or reading dozens of files, it recognizes the need for official documentation. It sends a targeted query through our public OSS `docs_agent` MCP.
3. **The Precision Retrieval (The Self-Hosted Brain):** Our Kubeflow-hosted Docs Agent receives the query. Powered by a fast, self-hosted model, it searches the vectorized documentation and codebase, isolating the exact file and snippet.
4. **The 'Thin Context' Package (The Hand-off):** The Docs Agent returns a highly structured data package back through the MCP containing exactly two things:
   - *The Golden Snippet:* The exact 150-token block of code or YAML.
   - *The Validation Link:* The direct URL to that specific file in the GitHub repo or official docs site.
5. **The Synthesis and Verification (The Final Output):** The user's IDE Agent receives this tiny, perfect context. It instantly generates the correct code in the user's editor and appends a note: "Here is your webhook configuration. Source: [link]". Crucially, to protect our cloud credits and context windows, we will strictly guide the end user's local agent to pull the repository and parse the provided codebase link for more details and broader context—relying on the user's local compute to extend the retrieved snippet, rather than having our API return a massive, expensive context payload.

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
*   **Architecture C: Advanced Agent Framework (LangGraph / Google ADK via KServe)**
    *   *Concept:* A stateful, graph-based agent reasoning loop served via KServe that handles complex multi-tool routing (e.g., deciding whether to query the Docs Vector DB vs. the Code AST Vector DB). Two framework options are provided:
        *   **LangGraph:** A cyclic, graph-based orchestration framework that models agent reasoning as stateful directed graphs with support for conditional branching, loops, and human-in-the-loop checkpoints.
        *   **Google Agent Development Kit (ADK):** Google's open-source agent framework with deterministic workflow primitives (sequential, parallel, loop). Offers predictable control flow without unbounded cycles and native multi-agent orchestration.
    *   *Pros:* Capable of deep "Plan-Act-Observe" reasoning loops, robust human-in-the-loop state pausing, and cyclic error correction (LangGraph). Deterministic, reproducible workflow execution with explicit pipeline composition and built-in multi-agent coordination (ADK).
    *   *Cons:* Highest complexity; LangGraph requires robust tracing to debug infinite loops. ADK's deterministic flow trades off some flexibility for predictability.

---

## 2. Repository Structure
The repository is fundamentally decoupled to separate infrastructure from agent logic and pipeline execution. 

This is an iterative effort, and contributors should expect to build and evolve this structure progressively over the duration of the project.

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

### Phase 1: Ingestion & Foundation
**Focus: Data Pipeline and Vector Persistence**
*   **KFP Ingestion:** Create robust Kubeflow Pipelines to crawl, heavily chunk, and embed the Kubeflow documentation.
*   **Code Ingestion:** Implement a parallel KFP pipeline to parse standard Kubeflow release code repositories, capturing Abstract Syntax Trees (AST) and context.
*   **Backend:** Stand up the Vector Database with declarative schemas. 
*   **Deliverable:** A functional database loaded with fresh embeddings that update on a schedule via KFP.

### Phase 2: Core Agent & Routing
**Focus: The "Brain" and Tool Calling**
*   **Router Implementation:** Build the semantic router / intent detection layer (LangGraph or Google ADK) to detect if a user is asking a conceptual question (route to docs) or a debugging/technical question (route to code).
*   **Tool Integration:** Implement MCP (Model Context Protocol) bridging the agent capability to internal search APIs securely. 
*   **Multi-Arch Support:** Write the baseline KServe manifests, Kagent CRD specifications, and ADK agent definitions as blueprints.
*   **Deliverable:** An exposed, queryable API endpoint capable of executing stateful RAG with correct contextual generation.

### Phase 3: Deployment, Security & UX
**Focus: Hardening, Feedback, and Release**
*   **Infrastructure as Code:** Finalize Terraform and Helm charts. Validate idempotent installations.
*   **Guardrails & Tracing:** Integrate validation models to sanitize outputs.
*   **Frontend & Feedback Loop:** Tie the frontend directly into the website UI, ensuring all inputs and outputs are explicitly logged, and implement the feedback loop to construct a golden dataset for continuous evaluation.
*   **Deliverable:** The final dogfooded deployment on the community cluster.

---

## 4. Hardened System Considerations (The Checklist)

To ensure this serves as an enterprise reference architecture, the stack is fortified against the following constraints:

### Baseline Requirements
1.  **Security Best Practices:** Agent-to-tool communication is strictly standardized using the **Model Context Protocol (MCP)**. This limits arbitrary remote code execution by confining tool calls to predefined schemas. Traffic between the agent, tools, and the Vector DB is secured via **Istio sidecars** enforcing mTLS policies.
2.  **OWASP LLM Top 10 Mitigations:**
    *   *Prompt Injection & System Prompting:* We must implement a properly scoped and secure system prompt that establishes strict behavior boundaries. Use segregated system prompts and structural boundary markers for user input. Inputs run through a lightweight classifier before reaching the LLM.
    *   *Excessive Agency:* The agent's service account operates with the principle of least privilege. MCP tools have zero write access to cluster state—they are strictly `Read-Only` (or isolated to issue-templating tools).
    *   *Data/Model Poisoning:* KFP ingestion pipelines enforce cryptographic signature checks on the source repositories to ensure no tampered code enters the vector DB.
3.  **Guardrails:** Implement a fast, local guardrail model (e.g., Llama-Guard or explicit heuristics) deployed natively in KServe that intercepts LLM outputs to strip toxic/violent language and ensure Kubeflow brand safety.
4.  **Idempotent Deployments:** All Helm charts and Terraform modules utilize remote state and declarative drift detection. Deployments are designed to run safely in automated git-sync modes (e.g., ArgoCD) without mutating persistent data destructively. 
5.  **RetryLogic:** Robust retry logic is a must for all tools. The agent implements exponential backoff with jitter for Vector DB retrievals and LLM API timeouts. If tools strictly fail, the agent is configured to transparently degrade, informing the user that "Live code context is currently unreachable."
6.  **Feedback Loops & Golden Datasets:** The frontend UI must require a feedback loop (e.g., thumbs up/down mechanism) and ensure all inputs and outputs are logged. If a user sees a bad response, this explicit feedback is extremely important because it allows us to systematically test and optimize the system for robustness. A server-side webhook collects the prompt, the traced vector context retrieved, and the agent's faulty response. This aggregated data constructs a "golden dataset" used to manually fix outputs or automatically fine-tune the LLM pipeline (e.g., adjusting retrieval logic, optimizing model temperatures, or shifting weights). These efforts integrate nicely with Katib, as called out in the [Optimizing RAG Pipelines with Katib](https://blog.kubeflow.org/katib/rag/) blog post, which can support these hyperparameter tuning efforts over the RAG architecture.
7.  **CI/CD & Code Scanning:** Custom agent code is scanned on PR via SonarQube/Trivy for vulnerabilities. Telemetry validation is mocked in CI to ensure OpenTelemetry spans correctly propagate.
8.  **Authentication, Scaling, & Parallelism:** The frontend is secured via standard OAuth2 proxy logic. We could explore relying heavily on **KServe’s scale-to-zero capabilities** to optimize costs during inactive windows while evaluating burst scaling (parallel inference pods) during high community traffic.

### Additional Enterprise-Grade Considerations
9.  **Websockets & Stateful Connections:** Because of the complex agent reasoning loops, standard HTTP request/response lifecycles may timeout. We could explore supporting WebSockets or Server-Sent Events (SSE) for long-lived streaming connections to the frontend.
10. **Connection Database (State Management):** To support these long-lived connections and iterative human-in-the-loop reasoning (where an agent asks clarifying questions), we should explore implementing a dedicated Connection DB (e.g., assessing Redis or a dedicated PostgreSQL table) to persist conversation thread IDs and connection state across the stateless backend pods.
11. **Observability & Telemetry Tracing (OpenTelemetry):** End-to-end distributed tracing is highly recommended. The framework could emit OpenTelemetry (OTel) spans tracking the exact lifecycle of a prompt: Frontend Request -> Agent Routing -> Vector DB Query Latency -> LLM Generation Time. This would enable mentors and operators to graph bottleneck latency and debug complex hallucination chains.
12. **Data Isolation & Multi-Tenancy (RBAC for Indices):** We could explore physically or logically isolating Vector Collections. The "Documentation" collection and the "Release Code" collection might have separated RBAC and indexing to prevent "noisy neighbor" semantic overlap. If we expand to multi-tenant capabilities (e.g., separate Kubeflow Sub-projects), namespace-level vector isolation could prevent cross-contamination.
13. **Environment Portability & Configurable Endpoints:** The architecture must be highly configurable via environment variables (e.g., LLM endpoints, DB connection strings, tool hosts). While our primary live deployment will be hosted on OCI and should leverage managed cloud-native backends where possible for resilience, the system must remain fully decoupled. This leaves room for a "bring your own" (BYO) approach if community members want to host the entire solution entirely on their own local Kubernetes cluster using open-source backends.
14. **User Logins, Rate Limiting, & Adaptive Quotas:** The system's exploratory target goal is supporting up to 1,000 concurrent users. To manage this safely and prevent budget exhaustion, we will require login for both the **Development MCP** and the **Website Frontend**. This login gateway enforces strict rate limiting and daily token quotas via middleware (e.g., Litellm), gathers valuable telemetry on Kubeflow platform usage patterns, and allows us to invite users to the Kubeflow mailing list—turning casual users into engaged community members. As a future state, we aim to provide additional tiered API credits to active maintainers, GSoC students, and core contributors. *Note: As part of the architectural design phase, the community will need to evaluate and agree upon the specific tech stack for managing user storage, authentication, and credit/quota logic.*

---

## 5. UI/UX Expectations for Frontend Chat

To ensure a high-quality user experience and build trust, the frontend chat interface must adhere to strict design and response expectations, similar to advanced commercial AI agents. 

### 1. User Input & Process Transparency (The "Thinking" Phase)
Before delivering the final answer, the UI explicitly shows the user the steps the agent is taking.
*   **User Query Pill:** The user's prompt is displayed at the top in a distinct, dark gray rounded container.
*   **Action Updates:** Plain text lines that narrate what the agent is currently doing (e.g., "I'll search for information...").
*   **Search Execution Boxes:** Outlined, full-width containers that show specific tool executions. They include:
    *   A magnifying glass icon on the left.
    *   The specific source being queried (e.g., "Searched in the Wiz docs").
    *   A green checkmark icon on the right to indicate successful completion.
*   **Source Citations:** A list of the specific documents retrieved, visually indicated by blue document icons and blue hyperlink text.

### 2. Main Response Structure (The "Answer" Phase)
The actual answer is highly structured for readability and scannability.
*   **Direct Answer:** The response begins with a clear, one-sentence direct answer to the user's question.
*   **Section Headings:** Uses bold, slightly larger text (e.g., H2 or H3 tags) to divide major concepts (e.g., "What Wiz Offers for AI Guardrails", "How to Use It").
*   **Bold Lead-ins:** Paragraphs often start with a bolded keyword or phrase followed by a colon (e.g., **Visibility:**, **Misconfiguration Detection:**) to allow users to quickly scan for relevant features.
*   **Bulleted Lists:** Uses unordered lists with simple dot bullets for enumerating items (like supported cloud providers or types of attacks).
*   **Inline Links with Icons:** Important technical terms link out to documentation. Uses a small, upward-right arrow icon (↗) next to links like "Security Graph" to denote an external or internal redirect.
*   **Call to Action:** Ends with a specific "Learn more" link pointing to the main documentation.

### 3. Utility Footer
*   **Feedback & Actions:** The bottom of the response includes minimalist line-art icons for user feedback (thumbs up / thumbs down) on the left, and a "copy to clipboard" icon on the right.

### 4. General Styling
*   **Theme:** The interface uses a dark mode color palette with high-contrast white/light-gray text for readability.
*   **Accent Colors:** Blue is used consistently for interactive elements (links, document icons), and green is used sparingly for success indicators.

---

## 6. Direct IDE Integration (The "BYO Agent" Experience)

While the frontend chat interface serves users browsing `kubeflow.org`, advanced users and active contributors will interact directly with the Kubeflow MCP via their own Integrated Development Environments (IDEs). 

### How it Works
Developers will be able to register our hosted `docs-agent` MCP directly into tools like Cursor, Windsurf, or generic Claude Desktop clients. 
1. **Developer Registration:** A developer signs into the Kubeflow developer portal to retrieve an API key/token.
2. **MCP Configuration:** They add standard MCP connection details (URL, API Token) to their local IDE’s MCP configuration file.
3. **Direct Architecture Access:** Once connected, the user's *local* code agent can autonomously route queries to our Kubeflow-hosted MCP to fetch exact codebase snippets, GitHub issues, and vectorized documentation directly into their local development environment—bypassing the web UI entirely.

This "Bring Your Own Agent" setup ensures the system meets developers precisely where they write code, utilizing their own local LLM compute while relying on our infrastructure purely for high-fidelity retrieval.
