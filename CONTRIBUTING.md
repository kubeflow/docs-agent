# Contributing to Kubeflow docs-agent

Welcome! We're excited you want to contribute to `kubeflow/docs-agent` — the official Kubeflow Documentation AI Assistant powered by Agentic RAG.

Before anything else, please read the [Kubeflow contributor's guide](https://www.kubeflow.org/docs/about/contributing/) for the Developer Certificate of Origin (DCO) agreement and general Kubeflow contribution policies.

---

## Repository Structure

This repository is organized into four decoupled layers. Please familiarize yourself with them before contributing:
```
kubeflow/docs-agent/
+-- agent/          # LangGraph agent graphs, MCP tools, KServe manifests, evaluation
+-- frontend/       # Conversational UI (React/Streamlit/Gradio)
+-- pipelines/      # KFP ingestion pipelines (docs + code)
+-- backend/        # Vector database schemas and manifests
+-- deployments/    # Terraform and Helm charts
+-- server/         # WebSocket and HTTPS API servers
```

For a full architectural overview, see [`gsoc2026_agentic_rag.md`](./gsoc2026_agentic_rag.md).

---

## Getting Started Locally

### Prerequisites
- Python 3.10+
- Git
- pip or a virtual environment manager (venv / conda)

### Setup

1. **Fork and clone the repository**
```bash
   git clone https://github.com/YOUR_USERNAME/docs-agent.git
   cd docs-agent
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r pipelines/requirements.txt
```

4. **Set up environment variables**

   Copy the example env file and fill in your values:
```bash
   cp .env.example .env
```

   Key variables to configure:
   | Variable | Description |
   |---|---|
   | `KSERVE_URL` | KServe LLM inference endpoint |
   | `MILVUS_HOST` | Milvus vector database host |
   | `MILVUS_PORT` | Milvus port (default: 19530) |
   | `EMBEDDING_MODEL` | Sentence transformer model name |

---

## Contribution Workflow

### 1. Find or open an issue
All contributions must be tied to a tracked GitHub Issue. Check the [Issues tab](https://github.com/kubeflow/docs-agent/issues) for open work, or open a new issue describing what you want to build. Reference the relevant section of `gsoc2026_agentic_rag.md` in your issue.

### 2. Create a feature branch
```bash
git checkout -b your-branch-name
```

Use descriptive branch names:
- `feat/langgraph-intent-classifier`
- `fix/milvus-connection-retry`
- `docs/update-agent-readme`

### 3. Make your changes
- Keep PRs focused — one feature or fix per PR
- Add or update tests where applicable
- Update the relevant `README.md` inside the affected directory

### 4. Commit with a clear message
Follow this format:
```
type: short description (max 72 chars)

Longer explanation if needed. Reference the issue number.

Closes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `scaffold`, `chore`

### 5. Push and open a Pull Request
```bash
git push origin your-branch-name
```

In your PR description:
- Explain **what** you changed and **why**
- Reference the related GitHub Issue
- Reference the relevant section of `gsoc2026_agentic_rag.md` if applicable

### 6. Review process
PRs require approval from a maintainer before merging. The `google-oss-prow` bot manages the review workflow. Once approved, a maintainer will assign the `lgtm` label and `franciscojavierarceo` will do the final approval.

---

## Code Style

- Python: follow [PEP 8](https://peps.python.org/pep-0008/)
- Use descriptive variable names — avoid single-letter names outside of loop counters
- Add docstrings to all functions and classes
- Keep functions small and single-purpose

---

## Need Help?

- Join the [CNCF Slack](https://slack.cncf.io/) and find us in `#kubeflow` and `#kubeflow-gsoc-participants`
- Open a GitHub Issue with the `question` label
- Review the full architecture spec: [`gsoc2026_agentic_rag.md`](./gsoc2026_agentic_rag.md)

We review PRs regularly and aim to respond within a few days. Thank you for contributing! ??
