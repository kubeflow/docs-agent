# Kubeflow Contributor Guide

Welcome to the Kubeflow project! We'd love to accept your patches and 
contributions to this project. Please read the 
[contributor's guide in our docs](https://www.kubeflow.org/docs/about/contributing/).

The contributor's guide:

* Shows you where to find the Developer Certificate of Origin (DCO) that you need 
  to agree to
* Helps you get started with your first contribution to Kubeflow
* Describes the pull request and review workflow in detail, including the
  OWNERS files and automated workflow tool

## Opening issues

Use a title of the form `<type>(<area>): <summary>` so the issue triage workflow
can attach source context and labels.

- Types: `bug`, `feat`, `chore`, `docs`, `test`, `ci`, `security`
- Areas: `mcp`, `pipelines`, `frontend`, `kagent`, `gateway`, `terraform`,
  `embeddings`, `ci`, `tests`, `docs`, `infra`

Example: `bug(mcp): search_kubeflow_docs returns Search failed on empty collection`.

Create the triage labels with `./scripts/sync-github-labels.sh`, then see
[`docs/agents/README.md`](docs/agents/README.md) for the instruction Markdown
Shristi compiles into a GitHub Agentic Workflow test PR.
