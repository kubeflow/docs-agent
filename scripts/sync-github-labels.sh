#!/usr/bin/env bash
# Create or update GitHub labels used by docs-agent issue triage.
# Usage: ./scripts/sync-github-labels.sh [owner/repo]
set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"

create_label() {
  local name="$1" color="$2" description="$3"
  if gh label list --repo "$REPO" --json name -q '.[].name' | grep -Fxq "$name"; then
    gh label edit "$name" --repo "$REPO" --color "$color" --description "$description" >/dev/null
  else
    gh label create "$name" --repo "$REPO" --color "$color" --description "$description" >/dev/null
  fi
}

create_label "kind/bug" "D93F0B" "Something is broken"
create_label "kind/feature" "2515FC" "New capability"
create_label "kind/chore" "C2E0C6" "Cleanup, refactor, or ops"
create_label "kind/docs" "0075CA" "Documentation only"
create_label "kind/security" "B60205" "Auth, injection, XSS, secrets, mesh"

create_label "area/mcp" "1D76DB" "FastMCP server, tools, /mcp handshake"
create_label "area/pipelines" "1D76DB" "KFP docs/issues/code ingestion"
create_label "area/frontend" "E99695" "Website chat widget"
create_label "area/kagent" "1D76DB" "Agent, ModelConfig, RemoteMCPServer CRDs"
create_label "area/gateway" "C5DEF5" "Istio edge, TLS, CORS, session JWT"
create_label "area/terraform" "C5DEF5" "Terraform platform stack"
create_label "area/embeddings" "FBCA04" "TEI embeddings client and service"
create_label "area/ci" "FBCA04" "GitHub Actions and CD"
create_label "area/tests" "FBCA04" "pytest, smoke, golden dataset"
create_label "area/docs" "0075CA" "README, runbooks, agent docs"
create_label "area/infra" "C5DEF5" "Milvus, KServe, Istio, namespaces"

create_label "needs-triage" "FBCA04" "Waiting for analyzer or maintainer triage"
create_label "needs-info" "D876E3" "Reporter needs to provide more detail"
create_label "good first issue" "7057FF" "Scoped task for new contributors"
create_label "help wanted" "008672" "Maintainers want community help"
create_label "maintainer-only" "B60205" "Agentic RAG core — discuss before coding"
create_label "gsoc-2026" "5319E7" "Tied to gsoc2026_agentic_rag.md"
create_label "duplicate" "CFD3D7" "Same defect as another issue"

create_label "priority/p0" "B60205" "Security incident, data loss, or public outage"
create_label "priority/p1" "D93F0B" "Major regression or blocker"
create_label "priority/p2" "FBCA04" "Normal actionable work"

echo "Labels synced on $REPO"
