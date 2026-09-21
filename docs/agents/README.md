# Agent docs — issue triage (first merge)

Instruction Markdown and labels for GitHub Agentic Workflows on **this repo**.
Merge this first. Then copy the instruction file, compile, and open a test PR.

| File | Role |
| --- | --- |
| [`issue-triage.md`](./issue-triage.md) | AW source — copy to `.github/workflows/issue-triage.md` |
| [`triage-labels.md`](./triage-labels.md) | `kind/*` and `area/*` labels (`area/mcp`, …) |
| [`architecture.md`](./architecture.md) | Trusted layer/file map the agent must read |
| [`area-map.json`](./area-map.json) | Same map as JSON |

## Create labels

GitHub will not apply a label that does not exist. After merge:

```bash
./scripts/sync-github-labels.sh kubeflow/docs-agent
```

Titles: `<type>(<area>): <summary>` — example `bug(mcp): search_kubeflow_docs returns Search failed`.

## Run initial AW testing with a PR

```bash
gh extension install github/gh-aw   # once
cp docs/agents/issue-triage.md .github/workflows/issue-triage.md
gh aw compile .github/workflows/issue-triage.md
git add .github/workflows/issue-triage.md .github/workflows/issue-triage.lock.yml
git commit -s -m "chore: compile docs-agent issue triage AW"
# open a PR with those two files
```

After that PR is on a repo with issues enabled, open a test issue:

- `bug(mcp): search_kubeflow_docs returns Search failed on empty collection`

Expect `kind/bug` + `area/mcp` and one triage comment.
