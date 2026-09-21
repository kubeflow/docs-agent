"""Triage labels and instruction Markdown stay in sync."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync-github-labels.sh"
LABELS_DOC = REPO_ROOT / "docs" / "agents" / "triage-labels.md"
WORKFLOW = REPO_ROOT / "docs" / "agents" / "issue-triage.md"
README = REPO_ROOT / "docs" / "agents" / "README.md"

CREATE_LABEL_RE = re.compile(r'^create_label "([^"]+)"')
ALLOWED_RE = re.compile(r"^\s+- ([a-zA-Z0-9_ /.-]+)\s*$")


def _labels_from_sync_script() -> list[str]:
    names = []
    for line in SYNC_SCRIPT.read_text(encoding="utf-8").splitlines():
        match = CREATE_LABEL_RE.match(line)
        if match:
            names.append(match.group(1))
    return names


def _allowed_from_workflow(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_allowed = False
    names = []
    for line in lines:
        if line.strip() == "allowed:":
            in_allowed = True
            continue
        if in_allowed:
            if line.strip() == "max: 5":
                break
            match = ALLOWED_RE.match(line)
            if match:
                names.append(match.group(1))
            elif line.strip() and not line.strip().startswith("-"):
                break
    return names


class TestTriageLabels:
    def test_sync_script_defines_area_mcp(self):
        assert "area/mcp" in _labels_from_sync_script()

    def test_every_sync_label_is_documented(self):
        doc = LABELS_DOC.read_text(encoding="utf-8")
        missing = [name for name in _labels_from_sync_script() if f"`{name}`" not in doc]
        assert missing == []

    def test_allowlist_matches_sync_script(self):
        assert set(_allowed_from_workflow(WORKFLOW)) == set(_labels_from_sync_script())

    def test_instruction_file_is_aw_source(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "safe-outputs:" in text
        assert "add-labels:" in text
        assert "docs/agents/architecture.md" in text
        assert "area/mcp" in text

    def test_readme_tells_how_to_open_a_compile_pr(self):
        text = README.read_text(encoding="utf-8")
        assert "gh aw compile" in text
        assert "issue-triage.md" in text
        assert "sync-github-labels.sh" in text
        assert "kfp-issue-triage" not in text
