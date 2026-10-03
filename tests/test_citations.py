"""LLM-facing evidence formatting in mcp-server/citations.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs-agent-mcp" / "mcp-server"))

import citations  # noqa: E402


def test_release_date_epoch_renders_as_iso_date():
    hits = [
        {
            "distance": 0.5,
            "entity": {
                "citation_url": "https://www.kubeflow.org/docs/releases/1.9",
                "version": "1.9",
                "release_date": 1721606400,
            },
        }
    ]

    text, cites = citations.format_docs_hits(hits)

    assert "**Release date:** 2024-07-22" in text
    assert "1721606400" not in text
    assert cites[0]["release_date"] == 1721606400


def test_release_date_passes_through_non_epoch_values():
    assert citations.format_release_date("2024-07-22") == "2024-07-22"
    assert citations.format_release_date(None) == "None"
