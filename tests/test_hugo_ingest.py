"""Tests for utils/hugo_ingest.py.

Only the production surface is covered: canonical_rag_ingest imports
parse_frontmatter and process_html_table. clean_hugo_markdown has no caller.

Expectations come from the module's docstrings and inline comments, the way
parse_canonical_document consumes the results, and docs/RAG_V4_ARCHITECTURE.md,
which names Hugo frontmatter and release tables as inputs v4 must handle.
"""

import sys
from pathlib import Path

import pytest

UTILS_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines" / "utils"
sys.path.insert(0, str(UTILS_DIR))

for _dependency in ("frontmatter", "html_table_rescuer", "markdownify", "bs4", "toml"):
    pytest.importorskip(_dependency, reason="hugo_ingest tests need the docs ingest dependencies")

import hugo_ingest  # noqa: E402
from hugo_ingest import parse_frontmatter, process_html_table  # noqa: E402


def gfm_tables(text):
    """Group consecutive `|` lines into tables, as canonical_rag_ingest does."""
    tables, current = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            current.append(line.strip())
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def cells(row):
    """Split a GFM row into cells, honouring escaped pipes."""
    inner = row.strip()[1:-1]
    return [cell.strip().replace("\0", "|") for cell in inner.replace("\\|", "\0").split("|")]


def rows(table):
    return [cells(line) for line in table]


# --- parse_frontmatter -------------------------------------------------------

YAML_PAGE = """---
title: Katib
description: Hyperparameter tuning
weight: 20
---
# Overview

Body text.
"""

TOML_PAGE = """+++
title = "Katib"
description = "Hyperparameter tuning"
weight = 20
+++
# Overview

Body text.
"""


@pytest.mark.parametrize("page", [YAML_PAGE, TOML_PAGE], ids=["yaml", "toml"])
def test_frontmatter_yields_the_fields_parse_canonical_document_reads(page):
    meta, body = parse_frontmatter(page)

    assert meta["title"] == "Katib"
    assert meta["description"] == "Hyperparameter tuning"
    # parse_canonical_document does int(meta.get("weight") or 0).
    assert int(meta.get("weight") or 0) == 20
    assert body.startswith("# Overview")
    assert "title" not in body and "+++" not in body and "---" not in body


def test_yaml_and_toml_frontmatter_parse_to_the_same_metadata():
    assert parse_frontmatter(YAML_PAGE) == parse_frontmatter(TOML_PAGE)


def test_page_without_frontmatter_keeps_its_body():
    page = "# Just markdown\n\nSome text.\n"
    meta, body = parse_frontmatter(page)

    assert meta == {}
    assert body.strip() == page.strip()


def test_byte_order_mark_does_not_hide_frontmatter():
    meta, body = parse_frontmatter("\ufeff" + YAML_PAGE)

    assert meta["title"] == "Katib"
    assert body.startswith("# Overview")


def test_missing_keys_are_absent_so_callers_fall_back_to_defaults():
    meta, _ = parse_frontmatter("---\ntitle: Only a title\n---\nbody\n")

    assert meta == {"title": "Only a title"}
    assert int(meta.get("weight") or 0) == 0


@pytest.mark.parametrize("content", ["", None], ids=["empty", "none"])
def test_empty_input_returns_empty_metadata_unchanged(content):
    assert parse_frontmatter(content) == ({}, content)


@pytest.mark.parametrize(
    "page",
    [
        "---\ntitle: [unclosed\n---\nbody\n",
        "+++\ntitle = \n+++\nbody\n",
    ],
    ids=["yaml", "toml"],
)
def test_malformed_frontmatter_falls_back_instead_of_failing_the_run(page):
    # One bad page must not abort ingest of the whole docs tree.
    assert parse_frontmatter(page) == ({}, page)


# --- process_html_table ------------------------------------------------------


def test_html_without_a_table_is_returned_unchanged():
    # Round-tripping through BeautifulSoup would rewrite this as
    # "Katib &amp; Trainer <br/> v1.9".
    html = "Katib & Trainer <br> v1.9, with <b>bold</b> and no tables."
    assert process_html_table(html) == html


def test_table_becomes_a_gfm_table_with_header_separator_and_rows():
    html = "<table><tr><th>Component</th><th>Version</th></tr><tr><td>Katib</td><td>v0.17</td></tr></table>"
    out = process_html_table(html)

    assert "<table" not in out
    [table] = gfm_tables(out)
    header, separator, *body = rows(table)
    assert header == ["Component", "Version"]
    assert all(set(cell) <= {"-", ":"} and cell for cell in separator)
    assert body == [["Katib", "v0.17"]]


def test_rowspan_value_is_repeated_into_every_spanned_row():
    # Release tables group several components under one release cell.
    html = (
        "<table><tr><th>Release</th><th>Component</th><th>Version</th></tr>"
        '<tr><td rowspan="2">1.9</td><td>Katib</td><td>v0.17</td></tr>'
        "<tr><td>Trainer</td><td>v1.8</td></tr></table>"
    )
    [table] = gfm_tables(process_html_table(html))

    assert rows(table)[2:] == [["1.9", "Katib", "v0.17"], ["1.9", "Trainer", "v1.8"]]


def test_colspan_keeps_every_row_the_same_width():
    html = '<table><tr><th>A</th><th>B</th></tr><tr><td colspan="2">merged</td></tr></table>'
    [table] = gfm_tables(process_html_table(html))

    widths = {len(row) for row in rows(table)}
    assert widths == {2}
    assert rows(table)[2][0] == "merged"


def test_pipe_inside_a_cell_does_not_add_a_column():
    html = "<table><tr><th>Flag</th><th>Values</th></tr><tr><td>--mode</td><td>a|b</td></tr></table>"
    [table] = gfm_tables(process_html_table(html))

    assert rows(table)[2] == ["--mode", "a|b"]


def test_text_around_a_table_is_kept_off_the_table_rows():
    html = "Intro para.\n<table><tr><th>H</th></tr><tr><td>v</td></tr></table>\nOutro para."
    out = process_html_table(html)

    assert "Intro para." in out and "Outro para." in out
    [table] = gfm_tables(out)
    assert not any("para" in line for line in table)


def test_text_between_two_tables_separates_them():
    html = (
        "<table><tr><th>X</th></tr><tr><td>1</td></tr></table>"
        "mid"
        "<table><tr><th>Y</th></tr><tr><td>2</td></tr></table>"
    )
    tables = gfm_tables(process_html_table(html))

    assert [rows(t)[0] for t in tables] == [["X"], ["Y"]]
    assert [rows(t)[2] for t in tables] == [["1"], ["2"]]


def test_falls_back_to_markdownify_when_the_table_parser_returns_nothing(monkeypatch):
    class EmptyParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse(self):
            return []

    monkeypatch.setattr(hugo_ingest, "TableParser", EmptyParser)
    html = "<table><tr><th>Component</th></tr><tr><td>Katib</td></tr></table>"
    out = process_html_table(html)

    assert "<table" not in out
    [table] = gfm_tables(out)
    assert rows(table)[0] == ["Component"]
    assert ["Katib"] in rows(table)
