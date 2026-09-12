"""Tests for canonical v4 parsing and token-aware chunking."""

import json
import sys
from pathlib import Path

import pytest

PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

from canonical_rag_ingest import (  # noqa: E402
    PARSER_VERSION,
    CHUNKER_VERSION,
    build_citation_url,
    build_milvus_records,
    chunk_canonical_document,
    estimate_tokens,
    extract_release_date,
    parse_and_chunk_file,
    parse_canonical_document,
    split_prose_by_tokens,
)

KUBEFLOW_DOC = """+++
title = "Install Kubeflow Pipelines"
description = "Standalone install guide"
weight = 42
+++

## Prerequisites

Install the [KFP SDK](https://pypi.org/project/kfp/) before continuing.

```python
import kfp
client = kfp.Client(host="<YOUR_KFP_ENDPOINT>")
```

NOTE: This guide assumes a running cluster.

### Configure access

| Component | Version |
| --- | --- |
| Pipelines | v2.3.0 |
| Metadata | v1.5.0 |
| Extra row with a much longer description to force table splitting when token target is tiny | v9.9.9 |

See [official docs](https://www.kubeflow.org/docs/pipelines/) for details.
"""

ALERT_DOC = """---
title: Alert Example
weight: 1
---

{{% alert title="Warning" color="warning" %}}
Do not delete production namespaces.
{{% /alert %}}
"""

HF_TOKEN_DOC = """+++
title = "GenAI setup"
+++

Use `access_token="<YOUR_HF_TOKEN>"` in your script.
"""

HTML_TABLE_DOC = """---
title: Release Components
weight: 100
---

## Component Versions

<table>
  <tr><td rowspan="2">AutoML WG</td><td>Katib</td><td>v0.19.0</td></tr>
  <tr><td>Trainer</td><td>v1.9.0</td></tr>
</table>
"""

RELEASE_DOC_HTML_TABLE = """+++
title = "Kubeflow Community Distribution 1.9"
description = "Information about the Kubeflow Community Distribution 1.9 release"
weight = 95
version = "1.9"
+++

## Kubeflow Community Distribution 1.9

<div class="table-responsive">
<table class="table table-bordered">
  <tbody>
    <tr>
      <th class="table-light">Release Date</th>
      <td>
        2024-07-22
      </td>
    </tr>
  </tbody>
</table>
</div>
"""

RELEASE_DOC_FRONTMATTER_DATE = """+++
title = "Kubeflow Community Distribution 1.8"
description = "Information about the Kubeflow Community Distribution 1.8 release"
weight = 96
version = "1.8"
release_date = "2024-01-15"
+++

## Kubeflow Community Distribution 1.8

No release table on this page.
"""

RELEASE_DOC_MISSING_DATE = """+++
title = "Kubeflow Community Distribution 9.9"
description = "Draft release page"
weight = 50
version = "9.9"
+++

## Kubeflow Community Distribution 9.9

Release timeline is TBD.
"""

RELEASE_DOC_INVALID_FRONTMATTER_DATE = """+++
title = "Kubeflow Community Distribution 9.8"
description = "Draft release page"
weight = 51
version = "9.8"
release_date = "TBD"
+++

## Kubeflow Community Distribution 9.8

<div class="table-responsive">
<table class="table table-bordered">
  <tbody>
    <tr>
      <th class="table-light">Release Date</th>
      <td>
        not-a-date
      </td>
    </tr>
  </tbody>
</table>
</div>
"""

RELEASE_DOC_GFM_TABLE = """+++
title = "Kubeflow Community Distribution 1.7"
description = "Information about the Kubeflow Community Distribution 1.7 release"
weight = 97
version = "1.7"
+++

## Kubeflow Community Distribution 1.7

| Release Date | 2023-09-18 |
| --- | --- |
"""

DOC_WITH_STRAY_ISO_DATE = """+++
title = "Install Kubeflow Pipelines"
description = "Standalone install guide"
weight = 42
+++

## Prerequisites

The cluster was provisioned on 2024-07-22 before continuing.
"""


class TestParseCanonicalDocument:
    def test_frontmatter_and_title(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        assert parsed.parser_version == PARSER_VERSION
        assert parsed.title == "Install Kubeflow Pipelines"
        assert parsed.weight == 42
        assert parsed.description == "Standalone install guide"

    def test_heading_hierarchy_and_section_path(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        paths = [section.section_path for section in parsed.sections if section.heading]
        assert "Install Kubeflow Pipelines > Prerequisites" in paths
        assert "Install Kubeflow Pipelines > Prerequisites > Configure access" in paths

    def test_preserves_code_fence(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        code_blocks = [
            block
            for section in parsed.sections
            for block in section.blocks
            if block.block_type == "code_fence"
        ]
        assert len(code_blocks) == 1
        assert "import kfp" in code_blocks[0].content
        assert code_blocks[0].language == "python"
        assert "<YOUR_KFP_ENDPOINT>" in code_blocks[0].content

    def test_preserves_links_in_prose(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        prose_blocks = [
            block
            for section in parsed.sections
            for block in section.blocks
            if block.block_type == "prose"
        ]
        assert prose_blocks
        assert any("[KFP SDK](https://pypi.org/project/kfp/)" in block.content for block in prose_blocks)
        assert any(link.url == "https://pypi.org/project/kfp/" for block in prose_blocks for link in block.links)

    def test_gfm_table_block(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        tables = [
            block
            for section in parsed.sections
            for block in section.blocks
            if block.block_type == "table"
        ]
        assert len(tables) == 1
        assert "Pipelines" in tables[0].content
        assert tables[0].table_format == "gfm"

    def test_admonition_shortcode_expansion(self):
        parsed = parse_canonical_document(ALERT_DOC)
        prose = [
            block.content
            for section in parsed.sections
            for block in section.blocks
            if block.block_type == "prose"
        ]
        assert prose
        assert "WARNING: Do not delete production namespaces." in prose[0]

    def test_inline_code_survives_html_like_tokens(self):
        parsed = parse_canonical_document(HF_TOKEN_DOC)
        prose = [
            block.content
            for section in parsed.sections
            for block in section.blocks
            if block.block_type == "prose"
        ]
        assert prose
        assert "<YOUR_HF_TOKEN>" in prose[0]

    def test_html_table_becomes_table_block(self):
        parsed = parse_canonical_document(HTML_TABLE_DOC)
        tables = [
            block
            for section in parsed.sections
            for block in section.blocks
            if block.block_type == "table"
        ]
        assert len(tables) == 1
        assert "Katib" in tables[0].content
        assert "AutoML WG" in tables[0].content

    def test_to_dict_is_json_serializable(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        json.dumps(parsed.to_dict())


class TestChunking:
    def test_section_first_code_chunk_is_atomic(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        chunks = chunk_canonical_document(parsed, target_tokens=350, overlap_tokens=50)
        code_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "code"]
        assert len(code_chunks) == 1
        assert "import kfp" in code_chunks[0]["content_text"]
        assert code_chunks[0]["section_path"].endswith("Prerequisites")

    def test_prose_chunks_carry_section_metadata(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        chunks = chunk_canonical_document(parsed, target_tokens=350, overlap_tokens=50)
        text_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "text"]
        assert text_chunks
        assert all(chunk["parser_version"] == PARSER_VERSION for chunk in chunks)
        assert all(chunk["chunker_version"] == CHUNKER_VERSION for chunk in chunks)
        assert all(chunk["section_path"] for chunk in text_chunks)

    def test_table_split_repeats_header(self):
        parsed = parse_canonical_document(KUBEFLOW_DOC)
        chunks = chunk_canonical_document(parsed, target_tokens=20, overlap_tokens=5)
        table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] in {"table", "table_row"}]
        assert len(table_chunks) >= 2
        for chunk in table_chunks:
            assert "Component" in chunk["content_text"]
            assert "Version" in chunk["content_text"]

    def test_split_prose_by_tokens_overlap(self):
        text = "word " * 800
        chunks = split_prose_by_tokens(text, target_tokens=100, overlap_tokens=20)
        assert len(chunks) > 1
        assert all(estimate_tokens(chunk) <= 150 for chunk in chunks)


class TestMilvusRecords:
    def test_record_fields_match_schema(self):
        file_data = {
            "path": "content/en/docs/components/pipelines/install.md",
            "file_name": "install.md",
            "content": KUBEFLOW_DOC,
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        required = {
            "file_unique_id",
            "repo_name",
            "file_path",
            "file_name",
            "citation_url",
            "chunk_index",
            "content_text",
            "title",
            "weight",
            "doc_type",
            "version",
            "release_date",
            "chunk_type",
            "section_path",
            "heading",
            "doc_status",
            "parser_version",
            "chunker_version",
        }
        for record in records:
            assert required.issubset(record.keys())
            assert len(record["content_text"]) <= 2000
            assert record["file_unique_id"] == "kubeflow/website:content/en/docs/components/pipelines/install.md"
            assert record["release_date"] is None

    def test_release_date_from_html_table(self):
        file_data = {
            "path": "content/en/docs/kubeflow-distribution/releases/kubeflow-1.9.md",
            "file_name": "kubeflow-1.9.md",
            "content": RELEASE_DOC_HTML_TABLE,
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        assert all(record["doc_type"] == "release" for record in records)
        assert all(record["version"] == "1.9" for record in records)
        assert all(record["release_date"] == 1721606400 for record in records)

    def test_release_date_from_frontmatter(self):
        file_data = {
            "path": "content/en/docs/kubeflow-distribution/releases/kubeflow-1.8.md",
            "file_name": "kubeflow-1.8.md",
            "content": RELEASE_DOC_FRONTMATTER_DATE,
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        assert all(record["doc_type"] == "release" for record in records)
        assert all(record["release_date"] == 1705276800 for record in records)

    def test_release_date_from_gfm_table(self):
        file_data = {
            "path": "content/en/docs/kubeflow-distribution/releases/kubeflow-1.7.md",
            "file_name": "kubeflow-1.7.md",
            "content": RELEASE_DOC_GFM_TABLE,
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        assert all(record["release_date"] == 1694995200 for record in records)

    def test_release_date_none_when_missing(self):
        file_data = {
            "path": "content/en/docs/kubeflow-distribution/releases/kubeflow-9.9.md",
            "file_name": "kubeflow-9.9.md",
            "content": RELEASE_DOC_MISSING_DATE,
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        assert all(record["doc_type"] == "release" for record in records)
        assert all(record["release_date"] is None for record in records)

    def test_release_date_none_for_invalid_dates(self):
        file_data = {
            "path": "content/en/docs/kubeflow-distribution/releases/kubeflow-9.8.md",
            "file_name": "kubeflow-9.8.md",
            "content": RELEASE_DOC_INVALID_FRONTMATTER_DATE,
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        assert all(record["release_date"] is None for record in records)

    def test_non_release_doc_does_not_extract_stray_iso_date(self):
        file_data = {
            "path": "content/en/docs/components/pipelines/install.md",
            "file_name": "install.md",
            "content": DOC_WITH_STRAY_ISO_DATE,
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records
        assert all(record["doc_type"] == "documentation" for record in records)
        assert all(record["release_date"] is None for record in records)

    def test_extract_release_date_helper(self):
        assert (
            extract_release_date(
                doc_type="release",
                source_text=RELEASE_DOC_HTML_TABLE,
                frontmatter={"version": "1.9"},
            )
            == 1721606400
        )
        assert (
            extract_release_date(
                doc_type="documentation",
                source_text=DOC_WITH_STRAY_ISO_DATE,
                frontmatter={},
            )
            is None
        )

    def test_citation_url_drops_md_and_index(self):
        url = build_citation_url("content/en/docs/pipelines/_index.md", "https://www.kubeflow.org/docs")
        assert url == "https://www.kubeflow.org/docs/pipelines"

    def test_deprecated_doc_status(self):
        file_data = {
            "path": "content/en/docs/components/pipelines/legacy-v1/overview.md",
            "file_name": "overview.md",
            "content": "+++\ntitle='Legacy'\n+++\n\n## Old docs\nBody",
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert records[0]["doc_status"] == "deprecated"

    def test_nav_index_empty_body(self):
        file_data = {
            "path": "content/en/docs/kserve/_index.md",
            "file_name": "_index.md",
            "content": "+++\ntitle = 'KServe'\ndescription = 'Serverless inference'\nweight = 3\n+++\n",
        }
        records = build_milvus_records(
            file_data,
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert len(records) == 1
        assert records[0]["chunk_type"] == "nav"
        assert "Section: KServe" in records[0]["content_text"]

    def test_parse_and_chunk_file_wrapper(self):
        payload = parse_and_chunk_file(
            {
                "path": "content/en/docs/pipelines/install.md",
                "file_name": "install.md",
                "content": KUBEFLOW_DOC,
            },
            repo_name="kubeflow/website",
            base_url="https://www.kubeflow.org/docs",
        )
        assert payload["parser_version"] == PARSER_VERSION
        assert payload["chunker_version"] == CHUNKER_VERSION
        assert payload["canonical"]["title"] == "Install Kubeflow Pipelines"
        assert payload["chunks"]
