"""Tests for latest-query retrieval, distribution docs ingest, pruning and ingest schedules."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("pymilvus", MagicMock())
ROOT = Path(__file__).resolve().parents[1] / "docs-agent-mcp"
for path in (ROOT / "mcp-server", ROOT / "pipelines", ROOT / "pipelines" / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import canonical_rag_ingest  # noqa: E402
import ingest_schedule  # noqa: E402
import intent_router  # noqa: E402
import milvus_search  # noqa: E402
import milvus_store  # noqa: E402
import reranker_client  # noqa: E402


def hit(doc_id, doc_type="documentation", release_date=None, text=""):
    return {
        "id": doc_id,
        "distance": 1.0,
        "entity": {
            "doc_type": doc_type,
            "release_date": release_date,
            "content_text": text or f"chunk {doc_id}",
            "file_path": f"docs/{doc_id}.md",
        },
    }


# --- latest/current queries ---------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Are the new workspaces in the new release?",
        "latest Kubeflow release",
        "what changed in the last release",
    ],
)
def test_release_questions_also_search_general_docs(query):
    plan = intent_router.pick_search_plan(query)
    assert plan.intent == "temporal"
    assert plan.add_general_hits is True


def test_new_alone_is_not_a_release_question():
    assert intent_router.pick_search_plan("create a new notebook server").intent == "general"


def test_merge_keeps_release_notes_first_and_shares_slots():
    release = [hit(f"r{i}", "release", 100 - i) for i in range(5)]
    general = [hit(f"g{i}") for i in range(5)]
    merged = intent_router.merge_release_and_general_hits(release, general, 5)
    assert [h["id"] for h in merged] == ["r0", "r1", "r2", "g0", "g1"]


def test_merge_fills_unused_slots_and_drops_duplicates():
    release = [hit("r0", "release", 1)]
    general = [hit("r0", "release", 1), hit("g0"), hit("g1"), hit("g2"), hit("g3"), hit("g4")]
    merged = intent_router.merge_release_and_general_hits(release, general, 5)
    assert [h["id"] for h in merged] == ["r0", "g0", "g1", "g2", "g3"]
    assert intent_router.merge_release_and_general_hits([], general[1:], 3) == general[1:4]


@pytest.fixture
def search(monkeypatch):
    """milvus_search with Milvus, embeddings and the reranker stubbed out."""
    calls = SimpleNamespace(bm25=[], hybrid=[])
    release = [hit(f"r{i}", "release", 100 - i) for i in range(6)]
    general = [hit("workspaces-guide"), hit("katib")]

    def bm25(collection, query, limit, fields, filter_expr=""):
        calls.bm25.append(filter_expr)
        return release

    def hybrid(collection, query, embedding, limit, fields, filter_expr="", candidate_depth=None):
        calls.hybrid.append(filter_expr)
        return general

    monkeypatch.setattr(milvus_search, "_load", lambda name: None)
    monkeypatch.setattr(milvus_search, "collection_has_bm25", lambda name: True)
    monkeypatch.setattr(milvus_search, "collection_has_release_fields", lambda name: True)
    monkeypatch.setattr(milvus_search, "_require_embedding", lambda query: [0.0])
    monkeypatch.setattr(milvus_search, "bm25_search", bm25)
    monkeypatch.setattr(milvus_search, "hybrid_search", hybrid)
    monkeypatch.setattr(reranker_client, "rerank_hits", lambda query, hits: hits)
    return calls


def test_latest_query_returns_release_notes_and_component_docs(search):
    hits, meta = milvus_search.search_docs_auto("Are the new workspaces in the new release?", 5, [])
    assert [h["id"] for h in hits] == ["r0", "r1", "r2", "workspaces-guide", "katib"]
    assert search.bm25 == ['doc_type == "release"']
    assert search.hybrid == [""]
    assert meta["intent"] == "temporal"
    assert meta["general_hits"] == 2


def test_latest_query_survives_a_failed_general_search(search, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(milvus_search, "hybrid_search", broken)
    hits, meta = milvus_search.search_docs_auto("latest Kubeflow release", 5, [])
    assert [h["id"] for h in hits] == ["r0", "r1", "r2", "r3", "r4"]
    assert meta["general_hits"] == 0


def test_reranker_gets_more_candidates_than_top_k(search, monkeypatch):
    limits = []

    def hybrid(collection, query, embedding, limit, fields, filter_expr="", candidate_depth=None):
        limits.append(limit)
        return [hit(f"g{i}") for i in range(limit)]

    monkeypatch.setattr(milvus_search, "hybrid_search", hybrid)
    monkeypatch.setattr(reranker_client, "rerank_hits", lambda query, hits: list(reversed(hits)))

    monkeypatch.setenv("RERANKER_URL", "")
    hits, _ = milvus_search.search_docs_auto("Kubeflow Workspaces overview", 5, [])
    assert limits == [5] and len(hits) == 5

    monkeypatch.setenv("RERANKER_URL", "http://reranker/rerank")
    hits, _ = milvus_search.search_docs_auto("Kubeflow Workspaces overview", 5, [])
    assert limits == [5, milvus_search.RERANK_CANDIDATE_DEPTH]
    assert [h["id"] for h in hits] == ["g19", "g18", "g17", "g16", "g15"]


# --- distribution docs ---------------------------------------------------------------

README = """[![FOSSA Status](https://app.fossa.com/badge.svg)](https://app.fossa.com/projects/x)

# Kubeflow Community Distribution

## Overview

All components are deployable with `kustomize`. In addition, experimental Helm charts for
selected components are available under `experimental/helm/`.
"""


def test_readme_is_indexed_as_distribution_with_a_title_and_github_citation():
    records = canonical_rag_ingest.build_milvus_records(
        {"path": "README.md", "content": README},
        repo_name="community-distribution",
        base_url="https://github.com/kubeflow/community-distribution/blob/master",
        doc_type="distribution",
    )
    assert records
    assert {r["doc_type"] for r in records} == {"distribution"}
    assert {r["title"] for r in records} == {"Kubeflow Community Distribution"}
    assert records[0]["citation_url"] == "https://github.com/kubeflow/community-distribution/blob/master/README.md"
    assert records[0]["file_unique_id"] == "community-distribution:README.md"
    assert any("Helm charts" in r["content_text"] for r in records)
    assert not any(canonical_rag_ingest.is_badge_only_chunk(r["content_text"]) for r in records)


def test_website_pages_still_infer_their_doc_type():
    records = canonical_rag_ingest.build_milvus_records(
        {
            "path": "content/en/docs/kubeflow-distribution/releases/kubeflow-26.03.md",
            "content": "---\ntitle: Kubeflow 26.03\n---\n\nRelease notes for this version of Kubeflow.\n",
        },
        repo_name="website",
        base_url="https://www.kubeflow.org/docs",
    )
    assert {r["doc_type"] for r in records} == {"release"}
    assert {r["title"] for r in records} == {"Kubeflow 26.03"}


@pytest.mark.parametrize(
    ("text", "badge_only"),
    [
        ("[Figure: FOSSA Status](https://app.fossa.com/projects/x?ref=badge_shield)", True),
        ("Figure: architecture diagram", True),
        ("Kubeflow Workspaces provides on-demand development environments.", False),
        ("[Figure: logo](https://x/y.png)\n\nKubeflow Hub stores models and their metadata.", False),
    ],
)
def test_badge_only_chunks_are_detected(text, badge_only):
    assert canonical_rag_ingest.is_badge_only_chunk(text) is badge_only


# --- store: collection names and pruning --------------------------------------------


@pytest.mark.parametrize("name", ["kubeflow_docs", "kubeflow_docs_staging", "kubeflow_docs_pr12"])
def test_docs_collection_and_test_copies_are_allowed(name):
    milvus_store.validate_production_collection_name(name)


@pytest.mark.parametrize("name", ["issues_rag", "kubeflow_docs_", "kubeflow_docs_a;drop", "", "other_kubeflow_docs"])
def test_other_collections_are_refused(name):
    with pytest.raises(ValueError):
        milvus_store.validate_production_collection_name(name)


def test_clean_rebuild_confirmation_names_the_target_collection(monkeypatch):
    monkeypatch.setenv(milvus_store.MAINTENANCE_LOCK_ENV, "lock")
    with pytest.raises(ValueError):
        milvus_store.validate_clean_rebuild_gates(
            clean_rebuild=True,
            clean_rebuild_confirmation="DROP kubeflow_docs",
            maintenance_lock_token="lock",
            collection_name="kubeflow_docs_staging",
        )
    milvus_store.validate_clean_rebuild_gates(
        clean_rebuild=True,
        clean_rebuild_confirmation="DROP kubeflow_docs_staging",
        maintenance_lock_token="lock",
        collection_name="kubeflow_docs_staging",
    )


class FakeIterator:
    def __init__(self, rows):
        self.pages = [rows, []]

    def next(self):
        return self.pages.pop(0)

    def close(self):
        pass


def fake_milvus(indexed_ids):
    client = MagicMock()
    client.query_iterator.return_value = FakeIterator([{"document_id": doc_id} for doc_id in indexed_ids])
    client.delete.return_value = {"delete_count": 1}
    return client


def test_stale_documents_are_found_under_the_source_prefix():
    indexed = [f"website:content/en/docs/p{i}.md" for i in range(10)]
    client = fake_milvus(indexed)
    stale = milvus_store.find_stale_document_ids(
        client,
        "kubeflow_docs",
        prune_prefix="website:content/en/docs/",
        current_ids=set(indexed[:9]),
    )
    assert stale == ["website:content/en/docs/p9.md"]
    assert client.query_iterator.call_args.kwargs["filter"] == 'document_id like "website:content/en/docs/%"'


def test_prune_refuses_when_too_much_of_the_source_is_missing():
    indexed = [f"website:content/en/docs/p{i}.md" for i in range(10)]
    with pytest.raises(RuntimeError, match="Refusing to prune 5 of 10"):
        milvus_store.find_stale_document_ids(
            fake_milvus(indexed),
            "kubeflow_docs",
            prune_prefix="website:content/en/docs/",
            current_ids=set(indexed[:5]),
        )


def test_first_ingest_of_a_source_has_nothing_to_prune():
    stale = milvus_store.find_stale_document_ids(
        fake_milvus([]),
        "kubeflow_docs",
        prune_prefix="community-distribution:",
        current_ids={"community-distribution:README.md"},
    )
    assert stale == []


# --- recurring runs -------------------------------------------------------------------


def write_config(tmp_path, schedules):
    path = tmp_path / "schedules.json"
    path.write_text(json.dumps({"schedules": schedules}))
    return str(path)


def test_schedule_config_is_validated(tmp_path):
    good = {"name": "docs-website", "pipeline": "docs", "cron": "0 0 2 * * *", "arguments": {}}
    assert ingest_schedule.load_config(write_config(tmp_path, [good]))["experiment"] == "flo-ingest"
    for bad in (
        {**good, "cron": "0 2 * * *"},  # five fields: KFP wants seconds first
        {**good, "pipeline": "unknown"},
        {**good, "arguments": []},
    ):
        with pytest.raises(ValueError):
            ingest_schedule.load_config(write_config(tmp_path, [bad]))
    with pytest.raises(ValueError):
        ingest_schedule.load_config(write_config(tmp_path, [good, good]))


def test_register_creates_new_recurring_runs_before_removing_old_ones(tmp_path, monkeypatch):
    config = ingest_schedule.load_config(
        write_config(
            tmp_path,
            [
                {"name": "docs-website", "pipeline": "docs", "cron": "0 0 2 * * *"},
                {
                    "name": "docs-distribution",
                    "pipeline": "docs",
                    "cron": "0 30 2 * * *",
                    "arguments": {"doc_type": "distribution"},
                },
            ],
        )
    )
    monkeypatch.setattr(ingest_schedule, "compile_pipeline", lambda key, out_dir: f"{out_dir}/{key}.yaml")
    events = []
    client = MagicMock()
    client.create_experiment.return_value = SimpleNamespace(experiment_id="exp")
    client.get_pipeline_id.return_value = "pipe"
    client.upload_pipeline_version.return_value = SimpleNamespace(pipeline_version_id="ver")
    client.list_recurring_runs.return_value = SimpleNamespace(
        recurring_runs=[SimpleNamespace(recurring_run_id="old", display_name="docs-website")],
        next_page_token="",
    )
    client.create_recurring_run.side_effect = lambda **kw: events.append(("create", kw["job_name"]))
    client.delete_recurring_run.side_effect = lambda run_id: events.append(("delete", run_id))

    ingest_schedule.register(client, config, "abc1234-1")

    assert events == [("create", "docs-website"), ("create", "docs-distribution"), ("delete", "old")]
    client.upload_pipeline_version.assert_called_once()  # both schedules share the docs pipeline
    created = client.create_recurring_run.call_args_list[1].kwargs
    assert created["params"] == {"doc_type": "distribution"}
    assert created["max_concurrency"] == 1 and created["no_catchup"] is True
    assert (created["pipeline_id"], created["version_id"]) == ("pipe", "ver")


def test_set_overrides_parse_json_values():
    assert ingest_schedule.parse_overrides(["collection_name=kubeflow_docs_staging", "prune_removed=false"]) == {
        "collection_name": "kubeflow_docs_staging",
        "prune_removed": False,
    }
