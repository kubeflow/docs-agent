"""Filter arguments on the MCP search tools are validated like the query (#275)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

MCP_SERVER_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "mcp-server"

sys.modules.setdefault("pymilvus", MagicMock())
sys.path.insert(0, str(MCP_SERVER_DIR))
spec = importlib.util.spec_from_file_location("docs_agent_mcp_server_filters", MCP_SERVER_DIR / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.fixture
def search_collection(monkeypatch):
    mock = MagicMock(return_value=[])
    monkeypatch.setattr(server, "search_collection", mock)
    return mock


def _text(result) -> str:
    return result.content[0].text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repo": 'kubeflow" or 1==1'}, "Invalid repo filter value"),
        ({"state": "open "}, "Invalid state filter value"),
    ],
)
def test_search_github_issues_rejects_invalid_filter(search_collection, kwargs, message):
    result = server.search_github_issues("pipelines", **kwargs)

    assert not result.is_error
    assert _text(result).startswith(f"Search rejected: {message}")
    search_collection.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resource_kind": "Deployment'"}, "Invalid resource_kind filter value"),
        ({"repo": "kubeflow/sdk; drop"}, "Invalid repo filter value"),
    ],
)
def test_search_kubeflow_code_rejects_invalid_filter(search_collection, kwargs, message):
    result = server.search_kubeflow_code("notebook webhook", **kwargs)

    assert not result.is_error
    assert _text(result).startswith(f"Search rejected: {message}")
    search_collection.assert_not_called()


def test_valid_filters_still_reach_search(search_collection):
    server.search_github_issues("pipelines", repo="kubeflow/pipelines", state="open")
    server.search_kubeflow_code("webhook", resource_kind="Deployment", repo="kubeflow/notebooks")

    issues_filter = search_collection.call_args_list[0].kwargs["filter_expr"]
    code_filter = search_collection.call_args_list[1].kwargs["filter_expr"]
    assert issues_filter == 'repo_name == "kubeflow/pipelines" and issue_state == "open"'
    assert code_filter == "resource_kind == 'Deployment' and repo_name == \"kubeflow/notebooks\""
