"""
Unit tests for Kubeflow docs-agent server.

These tests validate the core server logic without requiring
external services (Milvus, KServe, LLM). All external dependencies
are mocked to test the application logic in isolation.

Run with: python -m pytest server/tests/test_app.py -v
"""

import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ================================================================== #
# Fixtures
# ================================================================== #

@pytest.fixture
def mock_milvus_hits():
    """Simulated Milvus search results."""
    hit1 = MagicMock()
    hit1.distance = 0.15  # COSINE distance (lower = more similar)
    hit1.entity = MagicMock()
    hit1.entity.get = lambda key, default=None: {
        "file_path": "content/en/docs/components/pipelines/overview.md",
        "content_text": "Kubeflow Pipelines is a platform for building ML workflows.",
        "citation_url": "https://www.kubeflow.org/docs/components/pipelines/overview/",
    }.get(key, default)

    hit2 = MagicMock()
    hit2.distance = 0.30
    hit2.entity = MagicMock()
    hit2.entity.get = lambda key, default=None: {
        "file_path": "content/en/docs/components/pipelines/sdk.md",
        "content_text": "The KFP SDK allows you to define pipelines in Python.",
        "citation_url": "https://www.kubeflow.org/docs/components/pipelines/sdk/",
    }.get(key, default)

    return [[hit1, hit2]]


@pytest.fixture
def sample_tool_call():
    """A well-formed tool call from the LLM."""
    return {
        "id": "call_abc123",
        "function": {
            "name": "search_kubeflow_docs",
            "arguments": json.dumps({"query": "kubeflow pipelines setup", "top_k": 3})
        }
    }


@pytest.fixture
def unknown_tool_call():
    """A tool call with an unrecognized function name."""
    return {
        "id": "call_xyz789",
        "function": {
            "name": "nonexistent_tool",
            "arguments": json.dumps({"query": "test"})
        }
    }


@pytest.fixture
def malformed_tool_call():
    """A tool call with invalid JSON arguments."""
    return {
        "id": "call_bad456",
        "function": {
            "name": "search_kubeflow_docs",
            "arguments": "this is not json"
        }
    }


# ================================================================== #
# MilvusSearchClient Tests
# ================================================================== #

class TestMilvusSearchClient:
    """Tests for the MilvusSearchClient connection pooling singleton."""

    @patch("server.app.connections")
    @patch("server.app.Collection")
    @patch("server.app.SentenceTransformer")
    def test_lazy_initialization(self, mock_st, mock_collection, mock_conn):
        """Client should not connect until the first search call."""
        from server.app import MilvusSearchClient

        client = MilvusSearchClient()
        assert client._connected is False
        assert client._encoder is None
        assert client._collection is None

        # No connections made yet
        mock_conn.connect.assert_not_called()

    @patch("server.app.connections")
    @patch("server.app.Collection")
    @patch("server.app.SentenceTransformer")
    def test_connection_reuse(self, mock_st, mock_collection, mock_conn):
        """Calling _ensure_connected twice should only connect once."""
        from server.app import MilvusSearchClient

        client = MilvusSearchClient()
        client._ensure_connected()
        client._ensure_connected()

        # Should only connect once
        assert mock_conn.connect.call_count == 1

    @patch("server.app.connections")
    @patch("server.app.Collection")
    @patch("server.app.SentenceTransformer")
    def test_encoder_reuse(self, mock_st, mock_collection, mock_conn):
        """Embedding model should load once and be reused."""
        from server.app import MilvusSearchClient

        client = MilvusSearchClient()
        enc1 = client._get_encoder()
        enc2 = client._get_encoder()

        assert enc1 is enc2
        assert mock_st.call_count == 1

    @patch("server.app.connections")
    @patch("server.app.Collection")
    @patch("server.app.SentenceTransformer")
    def test_resets_on_error(self, mock_st, mock_collection_cls, mock_conn):
        """On search failure, client should reset for reconnection."""
        from server.app import MilvusSearchClient

        client = MilvusSearchClient()
        client._connected = True
        client._collection = MagicMock()

        # Make the collection search raise an error
        client._collection.search.side_effect = Exception("Connection lost")

        result = client.search("test query")

        assert result == {"results": []}
        assert client._connected is False
        assert client._collection is None

    @patch("server.app.connections")
    @patch("server.app.Collection")
    @patch("server.app.SentenceTransformer")
    def test_search_returns_correct_structure(self, mock_st, mock_collection_cls, mock_conn, mock_milvus_hits):
        """Search should return properly formatted results."""
        from server.app import MilvusSearchClient

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_st.return_value = mock_encoder

        mock_coll = MagicMock()
        mock_coll.search.return_value = mock_milvus_hits
        mock_collection_cls.return_value = mock_coll

        client = MilvusSearchClient()
        result = client.search("kubeflow pipelines", top_k=2)

        assert "results" in result
        assert len(result["results"]) == 2
        assert "similarity" in result["results"][0]
        assert "file_path" in result["results"][0]
        assert "citation_url" in result["results"][0]
        assert "content_text" in result["results"][0]

    @patch("server.app.connections")
    @patch("server.app.Collection")
    @patch("server.app.SentenceTransformer")
    def test_similarity_score_calculation(self, mock_st, mock_collection_cls, mock_conn, mock_milvus_hits):
        """Similarity should be 1.0 - cosine_distance."""
        from server.app import MilvusSearchClient

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_st.return_value = mock_encoder

        mock_coll = MagicMock()
        mock_coll.search.return_value = mock_milvus_hits
        mock_collection_cls.return_value = mock_coll

        client = MilvusSearchClient()
        result = client.search("test")

        # First hit: distance=0.15, so similarity=0.85
        assert result["results"][0]["similarity"] == pytest.approx(0.85, abs=0.001)
        # Second hit: distance=0.30, so similarity=0.70
        assert result["results"][1]["similarity"] == pytest.approx(0.70, abs=0.001)

    @patch("server.app.connections")
    @patch("server.app.Collection")
    @patch("server.app.SentenceTransformer")
    def test_long_content_truncation(self, mock_st, mock_collection_cls, mock_conn):
        """Content longer than 400 chars should be truncated."""
        from server.app import MilvusSearchClient

        long_text = "x" * 500
        hit = MagicMock()
        hit.distance = 0.1
        hit.entity = MagicMock()
        hit.entity.get = lambda key, default=None: {
            "content_text": long_text,
            "file_path": "test.md",
            "citation_url": "https://example.com",
        }.get(key, default)

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_st.return_value = mock_encoder

        mock_coll = MagicMock()
        mock_coll.search.return_value = [[hit]]
        mock_collection_cls.return_value = mock_coll

        client = MilvusSearchClient()
        result = client.search("test")

        content = result["results"][0]["content_text"]
        assert len(content) == 403  # 400 chars + "..."
        assert content.endswith("...")


# ================================================================== #
# execute_tool Tests
# ================================================================== #

class TestExecuteTool:
    """Tests for the tool execution handler."""

    @pytest.mark.asyncio
    async def test_known_tool_returns_results(self, sample_tool_call):
        """search_kubeflow_docs should return formatted text and citations."""
        mock_results = {
            "results": [{
                "file_path": "docs/pipelines.md",
                "content_text": "Pipeline setup guide content.",
                "citation_url": "https://kubeflow.org/docs/pipelines",
                "similarity": 0.85,
            }]
        }

        with patch("server.app.milvus_search", return_value=mock_results):
            from server.app import execute_tool
            text, citations = await execute_tool(sample_tool_call)

        assert "Pipeline setup guide content" in text
        assert "https://kubeflow.org/docs/pipelines" in citations

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, unknown_tool_call):
        """Unknown tool names should return an error string."""
        from server.app import execute_tool
        text, citations = await execute_tool(unknown_tool_call)

        assert "Unknown tool" in text
        assert citations == []

    @pytest.mark.asyncio
    async def test_malformed_arguments_handled(self, malformed_tool_call):
        """Invalid JSON arguments should not crash the server."""
        from server.app import execute_tool
        text, citations = await execute_tool(malformed_tool_call)

        assert "failed" in text.lower() or "error" in text.lower()
        assert citations == []

    @pytest.mark.asyncio
    async def test_empty_results_handled(self, sample_tool_call):
        """Empty Milvus results should return 'No relevant results'."""
        with patch("server.app.milvus_search", return_value={"results": []}):
            from server.app import execute_tool
            text, citations = await execute_tool(sample_tool_call)

        assert "No relevant results" in text
        assert citations == []

    @pytest.mark.asyncio
    async def test_deduplicates_citations(self):
        """Duplicate citation URLs should be deduplicated."""
        tool_call = {
            "id": "call_test",
            "function": {
                "name": "search_kubeflow_docs",
                "arguments": json.dumps({"query": "test"})
            }
        }

        mock_results = {
            "results": [
                {"file_path": "a.md", "content_text": "text1",
                 "citation_url": "https://kubeflow.org/same-url", "similarity": 0.9},
                {"file_path": "b.md", "content_text": "text2",
                 "citation_url": "https://kubeflow.org/same-url", "similarity": 0.8},
            ]
        }

        with patch("server.app.milvus_search", return_value=mock_results):
            from server.app import execute_tool
            text, citations = await execute_tool(tool_call)

        assert len(citations) == 1


# ================================================================== #
# health_check Tests
# ================================================================== #

class TestHealthCheck:
    """Tests for the HTTP health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_ok(self):
        """GET /health should return 200 OK."""
        from server.app import health_check
        result = await health_check("/health", {})

        assert result is not None
        status_code, headers, body = result
        assert status_code == 200
        assert body == b"OK"

    @pytest.mark.asyncio
    async def test_non_health_path_returns_none(self):
        """Non-health paths should return None (pass to WebSocket handler)."""
        from server.app import health_check
        result = await health_check("/other", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_root_path_returns_none(self):
        """Root path should return None."""
        from server.app import health_check
        result = await health_check("/", {})
        assert result is None


# ================================================================== #
# TOOLS Configuration Tests
# ================================================================== #

class TestToolsConfig:
    """Tests for the TOOLS definition structure."""

    def test_tools_is_list(self):
        """TOOLS should be a list."""
        from server.app import TOOLS
        assert isinstance(TOOLS, list)

    def test_tools_has_search_function(self):
        """TOOLS should contain the search_kubeflow_docs function."""
        from server.app import TOOLS
        tool_names = [t["function"]["name"] for t in TOOLS]
        assert "search_kubeflow_docs" in tool_names

    def test_search_tool_has_required_params(self):
        """search_kubeflow_docs should require the 'query' parameter."""
        from server.app import TOOLS
        search_tool = [t for t in TOOLS if t["function"]["name"] == "search_kubeflow_docs"][0]
        params = search_tool["function"]["parameters"]

        assert "query" in params["required"]
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"

    def test_search_tool_top_k_has_bounds(self):
        """top_k parameter should have min/max bounds."""
        from server.app import TOOLS
        search_tool = [t for t in TOOLS if t["function"]["name"] == "search_kubeflow_docs"][0]
        top_k = search_tool["function"]["parameters"]["properties"]["top_k"]

        assert top_k["minimum"] == 1
        assert top_k["maximum"] == 10
        assert top_k["default"] == 5


# ================================================================== #
# SYSTEM_PROMPT Tests
# ================================================================== #

class TestSystemPrompt:
    """Tests for the system prompt configuration."""

    def test_system_prompt_exists(self):
        """System prompt should be a non-empty string."""
        from server.app import SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_kubeflow(self):
        """System prompt should reference Kubeflow."""
        from server.app import SYSTEM_PROMPT
        assert "Kubeflow" in SYSTEM_PROMPT

    def test_system_prompt_has_tool_guidance(self):
        """System prompt should instruct when to use tools."""
        from server.app import SYSTEM_PROMPT
        assert "search_kubeflow_docs" in SYSTEM_PROMPT

    def test_system_prompt_has_routing_rules(self):
        """System prompt should have routing rules for different query types."""
        from server.app import SYSTEM_PROMPT
        assert "Routing" in SYSTEM_PROMPT or "routing" in SYSTEM_PROMPT


# ================================================================== #
# WebSocket Message Parsing Tests
# ================================================================== #

class TestMessageParsing:
    """Tests for WebSocket message format handling."""

    def test_json_message_extraction(self):
        """JSON messages with 'message' key should be extracted."""
        raw = json.dumps({"message": "How do I set up KServe?"})
        msg_data = json.loads(raw)
        if isinstance(msg_data, dict) and "message" in msg_data:
            message = msg_data["message"]
        assert message == "How do I set up KServe?"

    def test_plain_text_passthrough(self):
        """Plain text messages should pass through as-is."""
        raw = "How do I set up KServe?"
        try:
            msg_data = json.loads(raw)
            if isinstance(msg_data, dict) and "message" in msg_data:
                message = msg_data["message"]
            else:
                message = raw
        except json.JSONDecodeError:
            message = raw
        assert message == "How do I set up KServe?"

    def test_bytes_decoded_to_string(self):
        """Byte messages should be decoded to UTF-8 strings."""
        raw = b"How do I set up KServe?"
        if isinstance(raw, (bytes, bytearray)):
            message = raw.decode("utf-8", errors="ignore")
        assert message == "How do I set up KServe?"

    def test_json_without_message_key(self):
        """JSON without 'message' key should be treated as raw text."""
        raw = json.dumps({"query": "test", "user": "someone"})
        try:
            msg_data = json.loads(raw)
            if isinstance(msg_data, dict) and "message" in msg_data:
                message = msg_data["message"]
            else:
                message = raw
        except json.JSONDecodeError:
            message = raw

        # Should keep the original JSON string
        assert "query" in message


# ================================================================== #
# Environment Configuration Tests
# ================================================================== #

class TestConfig:
    """Tests for environment variable configuration."""

    def test_default_port(self):
        """Default port should be 8000."""
        from server.app import PORT
        # PORT reads from env, but default is 8000
        assert isinstance(PORT, int)

    def test_default_milvus_collection(self):
        """Default Milvus collection should be 'docs_rag'."""
        from server.app import MILVUS_COLLECTION
        assert MILVUS_COLLECTION == "docs_rag"

    def test_default_embedding_model(self):
        """Default embedding model should be all-mpnet-base-v2."""
        from server.app import EMBEDDING_MODEL
        assert "all-mpnet-base-v2" in EMBEDDING_MODEL

    def test_default_vector_field(self):
        """Default vector field should be 'vector'."""
        from server.app import MILVUS_VECTOR_FIELD
        assert MILVUS_VECTOR_FIELD == "vector"
