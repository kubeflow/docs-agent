import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path


class _FastMCPStub:
    def __init__(self, _name: str):
        self._name = _name

    def tool(self):
        def _decorator(func):
            return func

        return _decorator

    def run(self, **_kwargs):
        return None


def _load_server_module():
    module_name = "mcp_server_under_test"
    module_path = Path(__file__).resolve().parents[1] / "server.py"

    fastmcp_stub = types.ModuleType("fastmcp")
    fastmcp_stub.FastMCP = _FastMCPStub

    pymilvus_stub = types.ModuleType("pymilvus")

    class _MilvusClientStub:
        def __init__(self, **_kwargs):
            pass

    pymilvus_stub.MilvusClient = _MilvusClientStub

    sentence_stub = types.ModuleType("sentence_transformers")

    class _SentenceTransformerStub:
        def __init__(self, *_args, **_kwargs):
            pass

    sentence_stub.SentenceTransformer = _SentenceTransformerStub

    sys.modules["fastmcp"] = fastmcp_stub
    sys.modules["pymilvus"] = pymilvus_stub
    sys.modules["sentence_transformers"] = sentence_stub

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_init_is_thread_safe_and_idempotent(monkeypatch, caplog):
    server = _load_server_module()

    model_init_count = 0
    client_init_count = 0
    init_count_lock = threading.Lock()

    class FakeModel:
        pass

    class FakeClient:
        pass

    def fake_sentence_transformer(_model_name):
        nonlocal model_init_count
        with init_count_lock:
            model_init_count += 1
        return FakeModel()

    def fake_milvus_client(**_kwargs):
        nonlocal client_init_count
        with init_count_lock:
            client_init_count += 1
        return FakeClient()

    monkeypatch.setattr(server, "SentenceTransformer", fake_sentence_transformer)
    monkeypatch.setattr(server, "MilvusClient", fake_milvus_client)

    server.model = None
    server.client = None
    server._initialized = False

    workers = 32
    barrier = threading.Barrier(workers)
    errors = []
    seen_models = []
    seen_clients = []
    seen_lock = threading.Lock()

    def worker():
        try:
            barrier.wait()
            server._init()
            with seen_lock:
                seen_models.append(server.model)
                seen_clients.append(server.client)
        except Exception as exc:  # pragma: no cover
            with seen_lock:
                errors.append(exc)

    with caplog.at_level(logging.INFO, logger=server.__name__):
        threads = [threading.Thread(target=worker) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

    assert not errors
    assert model_init_count == 1
    assert client_init_count == 1
    assert server._initialized is True

    first_model = seen_models[0]
    first_client = seen_clients[0]

    assert first_model is not None
    assert first_client is not None
    assert all(model is first_model for model in seen_models)
    assert all(client is first_client for client in seen_clients)

    messages = [record.getMessage() for record in caplog.records]
    assert messages.count("Initializing shared MCP resources") == 1
    assert messages.count("Shared MCP resources initialized") == 1
