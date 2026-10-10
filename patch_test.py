import re

with open("tests/test_docs_pipeline.py", "r") as f:
    content = f.read()

content = content.replace("import pytest", "import pytest\nfrom unittest.mock import MagicMock")

# Mock MilvusClient globally or in fake_pymilvus_module?
# Actually, since milvus_store is imported inside kubeflow-pipeline.py, we just patch it in sys.modules or monkeypatch it.
# Let's replace embedding_dim with clean_rebuild args.
content = re.sub(r'embedding_dim=\d+,?', 'clean_rebuild=False,\n            clean_rebuild_confirmation="",\n            maintenance_lock_token="",', content)
content = content.replace('chunk_size=', 'target_tokens=')
content = content.replace('chunk_overlap=', 'overlap_tokens=')
content = re.sub(r'max_tei_chars=\d+,?\n\s*', '', content)

# FakeEmbeddingResponse
content = content.replace(
    'class FakeEmbeddingResponse:\n        def raise_for_status(self):\n            return None\n\n        def json(self):\n            return [[0.0] * 768]',
    'class FakeEmbeddingResponse:\n        def __init__(self, inputs=None):\n            self.inputs = inputs or []\n        def raise_for_status(self):\n            return None\n        def json(self):\n            return [[0.0] * 768] * len(self.inputs)'
)
content = content.replace(
    'monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeEmbeddingResponse())',
    'monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeEmbeddingResponse(kwargs.get("json", {}).get("inputs", [])))'
)

# Json record assertions
content = content.replace(
    'record = json.loads(output_path.read_text())',
    'records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]\n    content_text = "\\n".join([r["content_text"] for r in records])'
)
content = content.replace('record["content_text"]', 'content_text')

# Skip mismatch
content = content.replace(
    'def test_store_rejects_embedding_dim_mismatch',
    '@pytest.mark.skip(reason="embedding_dim is no longer a parameter")\ndef test_store_rejects_embedding_dim_mismatch'
)

# The mocks
# We will inject a patch for milvus_store.MilvusClient right before module.store_milvus.python_func
# First, find module.store_milvus.python_func
mock_client_code = """
        import milvus_store
        mock_client = MagicMock()
        mock_client.has_collection.return_value = True
        mock_client.describe_collection.return_value = {"description": "v=4"}
        mock_client.query.return_value = [{"id": 1}, {"id": 2}]
        if "queried" in locals():
            mock_client.query.side_effect = lambda collection_name, filter, **kwargs: queried.append(filter) or [{"id": 1}, {"id": 2}]
        if "deleted" in locals():
            mock_client.delete.side_effect = lambda collection_name, filter, **kwargs: deleted.append(filter) or {"delete_count": 2}
        else:
            mock_client.delete.return_value = {"delete_count": 2}
        if "inserted" in locals():
            mock_client.insert.side_effect = lambda collection_name, data, **kwargs: inserted.extend(data)
        monkeypatch.setattr(milvus_store, "MilvusClient", lambda *args, **kwargs: mock_client)
        
        module.store_milvus.python_func(
"""

content = content.replace('        module.store_milvus.python_func(', mock_client_code)

with open("tests/test_docs_pipeline.py", "w") as f:
    f.write(content)
