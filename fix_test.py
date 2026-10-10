import re

with open("tests/test_docs_pipeline.py", "r") as f:
    content = f.read()

# Fix mock_client.delete
content = re.sub(
    r'mock_client\.delete\.side_effect = lambda collection_name, filter, \*\*kwargs: deleted\.append\(filter\)',
    'mock_client.delete.side_effect = lambda collection_name, filter, **kwargs: deleted.append(filter) or {"delete_count": 2}',
    content
)

# For test_store_accepts_compatible_legacy_schema_without_last_updated, there is no `deleted`.
# We need to make sure the delete side effect in the first test doesn't crash.
content = content.replace(
    'mock_client.delete.side_effect = lambda collection_name, filter, **kwargs: deleted.append(filter) or {"delete_count": 2}',
    'mock_client.delete.side_effect = lambda collection_name, filter, **kwargs: {"delete_count": 2}',
    1
)

with open("tests/test_docs_pipeline.py", "w") as f:
    f.write(content)
