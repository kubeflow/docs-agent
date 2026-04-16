import yaml
import ast
import json
from pathlib import Path

# ── TEST 1: YAML parser ───────────────────────────────────
print("=" * 50)
print("TEST 1 - YAML Resource Parser")
print("=" * 50)

sample_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kubeflow-pipelines
  namespace: kubeflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kubeflow-pipelines
---
apiVersion: v1
kind: Service
metadata:
  name: kubeflow-pipelines-service
  namespace: kubeflow
spec:
  ports:
    - port: 8888
"""

units = []
docs = list(yaml.safe_load_all(sample_yaml))

for doc in docs:
    if not doc or "kind" not in doc:
        continue
    units.append({
        "text":         yaml.dump(doc, default_flow_style=False),
        "kind":         doc.get("kind", ""),
        "name":         doc.get("metadata", {}).get("name", ""),
        "content_type": "kubernetes_resource",
        "source_url":   "https://github.com/kubeflow/manifests/blob/master/example.yaml"
    })

print(f"Parsed {len(units)} Kubernetes resources")
for u in units:
    print(f"  kind={u['kind']}  name={u['name']}")

assert len(units) == 2, "Expected 2 resources"
assert units[0]["kind"] == "Deployment"
assert units[1]["kind"] == "Service"
print("YAML TEST PASSED")


# ── TEST 2: Python AST parser ─────────────────────────────
print()
print("=" * 50)
print("TEST 2 - Python AST Parser")
print("=" * 50)

sample_python = '''
def create_pipeline(name: str, description: str = ""):
    """Creates a new KFP pipeline with the given name."""
    pipeline = Pipeline(name=name)
    pipeline.description = description
    return pipeline


class PipelineRunner:
    """Handles execution of Kubeflow Pipelines."""

    def run(self, pipeline_id: str):
        """Run a pipeline by ID."""
        return self.client.run_pipeline(pipeline_id)

    def _validate(self, pipeline_id: str):
        if not pipeline_id:
            raise ValueError("pipeline_id cannot be empty")
'''

tree  = ast.parse(sample_python)
units = []

for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        continue
    snippet   = ast.get_source_segment(sample_python, node) or ""
    if len(snippet) < 20:
        continue
    docstring = ast.get_docstring(node) or ""
    units.append({
        "text":         snippet,
        "kind":         type(node).__name__,
        "name":         node.name,
        "docstring":    docstring,
        "content_type": "python_definition",
        "source_url":   f"https://github.com/kubeflow/manifests/blob/master/example.py#L{node.lineno}"
    })

print(f"Parsed {len(units)} Python definitions")
for u in units:
    print(f"  kind={u['kind']}  name={u['name']}")
    print(f"  docstring={u['docstring'][:50]}")

assert len(units) >= 3, "Expected at least 3 definitions"
print("PYTHON AST TEST PASSED")


# ── SUMMARY ───────────────────────────────────────────────
print()
print("=" * 50)
print("ALL TESTS PASSED - Code ingestion pipeline ready")
print("=" * 50)