"""Tests for code and manifest chunking utilities."""

import sys
from pathlib import Path


PIPELINES_DIR = Path(__file__).parent.parent / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

from code_utils import chunk_code_file, parse_json_file, parse_python_ast, parse_yaml_documents


class TestParseYamlDocuments:
    """Tests for Kubernetes YAML-aware chunking."""

    def test_extracts_metadata_from_multi_document_yaml(self):
        content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-pipeline
  namespace: kubeflow
---
apiVersion: v1
kind: Service
metadata:
  name: ml-pipeline
"""

        chunks = parse_yaml_documents(content, "apps/pipeline/deployment.yaml")

        assert len(chunks) == 2
        assert chunks[0]["resource_kind"] == "Deployment"
        assert chunks[0]["resource_name"] == "ml-pipeline"
        assert chunks[0]["resource_namespace"] == "kubeflow"
        assert chunks[0]["file_type"] == "yaml"
        assert chunks[1]["resource_kind"] == "Service"

    def test_marks_kustomization_files(self):
        content = """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
"""

        chunks = parse_yaml_documents(content, "apps/pipeline/kustomization.yaml")

        assert len(chunks) == 1
        assert chunks[0]["file_type"] == "kustomize"
        assert chunks[0]["resource_kind"] == "Kustomization"

    def test_invalid_yaml_falls_back_to_text_chunk(self):
        content = """apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Values.name }}
data:
  value: [unterminated
"""

        chunks = parse_yaml_documents(content, "templates/configmap.yaml")

        assert len(chunks) == 1
        assert chunks[0]["content"] == content.strip()
        assert chunks[0]["resource_kind"] == ""
        assert chunks[0]["file_type"] == "yaml"


class TestParsePythonAst:
    """Tests for Python AST-aware chunking."""

    def test_extracts_header_classes_functions_and_async_functions(self):
        content = '''"""Module docs."""
import os

CONSTANT = "value"

@decorator
def build_pipeline(name):
    return name

class PipelineCompiler:
    def compile(self):
        return True

async def run_pipeline():
    return "done"
'''

        chunks = parse_python_ast(content, "sdk/compiler.py")

        kinds_and_names = [(chunk["resource_kind"], chunk["resource_name"]) for chunk in chunks]
        assert kinds_and_names == [
            ("module_header", "compiler.py"),
            ("function", "build_pipeline"),
            ("class", "PipelineCompiler"),
            ("async_function", "run_pipeline"),
        ]
        assert chunks[1]["content"].startswith("@decorator")
        assert chunks[1]["file_type"] == "python"

    def test_returns_module_chunk_when_no_top_level_defs(self):
        content = "PIPELINE_ROOT = '/tmp/pipeline'\nDEFAULT_TIMEOUT = 30\n"

        chunks = parse_python_ast(content, "settings.py")

        assert len(chunks) == 1
        assert chunks[0]["resource_kind"] == "module"
        assert chunks[0]["resource_name"] == "settings.py"

    def test_syntax_error_returns_whole_file(self):
        content = "def broken(:\n    pass\n"

        chunks = parse_python_ast(content, "broken.py")

        assert len(chunks) == 1
        assert chunks[0]["content"] == content
        assert chunks[0]["resource_kind"] == ""
        assert chunks[0]["file_type"] == "python"


class TestChunkCodeFile:
    """Tests for file type routing and oversized chunk behavior."""

    def test_json_file_is_indexed_as_single_chunk(self):
        content = '{"name": "docs-agent", "private": true}'

        chunks = parse_json_file(content, "package.json")

        assert chunks == [
            {
                "content": content,
                "resource_kind": "",
                "resource_name": "package.json",
                "resource_namespace": "",
                "file_type": "json",
            }
        ]

    def test_generic_file_uses_text_fallback(self):
        content = "FROM python:3.11-slim\nRUN echo hello\n"

        chunks = chunk_code_file(content, "Dockerfile")

        assert len(chunks) == 1
        assert chunks[0]["resource_name"] == "Dockerfile"
        assert chunks[0]["file_type"] == "text"

    def test_oversized_yaml_subchunks_preserve_metadata(self):
        content = (
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "metadata:\n"
            "  name: large-config\n"
            "  namespace: kubeflow\n"
            "data:\n"
            f"  body: {'value ' * 120}\n"
        )

        chunks = chunk_code_file(content, "manifests/configmap.yaml", chunk_size=120, chunk_overlap=10)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["resource_kind"] == "ConfigMap"
            assert chunk["resource_name"] == "large-config"
            assert chunk["resource_namespace"] == "kubeflow"
            assert chunk["file_type"] == "yaml"
