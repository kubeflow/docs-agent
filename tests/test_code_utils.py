"""Tests for code and manifest chunking utilities."""

import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PIPELINES_DIR = Path(__file__).parent.parent / "docs-agent-mcp" / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

from code_utils import chunk_code_file, parse_json_file, parse_python_ast, parse_yaml_documents


def load_code_pipeline_module():
    """Load the hyphenated pipeline module so component python funcs are testable."""
    pytest.importorskip("kfp", reason="pipeline component tests require the KFP SDK")
    pipeline_path = PIPELINES_DIR / "code-pipeline.py"
    spec = importlib.util.spec_from_file_location("code_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload=None, text="", status_code=200, headers=None):
        self.payload = payload
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


def test_code_pipeline_preserves_github_canonical_citation_url(monkeypatch, tmp_path):
    """Katib's master branch URL must survive download and chunking unchanged."""
    module = load_code_pipeline_module()
    source_url = "https://github.com/kubeflow/katib/blob/master/examples/v1beta1/hp-tuning/random.yaml"
    contents_url = "https://api.github.com/repos/kubeflow/katib/contents/examples/v1beta1/hp-tuning"
    file_api_url = "https://api.github.com/repos/kubeflow/katib/contents/random.yaml"
    raw_url = "https://raw.githubusercontent.com/kubeflow/katib/master/examples/v1beta1/hp-tuning/random.yaml"
    yaml_text = "apiVersion: kubeflow.org/v1beta1\nkind: Experiment\nmetadata:\n  name: random\n"

    def fake_get(url, params=None, headers=None):
        if url == contents_url:
            return FakeResponse(
                [
                    {
                        "type": "file",
                        "name": "random.yaml",
                        "path": "examples/v1beta1/hp-tuning/random.yaml",
                        "url": file_api_url,
                        "download_url": raw_url,
                        "html_url": source_url,
                    }
                ]
            )
        if url == file_api_url:
            return FakeResponse(
                {
                    "content": base64.b64encode(yaml_text.encode()).decode(),
                    "html_url": source_url,
                }
            )
        if url == raw_url:
            return FakeResponse(text=yaml_text)
        raise AssertionError(f"Unexpected GitHub URL: {url}")

    monkeypatch.setattr("requests.get", fake_get)
    downloaded_path = tmp_path / "downloaded.jsonl"
    module.download_github_code.python_func(
        repos="kubeflow/katib",
        directory_paths="examples/v1beta1/hp-tuning",
        file_extensions="yaml,yml",
        github_token="test-token",
        code_data=SimpleNamespace(path=str(downloaded_path)),
    )

    downloaded = json.loads(downloaded_path.read_text())
    assert downloaded["citation_url"] == source_url

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse([[0.1, 0.2]]))
    embedded_path = tmp_path / "embedded.jsonl"
    module.chunk_and_embed_code.python_func(
        code_data=SimpleNamespace(path=str(downloaded_path)),
        chunk_size=1000,
        chunk_overlap=100,
        embeddings_service_url="http://embeddings.test/embed",
        embedding_batch_size=8,
        embedded_data=SimpleNamespace(path=str(embedded_path)),
    )

    embedded = json.loads(embedded_path.read_text())
    assert embedded["citation_url"] == source_url


def test_compiled_code_pipeline_injects_required_secrets(tmp_path):
    """Compilation must fail closed rather than silently omit Milvus auth."""
    module = load_code_pipeline_module()
    compiled_path = tmp_path / "code_rag_pipeline.yaml"

    module.kfp.compiler.Compiler().compile(
        pipeline_func=module.code_rag_pipeline,
        package_path=str(compiled_path),
    )

    compiled = compiled_path.read_text()
    assert "milvus-auth" in compiled
    assert "MILVUS_PASSWORD" in compiled
    assert "github-pat" in compiled


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


def test_code_pipeline_downloads_files_over_the_contents_api_size_cap(monkeypatch, tmp_path):
    """Above 1 MiB the Contents API returns no content, so the raw URL must be used."""
    module = load_code_pipeline_module()
    contents_url = "https://api.github.com/repos/kubeflow/manifests/contents/applications/kserve"
    file_api_url = "https://api.github.com/repos/kubeflow/manifests/contents/kserve.yaml"
    raw_url = "https://raw.githubusercontent.com/kubeflow/manifests/master/applications/kserve/kserve.yaml"
    html_url = "https://github.com/kubeflow/manifests/blob/master/applications/kserve/kserve.yaml"
    yaml_text = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: kserve\n" + ("# padding\n" * 2000)

    def fake_get(url, params=None, headers=None):
        if url == contents_url:
            return FakeResponse(
                [
                    {
                        "type": "file",
                        "name": "kserve.yaml",
                        "path": "applications/kserve/kserve.yaml",
                        "size": 7051337,
                        "url": file_api_url,
                        "download_url": raw_url,
                        "html_url": html_url,
                    }
                ]
            )
        if url == file_api_url:
            # What GitHub actually answers for a blob over the cap.
            return FakeResponse({"encoding": "none", "content": "", "html_url": html_url})
        if url == raw_url:
            return FakeResponse(text=yaml_text)
        raise AssertionError(f"Unexpected GitHub URL: {url}")

    monkeypatch.setattr("requests.get", fake_get)
    downloaded_path = tmp_path / "downloaded.jsonl"
    module.download_github_code.python_func(
        repos="kubeflow/manifests",
        directory_paths="applications/kserve",
        file_extensions="yaml,yml",
        github_token="test-token",
        code_data=SimpleNamespace(path=str(downloaded_path)),
    )

    downloaded = json.loads(downloaded_path.read_text())
    assert downloaded["content"] == yaml_text
    assert downloaded["citation_url"] == html_url


def test_code_pipeline_fails_when_a_configured_directory_is_missing(monkeypatch, tmp_path):
    """A directory that 404s must end the run, not report zero files and succeed."""
    module = load_code_pipeline_module()

    monkeypatch.setattr("requests.get", lambda url, params=None, headers=None: FakeResponse(status_code=404))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        module.download_github_code.python_func(
            repos="kubeflow/manifests",
            directory_paths="apps/katib",
            file_extensions="yaml,yml",
            github_token="test-token",
            code_data=SimpleNamespace(path=str(tmp_path / "downloaded.jsonl")),
        )


def test_code_pipeline_rejects_a_short_embeddings_response(monkeypatch, tmp_path):
    """A truncated batch would otherwise leave records with no embedding at all."""
    module = load_code_pipeline_module()
    source_path = tmp_path / "downloaded.jsonl"
    source_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "path": f"applications/katib/manifest-{index}.yaml",
                    "content": f"apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cm-{index}\n",
                    "file_name": f"manifest-{index}.yaml",
                    "repo": "kubeflow/manifests",
                    "citation_url": f"https://github.com/kubeflow/manifests/blob/master/m-{index}.yaml",
                }
            )
            for index in range(3)
        )
        + "\n"
    )

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse([[0.1, 0.2]]))

    with pytest.raises(RuntimeError, match="vectors for batch of 3"):
        module.chunk_and_embed_code.python_func(
            code_data=SimpleNamespace(path=str(source_path)),
            chunk_size=1000,
            chunk_overlap=100,
            embeddings_service_url="http://embeddings.test/embed",
            embedding_batch_size=8,
            embedded_data=SimpleNamespace(path=str(tmp_path / "embedded.jsonl")),
        )


def test_code_pipeline_waits_out_a_secondary_rate_limit(monkeypatch, tmp_path):
    """A secondary rate limit answers 429 with Retry-After and must be waited out."""
    module = load_code_pipeline_module()
    contents_url = "https://api.github.com/repos/kubeflow/manifests/contents/common/istio"
    raw_url = "https://raw.githubusercontent.com/kubeflow/manifests/master/common/istio/base.yaml"
    listing = [
        {
            "type": "file",
            "name": "base.yaml",
            "path": "common/istio/base.yaml",
            "download_url": raw_url,
            "html_url": "https://github.com/kubeflow/manifests/blob/master/common/istio/base.yaml",
        }
    ]
    attempts = []
    waited = []

    def fake_get(url, params=None, headers=None):
        attempts.append(url)
        if url == contents_url and attempts.count(contents_url) == 1:
            return FakeResponse(status_code=429, headers={"Retry-After": "7"})
        if url == contents_url:
            return FakeResponse(listing)
        return FakeResponse(text="apiVersion: v1\nkind: Namespace\nmetadata:\n  name: istio-system\n")

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: waited.append(seconds))

    downloaded_path = tmp_path / "downloaded.jsonl"
    module.download_github_code.python_func(
        repos="kubeflow/manifests",
        directory_paths="common/istio",
        file_extensions="yaml,yml",
        github_token="test-token",
        code_data=SimpleNamespace(path=str(downloaded_path)),
    )

    assert waited == [7]
    assert json.loads(downloaded_path.read_text())["file_name"] == "base.yaml"
