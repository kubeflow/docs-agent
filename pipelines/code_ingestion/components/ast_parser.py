"""
Code Ingestion — AST Parser Component

Multi-language parser that extracts logical code units:
  - Python: AST-based extraction of functions and classes with docstrings
  - Go: Regex-based splitting on func/struct boundaries
  - YAML/YML: Split by top-level Kubernetes resource kind
  - Markdown: Split by H2/H3 headings

Each extracted unit becomes a chunk with rich metadata for retrieval.
"""

import ast
import hashlib
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)

PATH_ALIAS_HINTS = {
    "common/istio": [
        "istio",
        "service mesh",
        "gateway",
        "authorization policy",
        "peer authentication",
        "virtual service",
        "sidecar",
        "envoy",
        "mtls",
        "ingress",
    ],
    "common/knative": [
        "knative",
        "serving",
        "eventing",
        "serverless",
        "scale to zero",
        "activator",
        "revision",
        "service",
        "net istio",
        "webhook",
    ],
    "common/dex": [
        "dex",
        "oidc",
        "oauth2",
        "authentication",
        "identity provider",
        "connector",
        "login",
    ],
    "common/cert-manager": [
        "cert manager",
        "certificate",
        "issuer",
        "clusterissuer",
        "cainjector",
        "tls",
        "webhook",
    ],
    "applications/pipeline": [
        "kubeflow pipelines",
        "kfp",
        "pipeline api server",
        "deployment",
        "service",
        "configmap",
        "role",
        "rolebinding",
        "serviceaccount",
        "crd",
        "webhook",
        "scheduled workflow",
    ],
    "applications/profiles": [
        "profiles",
        "namespaces",
        "rbac",
        "rolebinding",
        "serviceaccount",
        "user profile",
    ],
    "tests": [
        "tests",
        "e2e",
        "integration",
        "validation",
        "presubmit",
    ],
}


def split_search_terms(value: str) -> List[str]:
    """Split identifiers and paths into normalized search terms."""
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", expanded)
    return [token.lower() for token in normalized.split() if token]


def unique_terms(values: Iterable[str], limit: int = 24) -> List[str]:
    """Return unique normalized search terms while preserving order."""
    seen = set()
    ordered: List[str] = []
    for value in values:
        for token in split_search_terms(value):
            if token not in seen:
                seen.add(token)
                ordered.append(token)
                if len(ordered) >= limit:
                    return ordered
    return ordered


def get_path_aliases(file_path: str) -> List[str]:
    """Return path-aware semantic aliases for common Kubeflow manifest areas."""
    normalized = file_path.replace("\\", "/").lower()
    aliases: List[str] = []
    for prefix, hints in PATH_ALIAS_HINTS.items():
        if normalized.startswith(prefix):
            aliases.extend(hints)
    return aliases


def summarize_list(values: Any, limit: int = 8) -> str:
    """Summarize a list-like value for retrieval context lines."""
    if not isinstance(values, list):
        return ""
    flattened = [str(item) for item in values if item]
    return ", ".join(flattened[:limit])


def extract_container_names(parsed: Dict[str, Any]) -> List[str]:
    """Extract workload container names when present."""
    spec = parsed.get("spec")
    if not isinstance(spec, dict):
        return []

    template = spec.get("template", {})
    if isinstance(template, dict):
        template_spec = template.get("spec", {})
        if isinstance(template_spec, dict):
            containers = template_spec.get("containers", [])
            if isinstance(containers, list):
                return [
                    str(container.get("name"))
                    for container in containers
                    if isinstance(container, dict) and container.get("name")
                ]

    job_template = spec.get("jobTemplate", {})
    if isinstance(job_template, dict):
        nested_spec = job_template.get("spec", {})
        if isinstance(nested_spec, dict):
            nested_template = nested_spec.get("template", {})
            if isinstance(nested_template, dict):
                nested_template_spec = nested_template.get("spec", {})
                if isinstance(nested_template_spec, dict):
                    containers = nested_template_spec.get("containers", [])
                    if isinstance(containers, list):
                        return [
                            str(container.get("name"))
                            for container in containers
                            if isinstance(container, dict) and container.get("name")
                        ]
    return []


def build_manifest_context(
    parsed: Dict[str, Any],
    file_path: str,
    folder_context: str,
) -> str:
    """Build retrieval-oriented context text for a Kubernetes manifest."""
    metadata = parsed.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}

    kind = str(parsed.get("kind", "Unknown"))
    api_version = str(parsed.get("apiVersion", "unknown"))
    name = str(metadata.get("name", "unknown"))
    namespace = str(metadata.get("namespace", "cluster-scoped"))

    path_terms = unique_terms([file_path, os.path.basename(file_path), folder_context], limit=18)
    alias_terms = unique_terms(get_path_aliases(file_path), limit=18)
    label_keys = summarize_list(list((metadata.get("labels") or {}).keys()))
    annotation_keys = summarize_list(list((metadata.get("annotations") or {}).keys()))
    top_level_keys = summarize_list(list(parsed.keys()))

    summary_lines = [
        f"Manifest file path: {file_path}",
        f"Folder context: {folder_context}",
        f"Resource kind: {kind}",
        f"API version: {api_version}",
        f"Metadata name: {name}",
        f"Namespace: {namespace}",
    ]

    if path_terms:
        summary_lines.append(f"Path hints: {' '.join(path_terms)}")
    if alias_terms:
        summary_lines.append(f"Domain hints: {' '.join(alias_terms)}")
    if top_level_keys:
        summary_lines.append(f"Top-level keys: {top_level_keys}")
    if label_keys:
        summary_lines.append(f"Label keys: {label_keys}")
    if annotation_keys:
        summary_lines.append(f"Annotation keys: {annotation_keys}")

    spec = parsed.get("spec")
    spec = spec if isinstance(spec, dict) else {}

    if kind.lower() == "kustomization" or os.path.basename(file_path).lower() == "kustomization.yaml":
        resources = summarize_list(parsed.get("resources"))
        components = summarize_list(parsed.get("components"))
        bases = summarize_list(parsed.get("bases"))
        patches = summarize_list(parsed.get("patchesStrategicMerge"))
        if resources:
            summary_lines.append(f"Kustomize resources: {resources}")
        if components:
            summary_lines.append(f"Kustomize components: {components}")
        if bases:
            summary_lines.append(f"Kustomize bases: {bases}")
        if patches:
            summary_lines.append(f"Kustomize patches: {patches}")

    if kind in {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"}:
        container_names = summarize_list(extract_container_names(parsed))
        service_account = spec.get("serviceAccountName") or (
            spec.get("template", {}).get("spec", {}).get("serviceAccountName")
            if isinstance(spec.get("template"), dict)
            else None
        )
        if container_names:
            summary_lines.append(f"Workload containers: {container_names}")
        if service_account:
            summary_lines.append(f"Service account: {service_account}")

    if kind == "Service":
        service_type = spec.get("type")
        ports = spec.get("ports")
        selector = spec.get("selector")
        if service_type:
            summary_lines.append(f"Service type: {service_type}")
        if isinstance(selector, dict) and selector:
            summary_lines.append(
                f"Service selector keys: {', '.join(list(selector.keys())[:8])}"
            )
        if isinstance(ports, list) and ports:
            port_values = [str(port.get("port")) for port in ports if isinstance(port, dict) and port.get("port")]
            if port_values:
                summary_lines.append(f"Service ports: {', '.join(port_values[:8])}")

    if kind == "CustomResourceDefinition":
        crd_spec = spec
        names = crd_spec.get("names", {}) if isinstance(crd_spec.get("names"), dict) else {}
        versions = crd_spec.get("versions", [])
        if crd_spec.get("group"):
            summary_lines.append(f"CRD group: {crd_spec.get('group')}")
        if names.get("kind"):
            summary_lines.append(f"CRD served kind: {names.get('kind')}")
        if isinstance(versions, list) and versions:
            version_names = [str(version.get("name")) for version in versions if isinstance(version, dict) and version.get("name")]
            if version_names:
                summary_lines.append(f"CRD versions: {', '.join(version_names[:8])}")

    if kind in {"Role", "ClusterRole"}:
        rules = spec.get("rules", parsed.get("rules"))
        if isinstance(rules, list) and rules:
            resource_names = []
            verbs = []
            for rule in rules[:4]:
                if isinstance(rule, dict):
                    resource_names.extend(str(item) for item in rule.get("resources", [])[:4])
                    verbs.extend(str(item) for item in rule.get("verbs", [])[:4])
            if resource_names:
                summary_lines.append(f"RBAC resources: {', '.join(resource_names[:10])}")
            if verbs:
                summary_lines.append(f"RBAC verbs: {', '.join(verbs[:10])}")

    if kind in {"RoleBinding", "ClusterRoleBinding"}:
        role_ref = parsed.get("roleRef", {})
        subjects = parsed.get("subjects", [])
        if isinstance(role_ref, dict) and role_ref.get("name"):
            summary_lines.append(f"Binding roleRef: {role_ref.get('name')}")
        if isinstance(subjects, list) and subjects:
            subject_names = [
                str(subject.get("name"))
                for subject in subjects
                if isinstance(subject, dict) and subject.get("name")
            ]
            if subject_names:
                summary_lines.append(f"Binding subjects: {', '.join(subject_names[:10])}")

    if kind in {"AuthorizationPolicy", "PeerAuthentication", "VirtualService", "Gateway", "DestinationRule"}:
        selector = spec.get("selector", {})
        if isinstance(selector, dict):
            match_labels = selector.get("matchLabels", {})
            if isinstance(match_labels, dict) and match_labels:
                summary_lines.append(
                    f"Istio selector labels: {', '.join(list(match_labels.keys())[:8])}"
                )
        gateways = spec.get("gateways")
        hosts = spec.get("hosts")
        if isinstance(gateways, list) and gateways:
            summary_lines.append(f"Istio gateways: {', '.join(str(g) for g in gateways[:8])}")
        if isinstance(hosts, list) and hosts:
            summary_lines.append(f"Istio hosts: {', '.join(str(h) for h in hosts[:8])}")

    return "\n".join(f"# {line}" for line in summary_lines if line)


def generate_chunk_id(file_path: str, symbol_name: str, index: int) -> str:
    """Generate a deterministic chunk ID.

    Args:
        file_path: Relative file path.
        symbol_name: Function/class/resource name.
        index: Sequential index.

    Returns:
        SHA256 hash string (first 32 chars).
    """
    raw = f"{file_path}::{symbol_name}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ─── Python Parser ──────────────────────────────────────────────────────────

def parse_python(content: str, file_path: str, commit_sha: str,
                 folder_context: str) -> List[Dict[str, Any]]:
    """Parse Python source into function and class chunks via AST.

    Args:
        content: Python source code.
        file_path: Relative file path.
        commit_sha: Git commit SHA.
        folder_context: Top-level folder name.

    Returns:
        List of chunk dicts.
    """
    chunks = []
    lines = content.split("\n")

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logger.warning("Syntax error in %s: %s", file_path, e)
        # Fall back to whole-file chunk
        return [{
            "chunk_id": generate_chunk_id(file_path, "module", 0),
            "file_path": file_path,
            "extension": ".py",
            "language": "python",
            "symbol_name": os.path.basename(file_path),
            "chunk_text": content,
            "start_line": 1,
            "end_line": len(lines),
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        }]

    idx = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol_name = node.name
            start_line = node.lineno
            end_line = node.end_lineno or start_line

            # Extract the source lines
            chunk_lines = lines[start_line - 1 : end_line]
            chunk_text = "\n".join(chunk_lines)

            # Extract docstring if present
            docstring = ast.get_docstring(node) or ""
            symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"

            chunks.append({
                "chunk_id": generate_chunk_id(file_path, symbol_name, idx),
                "file_path": file_path,
                "extension": ".py",
                "language": "python",
                "symbol_name": f"{symbol_type}:{symbol_name}",
                "chunk_text": chunk_text,
                "start_line": start_line,
                "end_line": end_line,
                "commit_sha": commit_sha,
                "folder_context": folder_context,
            })
            idx += 1

    # If no functions/classes found, treat whole file as one chunk
    if not chunks:
        chunks.append({
            "chunk_id": generate_chunk_id(file_path, "module", 0),
            "file_path": file_path,
            "extension": ".py",
            "language": "python",
            "symbol_name": f"module:{os.path.basename(file_path)}",
            "chunk_text": content,
            "start_line": 1,
            "end_line": len(lines),
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        })

    return chunks


# ─── Go Parser ──────────────────────────────────────────────────────────────

def parse_go(content: str, file_path: str, commit_sha: str,
             folder_context: str) -> List[Dict[str, Any]]:
    """Parse Go source by splitting on func and type struct boundaries.

    Args:
        content: Go source code.
        file_path: Relative file path.
        commit_sha: Git commit SHA.
        folder_context: Top-level folder name.

    Returns:
        List of chunk dicts.
    """
    chunks = []
    lines = content.split("\n")

    # Match func declarations and type struct declarations
    pattern = re.compile(
        r"^(?:func\s+(?:\([^)]+\)\s+)?(\w+)|type\s+(\w+)\s+struct)\b",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(content))

    if not matches:
        # Whole file as one chunk
        return [{
            "chunk_id": generate_chunk_id(file_path, "file", 0),
            "file_path": file_path,
            "extension": ".go",
            "language": "go",
            "symbol_name": f"file:{os.path.basename(file_path)}",
            "chunk_text": content,
            "start_line": 1,
            "end_line": len(lines),
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        }]

    for i, match in enumerate(matches):
        symbol = match.group(1) or match.group(2)
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        chunk_text = content[start_pos:end_pos].rstrip()
        start_line = content[:start_pos].count("\n") + 1
        end_line = start_line + chunk_text.count("\n")

        is_struct = match.group(2) is not None
        symbol_type = "struct" if is_struct else "func"

        chunks.append({
            "chunk_id": generate_chunk_id(file_path, symbol, i),
            "file_path": file_path,
            "extension": ".go",
            "language": "go",
            "symbol_name": f"{symbol_type}:{symbol}",
            "chunk_text": chunk_text,
            "start_line": start_line,
            "end_line": end_line,
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        })

    return chunks


# ─── YAML Parser ────────────────────────────────────────────────────────────

def parse_yaml(content: str, file_path: str, commit_sha: str,
               folder_context: str) -> List[Dict[str, Any]]:
    """Parse YAML by splitting on Kubernetes resource kind boundaries.

    Args:
        content: YAML content (may contain multiple documents).
        file_path: Relative file path.
        commit_sha: Git commit SHA.
        folder_context: Top-level folder name.

    Returns:
        List of chunk dicts.
    """
    chunks = []

    # Split multi-document YAML
    documents = content.split("\n---")

    for idx, doc in enumerate(documents):
        doc = doc.strip()
        if not doc:
            continue

        # Try to parse as YAML
        try:
            parsed = yaml.safe_load(doc)
        except yaml.YAMLError:
            parsed = None

        if isinstance(parsed, dict):
            kind = parsed.get("kind", "Unknown")
            name = "unknown"
            metadata = parsed.get("metadata", {})
            if isinstance(metadata, dict):
                name = metadata.get("name", "unknown")
            symbol_name = f"{kind}:{name}"
            manifest_context = build_manifest_context(parsed, file_path, folder_context)
            chunk_body = f"{manifest_context}\n\n{doc}" if manifest_context else doc
        else:
            kind = "fragment"
            symbol_name = f"fragment:{idx}"
            chunk_body = doc

        # Calculate line numbers
        preceding = "\n---".join(documents[:idx])
        start_line = preceding.count("\n") + 1 if preceding else 1
        end_line = start_line + doc.count("\n")

        chunks.append({
            "chunk_id": generate_chunk_id(file_path, symbol_name, idx),
            "file_path": file_path,
            "extension": os.path.splitext(file_path)[1].lower(),
            "language": "yaml",
            "symbol_name": symbol_name,
            "chunk_text": chunk_body,
            "start_line": start_line,
            "end_line": end_line,
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        })

    if not chunks:
        chunks.append({
            "chunk_id": generate_chunk_id(file_path, "file", 0),
            "file_path": file_path,
            "extension": os.path.splitext(file_path)[1].lower(),
            "language": "yaml",
            "symbol_name": f"file:{os.path.basename(file_path)}",
            "chunk_text": content,
            "start_line": 1,
            "end_line": content.count("\n") + 1,
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        })

    return chunks


# ─── Markdown Parser ────────────────────────────────────────────────────────

def parse_markdown(content: str, file_path: str, commit_sha: str,
                   folder_context: str) -> List[Dict[str, Any]]:
    """Parse Markdown by H2/H3 headings.

    Args:
        content: Markdown content.
        file_path: Relative file path.
        commit_sha: Git commit SHA.
        folder_context: Top-level folder name.

    Returns:
        List of chunk dicts.
    """
    chunks = []
    heading_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    if not matches:
        return [{
            "chunk_id": generate_chunk_id(file_path, "doc", 0),
            "file_path": file_path,
            "extension": ".md",
            "language": "markdown",
            "symbol_name": f"doc:{os.path.basename(file_path)}",
            "chunk_text": content,
            "start_line": 1,
            "end_line": content.count("\n") + 1,
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        }]

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()

        start_line = content[:start].count("\n") + 1
        end_line = start_line + text.count("\n")

        chunks.append({
            "chunk_id": generate_chunk_id(file_path, heading, i),
            "file_path": file_path,
            "extension": ".md",
            "language": "markdown",
            "symbol_name": f"heading:{heading[:100]}",
            "chunk_text": text,
            "start_line": start_line,
            "end_line": end_line,
            "commit_sha": commit_sha,
            "folder_context": folder_context,
        })

    return chunks


# ─── Main Dispatcher ────────────────────────────────────────────────────────

PARSERS = {
    ".py": parse_python,
    ".go": parse_go,
    ".yaml": parse_yaml,
    ".yml": parse_yaml,
    ".md": parse_markdown,
}


def parse_file(
    content: str,
    file_path: str,
    extension: str,
    commit_sha: str,
    folder_context: str,
) -> List[Dict[str, Any]]:
    """Parse a file into chunks using the appropriate language parser.

    Args:
        content: File content string.
        file_path: Relative file path.
        extension: File extension (e.g., '.py').
        commit_sha: Git commit SHA.
        folder_context: Top-level folder name.

    Returns:
        List of chunk dicts.
    """
    parser = PARSERS.get(extension.lower())
    if parser is None:
        logger.warning("No parser for extension: %s (%s)", extension, file_path)
        return []

    try:
        return parser(content, file_path, commit_sha, folder_context)
    except Exception as e:
        logger.error("Parser error for %s: %s", file_path, e)
        return []


def parse_all_files(
    repo_dir: str,
    file_list: List[Dict[str, Any]],
    commit_sha: str,
) -> List[Dict[str, Any]]:
    """Parse all files in the file list.

    Args:
        repo_dir: Repository root directory.
        file_list: List of file info dicts from repo_cloner.
        commit_sha: Git commit SHA.

    Returns:
        List of all chunk dicts across all files.
    """
    all_chunks = []

    for i, file_info in enumerate(file_list):
        file_path = file_info["path"]
        extension = file_info["extension"]
        folder_context = file_info.get("folder_context", "root")

        full_path = os.path.join(repo_dir, file_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            logger.warning("Cannot read %s: %s", file_path, e)
            continue

        chunks = parse_file(content, file_path, extension, commit_sha, folder_context)
        all_chunks.extend(chunks)

        if (i + 1) % 50 == 0:
            logger.info("Parsed %d/%d files (%d chunks so far)", i + 1, len(file_list), len(all_chunks))

    logger.info("AST parsing complete: %d chunks from %d files.", len(all_chunks), len(file_list))
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Test Python parser
    py_code = '''
def hello_world():
    """Say hello."""
    print("Hello, World!")

class MyClass:
    """A test class."""
    def method(self):
        pass
'''
    chunks = parse_python(py_code, "test.py", "abc123", "tests")
    logger.info("=== Python Parser Test ===")
    for c in chunks:
        logger.info("  %s [L%d-%d]", c["symbol_name"], c["start_line"], c["end_line"])

    # Test YAML parser
    yaml_content = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
---
apiVersion: v1
kind: Service
metadata:
  name: my-service
'''
    chunks = parse_yaml(yaml_content, "deploy.yaml", "abc123", "apps")
    logger.info("=== YAML Parser Test ===")
    for c in chunks:
        logger.info("  %s [L%d-%d]", c["symbol_name"], c["start_line"], c["end_line"])
