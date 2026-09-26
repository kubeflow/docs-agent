"""
Utilities for YAML-aware and Python AST-aware code chunking.

Splits Kubernetes YAML manifests at document boundaries (---) and extracts
resource metadata (apiVersion, kind, name, namespace). For Python files,
uses the ast module to split at function/class boundaries.

Falls back to RecursiveCharacterTextSplitter when parsing fails.
"""

import ast
import re
import yaml


def parse_yaml_documents(content: str, file_path: str = "") -> list[dict]:
    """Parse a YAML file into per-document chunks with extracted K8s metadata.

    Each YAML document separated by '---' becomes one chunk. Metadata fields
    (apiVersion, kind, metadata.name, metadata.namespace) are extracted for
    searchable storage in Milvus.

    For kustomization.yaml files, the entire file is kept as a single chunk
    since they are typically small.

    Args:
        content: Raw YAML file content.
        file_path: Original file path (used for kustomization detection).

    Returns:
        List of dicts with keys: content, resource_kind, resource_name,
        resource_namespace, file_type.
    """
    chunks = []
    file_name = file_path.rsplit("/", 1)[-1] if file_path else ""

    # Determine file_type
    if file_name == "kustomization.yaml" or file_name == "kustomization.yml":
        file_type = "kustomize"
    elif file_name.endswith((".yaml", ".yml")):
        file_type = "yaml"
    else:
        file_type = "yaml"

    # Split on YAML document boundaries
    raw_docs = re.split(r'^---\s*$', content, flags=re.MULTILINE)

    for raw_doc in raw_docs:
        raw_doc = raw_doc.strip()
        if not raw_doc or len(raw_doc) < 10:
            continue

        resource_kind = ""
        resource_name = ""
        resource_namespace = ""

        try:
            parsed = yaml.safe_load(raw_doc)
            if isinstance(parsed, dict):
                resource_kind = str(parsed.get("kind", ""))
                metadata = parsed.get("metadata", {})
                if isinstance(metadata, dict):
                    resource_name = str(metadata.get("name", ""))
                    resource_namespace = str(metadata.get("namespace", ""))
                # Also capture apiVersion in the content header for context
        except yaml.YAMLError:
            # Likely a Helm template or invalid YAML — still index as text
            pass

        chunks.append({
            "content": raw_doc,
            "resource_kind": resource_kind,
            "resource_name": resource_name,
            "resource_namespace": resource_namespace,
            "file_type": file_type,
        })

    return chunks


def parse_python_ast(content: str, file_path: str = "") -> list[dict]:
    """Parse a Python file into per-function/class chunks using the ast module.

    Each top-level function or class becomes one chunk. Module-level code
    (imports, constants) is grouped into a single "module_header" chunk.

    Args:
        content: Raw Python file content.
        file_path: Original file path for metadata.

    Returns:
        List of dicts with keys: content, resource_kind, resource_name,
        resource_namespace, file_type.
    """
    chunks = []
    lines = content.split("\n")

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Can't parse — return whole file as one chunk
        return [{
            "content": content,
            "resource_kind": "",
            "resource_name": "",
            "resource_namespace": "",
            "file_type": "python",
        }]

    # Collect top-level function/class nodes with their line ranges
    nodes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(node)

    if not nodes:
        # No functions/classes — return whole file as one chunk
        return [{
            "content": content,
            "resource_kind": "module",
            "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
            "resource_namespace": "",
            "file_type": "python",
        }]

    # Extract module header (everything before first function/class)
    first_node_line = nodes[0].lineno  # 1-indexed
    if first_node_line > 1:
        header = "\n".join(lines[:first_node_line - 1]).strip()
        if header and len(header) >= 10:
            chunks.append({
                "content": header,
                "resource_kind": "module_header",
                "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
                "resource_namespace": "",
                "file_type": "python",
            })

    # Extract each function/class
    for i, node in enumerate(nodes):
        start_line = node.lineno - 1  # Convert to 0-indexed

        # Include decorators if present
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno - 1

        # End line: start of next node or end of file
        if i + 1 < len(nodes):
            next_start = nodes[i + 1].lineno - 1
            if nodes[i + 1].decorator_list:
                next_start = nodes[i + 1].decorator_list[0].lineno - 1
            end_line = next_start
        else:
            end_line = len(lines)

        chunk_content = "\n".join(lines[start_line:end_line]).rstrip()
        if not chunk_content or len(chunk_content) < 10:
            continue

        if isinstance(node, ast.ClassDef):
            kind = "class"
        elif isinstance(node, ast.AsyncFunctionDef):
            kind = "async_function"
        else:
            kind = "function"

        chunks.append({
            "content": chunk_content,
            "resource_kind": kind,
            "resource_name": node.name,
            "resource_namespace": "",
            "file_type": "python",
        })

    return chunks


def parse_json_file(content: str, file_path: str = "") -> list[dict]:
    """Return a JSON file as a single chunk.

    JSON config files (e.g., package.json, tsconfig.json) are typically small
    and should be indexed whole.

    Args:
        content: Raw JSON file content.
        file_path: Original file path for metadata.

    Returns:
        Single-element list with the file as one chunk.
    """
    return [{
        "content": content,
        "resource_kind": "",
        "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
        "resource_namespace": "",
        "file_type": "json",
    }]


def chunk_code_file(
    content: str,
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Route a code file to the appropriate parser based on extension.

    Supports .yaml/.yml (YAML-aware), .py (AST-aware), .json, and
    generic text fallback for other extensions.

    If the parser produces chunks larger than chunk_size, they are
    sub-split using RecursiveCharacterTextSplitter.

    Args:
        content: Raw file content.
        file_path: File path (used for extension detection and metadata).
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between sub-split chunks.

    Returns:
        List of dicts with keys: content, resource_kind, resource_name,
        resource_namespace, file_type.
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

    if ext in ("yaml", "yml"):
        raw_chunks = parse_yaml_documents(content, file_path)
    elif ext == "py":
        raw_chunks = parse_python_ast(content, file_path)
    elif ext == "json":
        raw_chunks = parse_json_file(content, file_path)
    else:
        # Generic text fallback (Dockerfile, Makefile, .sh, .go, etc.)
        raw_chunks = [{
            "content": content,
            "resource_kind": "",
            "resource_name": file_path.rsplit("/", 1)[-1] if file_path else "",
            "resource_namespace": "",
            "file_type": ext if ext else "text",
        }]

    # Sub-split oversized chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    result = []
    for chunk in raw_chunks:
        text = chunk["content"]
        if len(text) <= chunk_size:
            result.append(chunk)
        else:
            sub_texts = splitter.split_text(text)
            for sub_text in sub_texts:
                result.append({
                    "content": sub_text,
                    "resource_kind": chunk["resource_kind"],
                    "resource_name": chunk["resource_name"],
                    "resource_namespace": chunk["resource_namespace"],
                    "file_type": chunk["file_type"],
                })

    return result
