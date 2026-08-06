"""
Code Ingestion — KFP v2 Pipeline

Orchestrates the complete code ingestion flow:
  repo_cloner -> ast_parser -> chunker -> embedder -> loader

Usage:
  # Compile to YAML
  python pipelines/code_ingestion/pipeline.py

  # Run locally (without KFP)
  python -m pipelines.code_ingestion.pipeline --local
"""

import os
import sys

import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output

# Ensure package imports work both when executed as a script and as a module.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import docs pipeline for composition
try:
    from pipelines.docs_ingestion.pipeline import docs_ingestion_pipeline
except ImportError:
    # This might happen if PYTHONPATH is not set during some CI steps
    docs_ingestion_pipeline = None


# ─── KFP Components ─────────────────────────────────────────────────────────

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["gitpython==3.1.43"],
)
def clone_repo(
    repo_url: str,
    branch: str,
    clone_data: Output[Dataset],
):
    """Clone a git repository and collect file metadata.

    Args:
        repo_url: Repository URL to clone.
        branch: Branch name to clone.
        clone_data: Output dataset artifact.
    """
    import json
    import logging
    import os
    import subprocess
    import tempfile

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("repo_cloner")

    SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", ".mypy_cache"}
    EXTENSIONS = {".py", ".go", ".yaml", ".yml", ".md"}
    MIN_SIZE, MAX_SIZE = 200, 100_000

    clone_dir = tempfile.mkdtemp(prefix="code-ingest-")
    logger.info("Cloning %s -> %s", repo_url, clone_dir)

    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, clone_dir],
        check=True, capture_output=True, text=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=clone_dir, check=True,
    )
    commit_sha = result.stdout.strip()
    logger.info("Commit: %s", commit_sha[:12])

    files = []
    for root, dirs, fnames in os.walk(clone_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in fnames:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, clone_dir)
            _, ext = os.path.splitext(fn)
            if ext.lower() not in EXTENSIONS:
                continue
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if sz < MIN_SIZE or sz > MAX_SIZE:
                continue
            parts = rel.split(os.sep)
            folder = parts[0] if len(parts) > 1 else "root"
            files.append({"path": rel, "extension": ext.lower(),
                          "size_bytes": sz, "folder_context": folder})

    logger.info("Collected %d files", len(files))

    # Save file list + contents
    output = []
    for f in files:
        full = os.path.join(clone_dir, f["path"])
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception:
            continue
        output.append({**f, "content": content, "commit_sha": commit_sha})

    with open(clone_data.path, "w") as fh:
        for item in output:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Cleanup
    import shutil
    shutil.rmtree(clone_dir, ignore_errors=True)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["PyYAML==6.0.1"],
)
def parse_code(
    clone_data: Input[Dataset],
    parsed_data: Output[Dataset],
):
    """Parse files into logical code chunks using language-specific parsers.

    Args:
        clone_data: Input dataset from repo cloner.
        parsed_data: Output dataset of parsed chunks.
    """
    import ast as pyast
    import hashlib
    import json
    import logging
    import os
    import re

    import yaml

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ast_parser")

    PATH_ALIAS_HINTS = {
        "common/istio": [
            "istio", "service mesh", "gateway", "authorization policy",
            "peer authentication", "virtual service", "sidecar", "envoy", "mtls", "ingress",
        ],
        "common/knative": [
            "knative", "serving", "eventing", "serverless", "scale to zero",
            "activator", "revision", "service", "net istio", "webhook",
        ],
        "common/dex": [
            "dex", "oidc", "oauth2", "authentication", "identity provider",
            "connector", "login",
        ],
        "common/cert-manager": [
            "cert manager", "certificate", "issuer", "clusterissuer",
            "cainjector", "tls", "webhook",
        ],
        "applications/pipeline": [
            "kubeflow pipelines", "kfp", "pipeline api server", "deployment",
            "service", "configmap", "role", "rolebinding", "serviceaccount",
            "crd", "webhook", "scheduled workflow",
        ],
        "applications/profiles": [
            "profiles", "namespaces", "rbac", "rolebinding", "serviceaccount", "user profile",
        ],
        "tests": ["tests", "e2e", "integration", "validation", "presubmit"],
    }

    def gen_id(fp, sym, idx):
        return hashlib.sha256(f"{fp}::{sym}::{idx}".encode()).hexdigest()[:32]

    def split_terms(value):
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
        normalized = re.sub(r"[^A-Za-z0-9]+", " ", expanded)
        return [token.lower() for token in normalized.split() if token]

    def unique_terms(values, limit=24):
        seen = set()
        ordered = []
        for value in values:
            for token in split_terms(str(value)):
                if token not in seen:
                    seen.add(token)
                    ordered.append(token)
                    if len(ordered) >= limit:
                        return ordered
        return ordered

    def summarize_list(values, limit=8):
        if not isinstance(values, list):
            return ""
        flattened = [str(item) for item in values if item]
        return ", ".join(flattened[:limit])

    def get_path_aliases(fp):
        normalized = fp.replace("\\", "/").lower()
        aliases = []
        for prefix, hints in PATH_ALIAS_HINTS.items():
            if normalized.startswith(prefix):
                aliases.extend(hints)
        return aliases

    def extract_container_names(parsed):
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

    def build_manifest_context(parsed, fp, ctx):
        metadata = parsed.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        kind = str(parsed.get("kind", "Unknown"))
        api_version = str(parsed.get("apiVersion", "unknown"))
        name = str(metadata.get("name", "unknown"))
        namespace = str(metadata.get("namespace", "cluster-scoped"))
        path_terms = unique_terms([fp, os.path.basename(fp), ctx], limit=18)
        alias_terms = unique_terms(get_path_aliases(fp), limit=18)
        top_level_keys = summarize_list(list(parsed.keys()))
        label_keys = summarize_list(list((metadata.get("labels") or {}).keys()))
        annotation_keys = summarize_list(list((metadata.get("annotations") or {}).keys()))

        lines = [
            f"Manifest file path: {fp}",
            f"Folder context: {ctx}",
            f"Resource kind: {kind}",
            f"API version: {api_version}",
            f"Metadata name: {name}",
            f"Namespace: {namespace}",
        ]
        if path_terms:
            lines.append(f"Path hints: {' '.join(path_terms)}")
        if alias_terms:
            lines.append(f"Domain hints: {' '.join(alias_terms)}")
        if top_level_keys:
            lines.append(f"Top-level keys: {top_level_keys}")
        if label_keys:
            lines.append(f"Label keys: {label_keys}")
        if annotation_keys:
            lines.append(f"Annotation keys: {annotation_keys}")

        spec = parsed.get("spec")
        spec = spec if isinstance(spec, dict) else {}

        if kind.lower() == "kustomization" or os.path.basename(fp).lower() == "kustomization.yaml":
            resources = summarize_list(parsed.get("resources"))
            components = summarize_list(parsed.get("components"))
            bases = summarize_list(parsed.get("bases"))
            patches = summarize_list(parsed.get("patchesStrategicMerge"))
            if resources:
                lines.append(f"Kustomize resources: {resources}")
            if components:
                lines.append(f"Kustomize components: {components}")
            if bases:
                lines.append(f"Kustomize bases: {bases}")
            if patches:
                lines.append(f"Kustomize patches: {patches}")

        if kind in {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"}:
            container_names = summarize_list(extract_container_names(parsed))
            service_account = spec.get("serviceAccountName")
            if not service_account and isinstance(spec.get("template"), dict):
                template_spec = spec.get("template", {}).get("spec", {})
                if isinstance(template_spec, dict):
                    service_account = template_spec.get("serviceAccountName")
            if container_names:
                lines.append(f"Workload containers: {container_names}")
            if service_account:
                lines.append(f"Service account: {service_account}")

        if kind == "Service":
            service_type = spec.get("type")
            selector = spec.get("selector")
            ports = spec.get("ports")
            if service_type:
                lines.append(f"Service type: {service_type}")
            if isinstance(selector, dict) and selector:
                lines.append(f"Service selector keys: {', '.join(list(selector.keys())[:8])}")
            if isinstance(ports, list) and ports:
                port_values = [str(port.get('port')) for port in ports if isinstance(port, dict) and port.get('port')]
                if port_values:
                    lines.append(f"Service ports: {', '.join(port_values[:8])}")

        if kind == "CustomResourceDefinition":
            names = spec.get("names", {}) if isinstance(spec.get("names"), dict) else {}
            versions = spec.get("versions", [])
            if spec.get("group"):
                lines.append(f"CRD group: {spec.get('group')}")
            if names.get("kind"):
                lines.append(f"CRD served kind: {names.get('kind')}")
            if isinstance(versions, list) and versions:
                version_names = [str(version.get("name")) for version in versions if isinstance(version, dict) and version.get("name")]
                if version_names:
                    lines.append(f"CRD versions: {', '.join(version_names[:8])}")

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
                    lines.append(f"RBAC resources: {', '.join(resource_names[:10])}")
                if verbs:
                    lines.append(f"RBAC verbs: {', '.join(verbs[:10])}")

        if kind in {"RoleBinding", "ClusterRoleBinding"}:
            role_ref = parsed.get("roleRef", {})
            subjects = parsed.get("subjects", [])
            if isinstance(role_ref, dict) and role_ref.get("name"):
                lines.append(f"Binding roleRef: {role_ref.get('name')}")
            if isinstance(subjects, list) and subjects:
                subject_names = [str(subject.get("name")) for subject in subjects if isinstance(subject, dict) and subject.get("name")]
                if subject_names:
                    lines.append(f"Binding subjects: {', '.join(subject_names[:10])}")

        if kind in {"AuthorizationPolicy", "PeerAuthentication", "VirtualService", "Gateway", "DestinationRule"}:
            selector = spec.get("selector", {})
            if isinstance(selector, dict):
                match_labels = selector.get("matchLabels", {})
                if isinstance(match_labels, dict) and match_labels:
                    lines.append(f"Istio selector labels: {', '.join(list(match_labels.keys())[:8])}")
            gateways = spec.get("gateways")
            hosts = spec.get("hosts")
            if isinstance(gateways, list) and gateways:
                lines.append(f"Istio gateways: {', '.join(str(g) for g in gateways[:8])}")
            if isinstance(hosts, list) and hosts:
                lines.append(f"Istio hosts: {', '.join(str(h) for h in hosts[:8])}")

        return "\n".join(f"# {line}" for line in lines if line)

    def parse_python(content, fp, sha, ctx):
        chunks, lines = [], content.split("\n")
        try:
            tree = pyast.parse(content)
        except SyntaxError:
            return [{"chunk_id": gen_id(fp, "module", 0), "file_path": fp,
                     "extension": ".py", "language": "python",
                     "symbol_name": os.path.basename(fp), "chunk_text": content,
                     "start_line": 1, "end_line": len(lines),
                     "commit_sha": sha, "folder_context": ctx}]
        idx = 0
        for node in pyast.walk(tree):
            if isinstance(node, (pyast.FunctionDef, pyast.AsyncFunctionDef, pyast.ClassDef)):
                sl, el = node.lineno, node.end_lineno or node.lineno
                ct = "\n".join(lines[sl - 1:el])
                tp = "class" if isinstance(node, pyast.ClassDef) else "function"
                chunks.append({"chunk_id": gen_id(fp, node.name, idx), "file_path": fp,
                               "extension": ".py", "language": "python",
                               "symbol_name": f"{tp}:{node.name}", "chunk_text": ct,
                               "start_line": sl, "end_line": el,
                               "commit_sha": sha, "folder_context": ctx})
                idx += 1
        if not chunks:
            chunks.append({"chunk_id": gen_id(fp, "module", 0), "file_path": fp,
                           "extension": ".py", "language": "python",
                           "symbol_name": f"module:{os.path.basename(fp)}", "chunk_text": content,
                           "start_line": 1, "end_line": len(lines),
                           "commit_sha": sha, "folder_context": ctx})
        return chunks

    def parse_go(content, fp, sha, ctx):
        pat = re.compile(r"^(?:func\s+(?:\([^)]+\)\s+)?(\w+)|type\s+(\w+)\s+struct)\b", re.MULTILINE)
        matches = list(pat.finditer(content))
        if not matches:
            return [{"chunk_id": gen_id(fp, "file", 0), "file_path": fp,
                     "extension": ".go", "language": "go",
                     "symbol_name": f"file:{os.path.basename(fp)}", "chunk_text": content,
                     "start_line": 1, "end_line": content.count("\n") + 1,
                     "commit_sha": sha, "folder_context": ctx}]
        chunks = []
        for i, m in enumerate(matches):
            sym = m.group(1) or m.group(2)
            s, e = m.start(), matches[i + 1].start() if i + 1 < len(matches) else len(content)
            ct = content[s:e].rstrip()
            sl = content[:s].count("\n") + 1
            tp = "struct" if m.group(2) else "func"
            chunks.append({"chunk_id": gen_id(fp, sym, i), "file_path": fp,
                           "extension": ".go", "language": "go",
                           "symbol_name": f"{tp}:{sym}", "chunk_text": ct,
                           "start_line": sl, "end_line": sl + ct.count("\n"),
                           "commit_sha": sha, "folder_context": ctx})
        return chunks

    def parse_yaml_file(content, fp, sha, ctx):
        ext = os.path.splitext(fp)[1].lower()
        docs = content.split("\n---")
        chunks = []
        for idx, doc in enumerate(docs):
            doc = doc.strip()
            if not doc:
                continue
            try:
                parsed = yaml.safe_load(doc)
            except yaml.YAMLError:
                parsed = None
            if isinstance(parsed, dict):
                kind = parsed.get("kind", "Unknown")
                md = parsed.get("metadata", {})
                name = md.get("name", "unknown") if isinstance(md, dict) else "unknown"
                sym = f"{kind}:{name}"
                manifest_context = build_manifest_context(parsed, fp, ctx)
                chunk_body = f"{manifest_context}\n\n{doc}" if manifest_context else doc
            else:
                sym = f"fragment:{idx}"
                chunk_body = doc
            pre = "\n---".join(docs[:idx])
            sl = pre.count("\n") + 1 if pre else 1
            chunks.append({"chunk_id": gen_id(fp, sym, idx), "file_path": fp,
                           "extension": ext, "language": "yaml",
                           "symbol_name": sym, "chunk_text": chunk_body,
                           "start_line": sl, "end_line": sl + doc.count("\n"),
                           "commit_sha": sha, "folder_context": ctx})
        return chunks or [{"chunk_id": gen_id(fp, "file", 0), "file_path": fp,
                           "extension": ext, "language": "yaml",
                           "symbol_name": f"file:{os.path.basename(fp)}", "chunk_text": content,
                           "start_line": 1, "end_line": content.count("\n") + 1,
                           "commit_sha": sha, "folder_context": ctx}]

    def parse_md(content, fp, sha, ctx):
        pat = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
        matches = list(pat.finditer(content))
        if not matches:
            return [{"chunk_id": gen_id(fp, "doc", 0), "file_path": fp,
                     "extension": ".md", "language": "markdown",
                     "symbol_name": f"doc:{os.path.basename(fp)}", "chunk_text": content,
                     "start_line": 1, "end_line": content.count("\n") + 1,
                     "commit_sha": sha, "folder_context": ctx}]
        chunks = []
        for i, m in enumerate(matches):
            h = m.group(2).strip()
            s = m.start()
            e = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[s:e].strip()
            sl = content[:s].count("\n") + 1
            chunks.append({"chunk_id": gen_id(fp, h, i), "file_path": fp,
                           "extension": ".md", "language": "markdown",
                           "symbol_name": f"heading:{h[:100]}", "chunk_text": text,
                           "start_line": sl, "end_line": sl + text.count("\n"),
                           "commit_sha": sha, "folder_context": ctx})
        return chunks

    PARSERS = {".py": parse_python, ".go": parse_go,
               ".yaml": parse_yaml_file, ".yml": parse_yaml_file, ".md": parse_md}

    files = []
    with open(clone_data.path) as f:
        for line in f:
            if line.strip():
                files.append(json.loads(line))

    all_chunks = []
    for fi in files:
        parser = PARSERS.get(fi["extension"])
        if not parser:
            continue
        try:
            chunks = parser(fi["content"], fi["path"], fi["commit_sha"], fi["folder_context"])
            all_chunks.extend(chunks)
        except Exception as ex:
            logger.warning("Error parsing %s: %s", fi["path"], ex)

    logger.info("Parsed %d chunks from %d files", len(all_chunks), len(files))

    with open(parsed_data.path, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["tiktoken==0.7.0"],
)
def chunk_code(
    parsed_data: Input[Dataset],
    chunked_data: Output[Dataset],
):
    """Post-process parsed chunks with token limits and context headers.

    Args:
        parsed_data: Input dataset of parsed chunks.
        chunked_data: Output dataset of token-bounded chunks.
    """
    import hashlib
    import json
    import logging

    import tiktoken

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("chunker")

    enc = tiktoken.get_encoding("cl100k_base")
    count = lambda t: len(enc.encode(t))

    MIN_T, MAX_T = 50, 512

    def build_path_hints(chunk):
        raw = " ".join(str(chunk.get(key, "")) for key in ("file_path", "folder_context", "symbol_name"))
        expanded = raw.replace("/", " ").replace("_", " ").replace("-", " ")
        expanded = "".join(
            (
                f" {char}" if index > 0 and char.isupper() and expanded[index - 1].islower() else char
            )
            for index, char in enumerate(expanded)
        )
        return " ".join(expanded.split()).lower()

    raw = []
    with open(parsed_data.path) as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))

    processed = []
    for chunk in raw:
        header = (
            f"# File: {chunk.get('file_path', '?')} | Symbol: {chunk.get('symbol_name', '?')} "
            f"| Lang: {chunk.get('language', '?')} | Folder: {chunk.get('folder_context', '?')}"
        )
        path_hints = build_path_hints(chunk)
        if path_hints:
            header = f"{header}\n# Path Hints: {path_hints}"
        full = f"{header}\n\n{chunk['chunk_text']}"
        tc = count(full)

        if tc < MIN_T:
            continue

        if tc <= MAX_T:
            chunk["chunk_text"] = full[:8192]
            chunk["token_count"] = tc
            processed.append(chunk)
        else:
            # Split oversized
            parts = full.split("\n\n")
            cur, subs = "", []
            for p in parts:
                cand = cur + "\n\n" + p if cur else p
                if count(cand) > MAX_T:
                    if cur.strip():
                        subs.append(cur.strip())
                    cur = p
                else:
                    cur = cand
            if cur.strip():
                subs.append(cur.strip())

            for si, sub in enumerate(subs):
                st = count(sub)
                if st < MIN_T:
                    continue
                sc = chunk.copy()
                sc["chunk_id"] = hashlib.sha256(f"{chunk['chunk_id']}::{si}".encode()).hexdigest()[:32]
                sc["chunk_text"] = sub[:8192]
                sc["token_count"] = st
                processed.append(sc)

    logger.info("Chunked %d -> %d chunks", len(raw), len(processed))

    with open(chunked_data.path, "w") as f:
        for c in processed:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["sentence-transformers==2.7.0", "torch==2.3.0"],
)
def embed_code(
    chunked_data: Input[Dataset],
    embedding_model: str,
    embedded_data: Output[Dataset],
):
    """Embed code chunks using configurable model.

    Args:
        chunked_data: Input dataset of chunked code.
        embedding_model: Model name for embeddings.
        embedded_data: Output dataset with embeddings.
    """
    import json
    import logging

    from sentence_transformers import SentenceTransformer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("embedder")

    chunks = []
    with open(chunked_data.path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    logger.info("Embedding %d code chunks with %s", len(chunks), embedding_model)
    model = SentenceTransformer(embedding_model)

    texts = [c["chunk_text"] for c in chunks]
    bs = 32
    all_embs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        embs = model.encode(batch, show_progress_bar=False)
        all_embs.extend([e.tolist() for e in embs])
        logger.info("Batch %d/%d", i // bs + 1, (len(texts) + bs - 1) // bs)

    for c, e in zip(chunks, all_embs):
        c["embedding"] = e

    with open(embedded_data.path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pymilvus==2.4.0"],
)
def load_code(
    embedded_data: Input[Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    embedding_dim: int,
):
    """Load embedded code chunks into Milvus code_collection.

    Args:
        embedded_data: Input dataset with embedded chunks.
        milvus_host: Milvus server host.
        milvus_port: Milvus server port.
        collection_name: Target collection name.
        embedding_dim: Vector dimension.
    """
    import json
    import logging

    from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                          connections, utility)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("loader")

    connections.connect("default", host=milvus_host, port=milvus_port)

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema("file_path", DataType.VARCHAR, max_length=512),
            FieldSchema("extension", DataType.VARCHAR, max_length=16),
            FieldSchema("language", DataType.VARCHAR, max_length=32),
            FieldSchema("symbol_name", DataType.VARCHAR, max_length=256),
            FieldSchema("folder_context", DataType.VARCHAR, max_length=128),
            FieldSchema("chunk_text", DataType.VARCHAR, max_length=8192),
            FieldSchema("start_line", DataType.INT64),
            FieldSchema("end_line", DataType.INT64),
            FieldSchema("commit_sha", DataType.VARCHAR, max_length=64),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        schema = CollectionSchema(fields, "Kubeflow manifests code chunks")
        collection = Collection(collection_name, schema)
        collection.create_index("embedding", {
            "metric_type": "COSINE", "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        })
    else:
        collection = Collection(collection_name)

    collection.load()

    chunks = []
    with open(embedded_data.path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    rows = []
    for c in chunks:
        rows.append({
            "chunk_id": str(c["chunk_id"])[:128],
            "file_path": str(c.get("file_path", ""))[:512],
            "extension": str(c.get("extension", ""))[:16],
            "language": str(c.get("language", ""))[:32],
            "symbol_name": str(c.get("symbol_name", ""))[:256],
            "folder_context": str(c.get("folder_context", ""))[:128],
            "chunk_text": str(c.get("chunk_text", ""))[:8192],
            "start_line": int(c.get("start_line", 0)),
            "end_line": int(c.get("end_line", 0)),
            "commit_sha": str(c.get("commit_sha", ""))[:64],
            "chunk_index": int(c.get("chunk_index", 0)),
            "embedding": c["embedding"],
        })

    bs = 100
    inserted = 0
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        collection.upsert(batch)
        inserted += len(batch)

    collection.flush()
    logger.info("Loaded %d chunks into %s. Total: %d",
                inserted, collection_name, collection.num_entities)


# ─── Pipeline Definition ────────────────────────────────────────────────────

@dsl.pipeline(
    name="code-ingestion-pipeline",
    description="Clone kubeflow/manifests, parse code by language, embed, and load into Milvus",
)
def code_ingestion_pipeline(
    repo_url: str = "https://github.com/kubeflow/manifests",
    branch: str = "master",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    milvus_host: str = "localhost",
    milvus_port: str = "19530",
    collection_name: str = "code_collection",
    embedding_dim: int = 384,
):
    """Full code ingestion pipeline: clone -> parse -> chunk -> embed -> load."""

    clone_task = clone_repo(repo_url=repo_url, branch=branch)
    clone_task.set_retry(num_retries=3, backoff_duration="30s", backoff_factor=2.0)

    parse_task = parse_code(clone_data=clone_task.outputs["clone_data"])
    parse_task.set_retry(num_retries=3, backoff_duration="30s", backoff_factor=2.0)

    chunk_task = chunk_code(parsed_data=parse_task.outputs["parsed_data"])
    chunk_task.set_retry(num_retries=3, backoff_duration="30s", backoff_factor=2.0)

    embed_task = embed_code(
        chunked_data=chunk_task.outputs["chunked_data"],
        embedding_model=embedding_model,
    )
    embed_task.set_retry(num_retries=3, backoff_duration="30s", backoff_factor=2.0)

    load_task = load_code(
        embedded_data=embed_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        embedding_dim=embedding_dim,
    )
    load_task.set_retry(num_retries=3, backoff_duration="30s", backoff_factor=2.0)


# ─── Parent Pipeline (Composes Both) ────────────────────────────────────────

if docs_ingestion_pipeline is not None:
    @dsl.pipeline(
        name="full-ingestion-pipeline",
        description="Run both docs and code ingestion pipelines in parallel",
    )
    def full_ingestion_pipeline(
        # Docs params
        docs_base_url: str = "https://www.kubeflow.org",
        docs_crawl_delay: float = 1.0,
        docs_max_pages: int = 0,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        # Code params
        code_repo_url: str = "https://github.com/kubeflow/manifests",
        code_branch: str = "master",
        # Shared params
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        embedding_dim: int = 384,
    ):
        """Parent pipeline that runs docs + code ingestion in parallel."""
        docs_ingestion_pipeline(
            base_url=docs_base_url,
            crawl_delay=docs_crawl_delay,
            max_pages=docs_max_pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name="docs_collection",
            embedding_dim=embedding_dim,
        )
        code_ingestion_pipeline(
            repo_url=code_repo_url,
            branch=code_branch,
            embedding_model=embedding_model,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            collection_name="code_collection",
            embedding_dim=embedding_dim,
        )
else:
    full_ingestion_pipeline = None




# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--local" in sys.argv:
        print("Running code ingestion pipeline locally...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from pipelines.code_ingestion.components.repo_cloner import clone_repo as do_clone
        from pipelines.code_ingestion.components.ast_parser import parse_all_files
        from pipelines.code_ingestion.components.chunker import process_chunks
        from pipelines.code_ingestion.components.embedder import embed_code_chunks
        from pipelines.code_ingestion.components.loader import load_to_milvus
        import logging, shutil
        logging.basicConfig(level=logging.INFO)

        result = do_clone()
        chunks = parse_all_files(result["repo_dir"], result["file_list"], result["commit_sha"])
        processed = process_chunks(chunks)
        embedded = embed_code_chunks(processed)
        summary = load_to_milvus(embedded)
        print(f"Pipeline complete: {summary}")
        shutil.rmtree(result["repo_dir"], ignore_errors=True)
    else:
        output_path = os.path.join(os.path.dirname(__file__), "pipeline.yaml")
        kfp.compiler.Compiler().compile(
            pipeline_func=code_ingestion_pipeline,
            package_path=output_path,
        )
        print(f"Compiled code ingestion pipeline to: {output_path}")

        if full_ingestion_pipeline is not None:
            full_output_path = os.path.join(
                os.path.dirname(__file__),
                "full_pipeline.yaml",
            )
            kfp.compiler.Compiler().compile(
                pipeline_func=full_ingestion_pipeline,
                package_path=full_output_path,
            )
            print(f"Compiled full ingestion pipeline to: {full_output_path}")
        else:
            print(
                "Skipped full ingestion pipeline compilation because the docs "
                "pipeline import was unavailable."
            )
