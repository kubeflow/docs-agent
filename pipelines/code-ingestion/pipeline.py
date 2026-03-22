import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *


@dsl.component(
    base_image="python:3.9",
)
def download_code_repository(
    repo_owner: str,
    repo_name: str,
    branch: str,
    github_token: str,
    code_data: dsl.Output[dsl.Dataset]
):
    import subprocess
    import os
    import json

    clone_dir = "/tmp/repo_clone"
    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    if github_token:
        repo_url = f"https://{github_token}@github.com/{repo_owner}/{repo_name}.git"

    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, clone_dir],
        check=True, capture_output=True, text=True
    )
    print(f"Cloned {repo_owner}/{repo_name} @ {branch}")

    VALID_EXTENSIONS = {".yaml", ".yml", ".sh", ".bash"}
    VALID_FILENAMES = {"Dockerfile"}
    SKIP_DIRS = {"vendor", ".github", "tests", "test", "node_modules", ".git", "__pycache__"}

    def should_skip_dir(dirname):
        return dirname in SKIP_DIRS or dirname.startswith(".")

    def is_valid_file(name):
        if name in VALID_FILENAMES or name.startswith("Dockerfile"):
            return True
        for ext in VALID_EXTENSIONS:
            if name.endswith(ext):
                return True
        return False

    files = []
    for root, dirs, filenames in os.walk(clone_dir):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        for fname in filenames:
            if not is_valid_file(fname):
                continue
            abs_path = os.path.join(root, fname)
            rel_path = os.path.relpath(abs_path, clone_dir)

            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, IOError):
                continue

            files.append({
                "path": rel_path,
                "content": content,
                "file_name": fname
            })

    print(f"Found {len(files)} code files")

    with open(code_data.path, "w", encoding="utf-8") as f:
        for file_data in files:
            f.write(json.dumps(file_data, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pyyaml"]
)
def parse_and_chunk_code(
    code_data: dsl.Input[dsl.Dataset],
    repo_name: str,
    parsed_data: dsl.Output[dsl.Dataset]
):
    import json
    import os
    import re
    from typing import List, Dict, Any, Optional

    KNOWN_TOP_DIRS = {"applications", "common", "experimental"}

    def _infer_component_name(file_path):
        parts = file_path.replace("\\", "/").split("/")
        skip = {"base", "overlays", "upstream", "default", ".", "", "installs", "components"}

        for top_dir in ("applications", "common", "experimental"):
            if top_dir in parts:
                idx = parts.index(top_dir)
                if idx + 1 < len(parts) and parts[idx + 1] not in skip:
                    return parts[idx + 1]

        if parts[0] == "scripts":
            return "scripts"

        for part in parts[:-1]:
            if part not in skip and part not in KNOWN_TOP_DIRS:
                return part

        return ""

    def _safe_dump_yaml_block(key, value):
        try:
            import yaml
            dumped = yaml.dump({key: value}, default_flow_style=False, allow_unicode=True)
            return dumped.strip()
        except Exception:
            return f"{key}: {value}"

    def _extract_labels(doc):
        metadata = doc.get("metadata", {}) or {}
        labels = metadata.get("labels", {}) or {}
        annotations = metadata.get("annotations", {}) or {}
        result = {}
        for k, v in labels.items():
            result[f"label:{k}"] = str(v)
        for k, v in list(annotations.items())[:5]:
            result[f"annotation:{k}"] = str(v)
        return result

    def parse_yaml_manifest(file_path, content):
        import yaml
        chunks = []
        raw_docs = content.split("\n---")

        for raw_doc in raw_docs:
            raw_doc = raw_doc.strip()
            if not raw_doc or raw_doc == "---":
                continue
            try:
                doc = yaml.safe_load(raw_doc)
            except yaml.YAMLError:
                continue
            if not isinstance(doc, dict):
                continue

            kind = doc.get("kind", "")
            api_version = doc.get("apiVersion", "")
            metadata_block = doc.get("metadata", {}) or {}
            name = metadata_block.get("name", "")
            namespace = metadata_block.get("namespace", "")
            component = _infer_component_name(file_path)
            labels_map = _extract_labels(doc)

            header_parts = []
            if kind and name:
                loc = f"# {kind}/{name}"
                if namespace:
                    loc += f" in namespace {namespace}"
                header_parts.append(loc)
            if api_version:
                header_parts.append(f"# apiVersion: {api_version}")
            header_parts.append(f"# File: {file_path}")
            if component:
                header_parts.append(f"# Component: {component}")
            header = "\n".join(header_parts)

            code_meta = {
                "code_entity_type": kind if kind else "k8s_resource",
                "code_entity_name": name,
                "component_name": component,
            }
            if namespace:
                code_meta["k8s_namespace"] = namespace
            if api_version:
                code_meta["api_version"] = api_version
            if labels_map:
                code_meta["labels"] = labels_map

            top_level_keys = [k for k in doc.keys() if k not in ("apiVersion", "kind", "metadata")]

            if not top_level_keys:
                chunk_text = f"{header}\n\n{raw_doc}"
                chunks.append({"language": "yaml", "code_metadata": code_meta, "content_text": chunk_text[:4000]})
                continue

            for key in top_level_keys:
                section_text = _safe_dump_yaml_block(key, doc[key])
                chunk_text = f"{header}\n# Section: {key}\n\n{section_text}"
                chunks.append({"language": "yaml", "code_metadata": {**code_meta, "section": key}, "content_text": chunk_text[:4000]})

        if not chunks:
            chunks.append({
                "language": "yaml",
                "code_metadata": {"code_entity_type": "", "code_entity_name": "", "component_name": _infer_component_name(file_path)},
                "content_text": content[:4000]
            })
        return chunks

    def parse_kustomization(file_path, content):
        import yaml
        try:
            doc = yaml.safe_load(content)
        except Exception:
            doc = {}
        if not isinstance(doc, dict):
            doc = {}

        component = _infer_component_name(file_path)
        resources = doc.get("resources", []) or []
        bases = doc.get("bases", []) or []
        images = doc.get("images", []) or []
        namespace = doc.get("namespace", "")
        patches_sm = doc.get("patchesStrategicMerge", []) or []
        patches_json = doc.get("patchesJson6902", []) or []
        patches_inline = doc.get("patches", []) or []
        config_map_gen = doc.get("configMapGenerator", []) or []
        secret_gen = doc.get("secretGenerator", []) or []
        components_list = doc.get("components", []) or []

        is_overlay = "overlays" in file_path or "overlay" in file_path
        kustomize_type = "overlay" if is_overlay else "base"

        header_parts = [f"# Kustomization ({kustomize_type})", f"# File: {file_path}"]
        if component:
            header_parts.append(f"# Component: {component}")
        if namespace:
            header_parts.append(f"# Namespace: {namespace}")
        if resources:
            header_parts.append(f"# Resources: {', '.join(str(r) for r in resources[:10])}")
        if bases:
            header_parts.append(f"# Bases: {', '.join(str(b) for b in bases[:10])}")
        if images:
            img_names = [img.get("name", str(img)) if isinstance(img, dict) else str(img) for img in images[:5]]
            header_parts.append(f"# Images: {', '.join(img_names)}")
        if components_list:
            header_parts.append(f"# Components: {', '.join(str(c) for c in components_list[:5])}")

        header = "\n".join(header_parts)
        chunk_text = f"{header}\n\n{content}"

        code_meta = {
            "code_entity_type": "kustomization",
            "code_entity_name": component if component else "kustomization",
            "component_name": component,
            "kustomize_type": kustomize_type,
        }
        if namespace:
            code_meta["k8s_namespace"] = namespace
        if resources:
            code_meta["resources"] = resources[:20]
        if images:
            code_meta["images"] = [img.get("name", str(img)) if isinstance(img, dict) else str(img) for img in images[:10]]
        if patches_sm or patches_json or patches_inline:
            code_meta["has_patches"] = True
        if config_map_gen:
            code_meta["config_map_generators"] = [g.get("name", "") for g in config_map_gen if isinstance(g, dict)]

        return [{"language": "yaml", "code_metadata": code_meta, "content_text": chunk_text[:4000]}]

    def parse_dockerfile(file_path, content):
        component = _infer_component_name(file_path)
        from_images = re.findall(r"^FROM\s+(\S+)(?:\s+[Aa][Ss]\s+(\S+))?", content, re.MULTILINE)
        expose_ports = re.findall(r"^EXPOSE\s+(.+)", content, re.MULTILINE)
        entrypoints = re.findall(r"^(?:ENTRYPOINT|CMD)\s+(.+)", content, re.MULTILINE)
        labels = re.findall(r'^LABEL\s+(.+)', content, re.MULTILINE)
        args = re.findall(r'^ARG\s+(\S+)', content, re.MULTILINE)
        workdirs = re.findall(r'^WORKDIR\s+(\S+)', content, re.MULTILINE)
        healthchecks = re.findall(r'^HEALTHCHECK\s+(.+)', content, re.MULTILINE)
        envs = re.findall(r'^ENV\s+(\S+)', content, re.MULTILINE)

        base_imgs = [img[0] for img in from_images]
        stage_names = [img[1] for img in from_images if img[1]]
        is_multi_stage = len(from_images) > 1

        header_parts = [f"# Dockerfile", f"# File: {file_path}"]
        if component:
            header_parts.append(f"# Component: {component}")
        if base_imgs:
            header_parts.append(f"# Base images: {', '.join(base_imgs)}")
        if is_multi_stage:
            header_parts.append(f"# Multi-stage build: {len(from_images)} stages")
        if expose_ports:
            header_parts.append(f"# Exposed ports: {', '.join(expose_ports)}")
        if entrypoints:
            header_parts.append(f"# Entrypoint: {entrypoints[-1].strip()}")

        header = "\n".join(header_parts)
        chunk_text = f"{header}\n\n{content}"

        code_meta = {
            "code_entity_type": "dockerfile",
            "code_entity_name": base_imgs[0] if base_imgs else "",
            "component_name": component,
            "base_images": base_imgs,
        }
        if is_multi_stage:
            code_meta["is_multi_stage"] = True
            code_meta["stage_names"] = stage_names
        if expose_ports:
            code_meta["exposed_ports"] = [p.strip() for p in expose_ports]
        if entrypoints:
            code_meta["entrypoint"] = entrypoints[-1].strip()
        if args:
            code_meta["build_args"] = args[:10]
        if envs:
            code_meta["env_vars"] = envs[:10]

        return [{"language": "dockerfile", "code_metadata": code_meta, "content_text": chunk_text[:4000]}]

    def parse_shell(file_path, content):
        component = _infer_component_name(file_path)
        shebang_match = re.match(r"^#!\s*(.+)", content)
        shebang = shebang_match.group(1).strip() if shebang_match else ""

        sourced_files = re.findall(r'^\s*(?:source|\\.)\s+["\']?([^\s"\']+)', content, re.MULTILINE)
        export_vars = re.findall(r'^\s*export\s+(\w+)=', content, re.MULTILINE)
        set_flags = re.findall(r'^\s*set\s+(-\w+)', content, re.MULTILINE)

        function_pattern = re.compile(
            r"^(?:function\s+(\w[\w-]*)\s*(?:\(\s*\))?\s*\{|(\w[\w-]*)\s*\(\s*\)\s*\{)",
            re.MULTILINE
        )
        function_matches = list(function_pattern.finditer(content))

        base_header_parts = [f"# Shell script", f"# File: {file_path}"]
        if component:
            base_header_parts.append(f"# Component: {component}")
        if shebang:
            base_header_parts.append(f"# Shebang: {shebang}")
        if sourced_files:
            base_header_parts.append(f"# Sources: {', '.join(sourced_files[:5])}")

        base_meta = {"component_name": component}
        if shebang:
            base_meta["shebang"] = shebang
        if sourced_files:
            base_meta["sourced_files"] = sourced_files
        if export_vars:
            base_meta["exported_variables"] = export_vars[:15]

        if len(function_matches) < 2:
            header = "\n".join(base_header_parts)
            chunk_text = f"{header}\n\n{content}"
            meta = {**base_meta, "code_entity_type": "script", "code_entity_name": file_path.split("/")[-1]}
            if function_matches:
                fn = function_matches[0].group(1) or function_matches[0].group(2)
                meta["functions"] = [fn]
            return [{"language": "shell", "code_metadata": meta, "content_text": chunk_text[:4000]}]

        chunks = []
        all_func_names = [m.group(1) or m.group(2) for m in function_matches]

        preamble_end = function_matches[0].start()
        preamble = content[:preamble_end].strip()
        if preamble:
            header = "\n".join(base_header_parts)
            chunk_text = f"{header}\n# Section: preamble\n\n{preamble}"
            chunks.append({
                "language": "shell",
                "code_metadata": {**base_meta, "code_entity_type": "script", "code_entity_name": file_path.split("/")[-1], "functions": all_func_names},
                "content_text": chunk_text[:4000]
            })

        for i, match in enumerate(function_matches):
            func_name = match.group(1) or match.group(2)
            start = match.start()
            end = function_matches[i + 1].start() if i + 1 < len(function_matches) else len(content)
            func_body = content[start:end].strip()

            func_header_parts = base_header_parts + [f"# Function: {func_name}"]
            header = "\n".join(func_header_parts)
            chunk_text = f"{header}\n\n{func_body}"

            chunks.append({
                "language": "shell",
                "code_metadata": {**base_meta, "code_entity_type": "function", "code_entity_name": func_name},
                "content_text": chunk_text[:4000]
            })

        return chunks if chunks else [{"language": "shell", "code_metadata": {**base_meta, "code_entity_type": "script", "code_entity_name": file_path.split("/")[-1]}, "content_text": content[:4000]}]

    def dispatch_parse(file_path, content):
        filename = file_path.split("/")[-1].lower()
        kustomization_names = {"kustomization.yaml", "kustomization.yml", "kustomization"}
        if filename in kustomization_names:
            return parse_kustomization(file_path, content)
        if filename.startswith("dockerfile"):
            return parse_dockerfile(file_path, content)
        if filename.endswith(".sh") or filename.endswith(".bash"):
            return parse_shell(file_path, content)
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            return parse_yaml_manifest(file_path, content)
        return [{"language": "", "code_metadata": {"code_entity_type": "", "code_entity_name": "", "component_name": _infer_component_name(file_path)}, "content_text": content[:4000]}]

    records = []

    with open(code_data.path, "r", encoding="utf-8") as f:
        for line in f:
            file_data = json.loads(line)
            file_path = file_data["path"]
            content = file_data["content"]

            if len(content.strip()) < 10:
                continue

            chunks = dispatch_parse(file_path, content)
            github_url = f"https://github.com/kubeflow/{repo_name}/blob/master/{file_path}"

            for chunk_idx, chunk in enumerate(chunks):
                file_unique_id = f"{repo_name}:{file_path}:{chunk_idx}"
                records.append({
                    "file_unique_id": file_unique_id,
                    "repo_name": repo_name,
                    "file_path": file_path,
                    "file_name": file_data["file_name"],
                    "citation_url": github_url[:1024],
                    "chunk_index": chunk_idx,
                    "content_text": chunk["content_text"],
                    "language": chunk["language"],
                    "code_metadata": chunk["code_metadata"],
                })

    print(f"Parsed {len(records)} chunks from code files")

    with open(parsed_data.path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["sentence-transformers"]
)
def embed_code_chunks(
    parsed_data: dsl.Input[dsl.Dataset],
    embedded_data: dsl.Output[dsl.Dataset]
):
    import json
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    print(f"Model loaded on {device}")

    records = []
    with open(parsed_data.path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            embedding = model.encode(record["content_text"]).tolist()
            record["embedding"] = embedding
            records.append(record)

    print(f"Embedded {len(records)} chunks")

    with open(embedded_data.path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pymilvus", "numpy"]
)
def store_code_milvus(
    embedded_data: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str
):
    from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
    import json
    from datetime import datetime

    connections.connect("default", host=milvus_host, port=milvus_port)

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_unique_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="citation_url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=4000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="last_updated", dtype=DataType.INT64),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=64), # for code 
            FieldSchema(name="code_metadata", dtype=DataType.JSON, nullable=True), # for code 
        ]
        schema = CollectionSchema(fields, "Unified RAG collection for documentation and code")
        collection = Collection(collection_name, schema)
        print(f"Created new collection: {collection_name}")
    else:
        collection = Collection(collection_name)
        print(f"Using existing collection: {collection_name}")

    partition_name = "code"
    if not collection.has_partition(partition_name):
        collection.create_partition(partition_name)
        print(f"Created partition: {partition_name}")
    else:
        print(f"Partition already exists: {partition_name}")

    records = []
    timestamp = int(datetime.now().timestamp())
    repo_names_seen = set()

    with open(embedded_data.path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            repo_names_seen.add(record["repo_name"])
            records.append({
                "file_unique_id": record["file_unique_id"],
                "repo_name": record["repo_name"],
                "file_path": record["file_path"],
                "file_name": record["file_name"],
                "citation_url": record["citation_url"],
                "chunk_index": record["chunk_index"],
                "content_text": record["content_text"],
                "vector": record["embedding"],
                "last_updated": timestamp,
                "language": record.get("language", ""),
                "code_metadata": record.get("code_metadata", None),
            })

    if records:
        collection.load()
        for repo in repo_names_seen:
            try:
                expr = f'repo_name == "{repo}"'
                collection.delete(expr, partition_name=partition_name)
                print(f"Deleted old chunks for repo '{repo}' from partition '{partition_name}'")
            except Exception as e:
                print(f"No old data to delete for '{repo}': {e}")

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            collection.insert(batch, partition_name=partition_name)

        collection.flush()

        try:
            collection.index()
        except Exception:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": min(1024, max(100, len(records)))}
            }
            collection.create_index("vector", index_params)

        collection.load()
        print(f"Inserted {len(records)} records into partition '{partition_name}'. Total: {collection.num_entities}")


@dsl.pipeline(
    name="code-ingestion",
    description="Code ingestion pipeline for parsing and embedding Kubeflow release code repositories"
)
def code_ingestion_pipeline(
    repo_owner: str = "kubeflow",
    repo_name: str = "manifests",
    branch: str = "master",
    github_token: str = "",
    milvus_host: str = "milvus-standalone-final.docs-agent.svc.cluster.local",
    milvus_port: str = "19530",
    collection_name: str = "docs_rag"
):
    download_task = download_code_repository(
        repo_owner=repo_owner,
        repo_name=repo_name,
        branch=branch,
        github_token=github_token
    )

    parse_task = parse_and_chunk_code(
        code_data=download_task.outputs["code_data"],
        repo_name=repo_name,
    )

    embed_task = embed_code_chunks(
        parsed_data=parse_task.outputs["parsed_data"],
    )

    store_task = store_code_milvus(
        embedded_data=embed_task.outputs["embedded_data"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    import os
    os.environ["KFP_DISABLE_EXECUTION_CACHING_BY_DEFAULT"] = "true"

    kfp.compiler.Compiler().compile(
        pipeline_func=code_ingestion_pipeline,
        package_path="code_ingestion_pipeline.yaml"
    )
    print("Pipeline compiled to code_ingestion_pipeline.yaml")