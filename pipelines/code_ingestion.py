import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["gitpython"]
)
def clone_manifests(
    repo_url: str,
    branch: str,
    cloned_repo: dsl.Output[dsl.Dataset]
):
    import git
    import json
    from pathlib import Path

    print(f"Cloning {repo_url} branch={branch}")
    git.Repo.clone_from(
        repo_url,
        "/tmp/manifests",
        branch=branch,
        depth=1
    )

    all_files = [
        str(f) for f in Path("/tmp/manifests").rglob("*")
        if f.is_file()
    ]

    print(f"Found {len(all_files)} files")

    with open(cloned_repo.path, "w") as f:
        json.dump({
            "repo_path": "/tmp/manifests",
            "repo_url":  repo_url,
            "branch":    branch,
            "files":     all_files
        }, f)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pyyaml"]
)
def parse_yaml_resources(
    cloned_repo: dsl.Input[dsl.Dataset],
    parsed_yaml: dsl.Output[dsl.Dataset]
):
    import yaml
    import json

    with open(cloned_repo.path) as f:
        meta = json.load(f)

    repo_path = meta["repo_path"]
    repo_url  = meta["repo_url"]
    branch    = meta["branch"]
    units     = []

    for file_path in meta["files"]:
        if not file_path.endswith((".yaml", ".yml")):
            continue
        try:
            with open(file_path, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            for doc in docs:
                if not doc or "kind" not in doc:
                    continue
                rel = file_path.replace(repo_path + "/", "").replace(repo_path + "\\", "")
                units.append({
                    "text":         yaml.dump(doc, default_flow_style=False),
                    "file_path":    rel,
                    "source_url":   f"{repo_url}/blob/{branch}/{rel}",
                    "kind":         doc.get("kind", ""),
                    "name":         doc.get("metadata", {}).get("name", "") if doc.get("metadata") else "",
                    "content_type": "kubernetes_resource"
                })
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    print(f"Parsed {len(units)} YAML resources")

    with open(parsed_yaml.path, "w") as f:
        json.dump(units, f)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[]
)
def parse_python_ast(
    cloned_repo: dsl.Input[dsl.Dataset],
    parsed_python: dsl.Output[dsl.Dataset]
):
    import ast
    import json
    from pathlib import Path

    with open(cloned_repo.path) as f:
        meta = json.load(f)

    repo_path = meta["repo_path"]
    repo_url  = meta["repo_url"]
    branch    = meta["branch"]
    units     = []

    for file_path in meta["files"]:
        if not file_path.endswith(".py"):
            continue
        try:
            source = Path(file_path).read_text(encoding="utf-8")
            tree   = ast.parse(source)
            rel    = file_path.replace(repo_path + "/", "").replace(repo_path + "\\", "")

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                snippet   = ast.get_source_segment(source, node) or ""
                if len(snippet) < 20:
                    continue
                docstring = ast.get_docstring(node) or ""
                units.append({
                    "text":         snippet,
                    "file_path":    rel,
                    "source_url":   f"{repo_url}/blob/{branch}/{rel}#L{node.lineno}",
                    "kind":         type(node).__name__,
                    "name":         node.name,
                    "content_type": "python_definition",
                    "docstring":    docstring[:300]
                })
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    print(f"Parsed {len(units)} Python definitions")

    with open(parsed_python.path, "w") as f:
        json.dump(units, f)


@dsl.component(
    base_image="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["sentence-transformers"]
)
def embed_code(
    parsed_yaml: dsl.Input[dsl.Dataset],
    parsed_python: dsl.Input[dsl.Dataset],
    embedded_code: dsl.Output[dsl.Dataset]
):
    import json
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    print(f"Model loaded on {device}")

    # Load both parsed sources
    units = []
    for input_file in [parsed_yaml.path, parsed_python.path]:
        with open(input_file) as f:
            units.extend(json.load(f))

    print(f"Total units to embed: {len(units)}")

    records = []
    for unit in units:
        text      = unit["text"][:1000]
        embedding = model.encode(text).tolist()
        records.append({
            "text":         unit["text"][:4000],
            "file_path":    unit["file_path"],
            "source_url":   unit["source_url"],
            "kind":         unit.get("kind", ""),
            "name":         unit.get("name", ""),
            "content_type": unit.get("content_type", ""),
            "embedding":    embedding
        })

    print(f"Embedded {len(records)} units")

    with open(embedded_code.path, "w") as f:
        json.dump(records, f)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pymilvus"]
)
def store_code_index(
    embedded_code: dsl.Input[dsl.Dataset],
    milvus_host: str,
    milvus_port: str,
    collection_name: str
):
    import json
    from datetime import datetime
    from pymilvus import (
        connections, utility,
        FieldSchema, CollectionSchema, DataType, Collection
    )

    connections.connect("default", host=milvus_host, port=milvus_port)

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    fields = [
        FieldSchema(name="id",           dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="text",         dtype=DataType.VARCHAR,        max_length=4000),
        FieldSchema(name="file_path",    dtype=DataType.VARCHAR,        max_length=500),
        FieldSchema(name="source_url",   dtype=DataType.VARCHAR,        max_length=500),
        FieldSchema(name="kind",         dtype=DataType.VARCHAR,        max_length=100),
        FieldSchema(name="name",         dtype=DataType.VARCHAR,        max_length=200),
        FieldSchema(name="content_type", dtype=DataType.VARCHAR,        max_length=50),
        FieldSchema(name="last_updated", dtype=DataType.INT64),
        FieldSchema(name="vector",       dtype=DataType.FLOAT_VECTOR,   dim=768),
    ]

    schema     = CollectionSchema(fields, "Kubeflow manifests code index")
    collection = Collection(collection_name, schema)
    print(f"Created collection: {collection_name}")

    with open(embedded_code.path) as f:
        records = json.load(f)

    timestamp  = int(datetime.now().timestamp())
    batch_size = 500

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        collection.insert([{
            "text":         r["text"],
            "file_path":    r["file_path"],
            "source_url":   r["source_url"],
            "kind":         r["kind"],
            "name":         r["name"],
            "content_type": r["content_type"],
            "last_updated": timestamp,
            "vector":       r["embedding"]
        } for r in batch])

    collection.flush()

    index_params = {
        "metric_type": "COSINE",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": min(1024, len(records))}
    }
    collection.create_index("vector", index_params)
    collection.load()
    print(f"Stored {collection.num_entities} records in {collection_name}")


@dsl.pipeline(
    name="kubeflow-code-ingestion",
    description="AST-based code ingestion for kubeflow/manifests"
)
def code_ingestion_pipeline(
    repo_url:        str = "https://github.com/kubeflow/manifests",
    branch:          str = "master",
    milvus_host:     str = "milvus-standalone-final.docs-agent.svc.cluster.local",
    milvus_port:     str = "19530",
    collection_name: str = "code_index"
):
    clone_task  = clone_manifests(repo_url=repo_url, branch=branch)

    yaml_task   = parse_yaml_resources(
        cloned_repo=clone_task.outputs["cloned_repo"]
    )

    python_task = parse_python_ast(
        cloned_repo=clone_task.outputs["cloned_repo"]
    )

    embed_task  = embed_code(
        parsed_yaml=yaml_task.outputs["parsed_yaml"],
        parsed_python=python_task.outputs["parsed_python"]
    )

    store_task  = store_code_index(
        embedded_code=embed_task.outputs["embedded_code"],
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=code_ingestion_pipeline,
        package_path="code_ingestion_pipeline.yaml"
    )
    print("Pipeline compiled to code_ingestion_pipeline.yaml")