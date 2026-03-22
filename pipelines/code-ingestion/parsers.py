# parsers.py is used only by the test_parsers.py 
"""
Structure-aware code parsers for the code ingestion pipeline.

This module is the canonical, testable implementation of all parsing logic.
It is imported by test_parsers.py for unit testing. The KFP pipeline
(pipeline.py) inlines a copy of this logic because KFP @dsl.component
functions cannot import external modules — they are serialized into
self-contained container images.

When updating parsing logic, update BOTH this file AND pipeline.py.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


KNOWN_TOP_DIRS = {"applications", "common", "experimental"}


@dataclass
class ParsedChunk:
    content_text: str
    language: str = ""
    code_metadata: Optional[Dict[str, Any]] = None


def _infer_component_name(file_path: str) -> str:
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


def _safe_dump_yaml_block(key: str, value: Any) -> str:
    try:
        import yaml
        dumped = yaml.dump({key: value}, default_flow_style=False, allow_unicode=True)
        return dumped.strip()
    except Exception:
        return f"{key}: {value}"


def _extract_labels(doc: dict) -> Dict[str, str]:
    metadata = doc.get("metadata", {}) or {}
    labels = metadata.get("labels", {}) or {}
    annotations = metadata.get("annotations", {}) or {}
    result = {}
    for k, v in labels.items():
        result[f"label:{k}"] = str(v)
    for k, v in list(annotations.items())[:5]:
        result[f"annotation:{k}"] = str(v)
    return result


class YAMLManifestParser:
    def parse(self, file_path: str, content: str) -> List[ParsedChunk]:
        try:
            import yaml
        except ImportError:
            return [ParsedChunk(
                content_text=content[:4000],
                language="yaml",
                code_metadata={
                    "code_entity_type": "",
                    "code_entity_name": "",
                    "component_name": _infer_component_name(file_path),
                }
            )]

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
                "code_entity_type": kind if kind else "",
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
                chunks.append(ParsedChunk(
                    content_text=chunk_text[:4000],
                    language="yaml",
                    code_metadata=code_meta,
                ))
                continue

            for key in top_level_keys:
                section_text = _safe_dump_yaml_block(key, doc[key])
                chunk_text = f"{header}\n# Section: {key}\n\n{section_text}"

                chunks.append(ParsedChunk(
                    content_text=chunk_text[:4000],
                    language="yaml",
                    code_metadata={**code_meta, "section": key},
                ))

        if not chunks:
            chunks.append(ParsedChunk(
                content_text=content[:4000],
                language="yaml",
                code_metadata={
                    "code_entity_type": "",
                    "code_entity_name": "",
                    "component_name": _infer_component_name(file_path),
                }
            ))

        return chunks


class KustomizationParser:
    KUSTOMIZATION_FILENAMES = {"kustomization.yaml", "kustomization.yml", "kustomization"}

    def parse(self, file_path: str, content: str) -> List[ParsedChunk]:
        try:
            import yaml
            doc = yaml.safe_load(content)
        except Exception:
            doc = {}

        if not isinstance(doc, dict):
            doc = {}

        component = _infer_component_name(file_path)
        resources = doc.get("resources", []) or []
        bases = doc.get("bases", []) or []
        patches_sm = doc.get("patchesStrategicMerge", []) or []
        patches_json = doc.get("patchesJson6902", []) or []
        patches_inline = doc.get("patches", []) or []
        config_map_gen = doc.get("configMapGenerator", []) or []
        secret_gen = doc.get("secretGenerator", []) or []
        images = doc.get("images", []) or []
        transformers = doc.get("transformers", []) or []
        components_list = doc.get("components", []) or []
        namespace = doc.get("namespace", "")
        common_labels = doc.get("commonLabels", {}) or {}
        common_annotations = doc.get("commonAnnotations", {}) or {}

        is_overlay = "overlays" in file_path or "overlay" in file_path
        kustomize_type = "overlay" if is_overlay else "base"

        header_parts = [
            f"# Kustomization ({kustomize_type})",
            f"# File: {file_path}",
        ]
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
        if bases:
            code_meta["bases"] = bases[:10]
        if images:
            code_meta["images"] = [img.get("name", str(img)) if isinstance(img, dict) else str(img) for img in images[:10]]
        if patches_sm or patches_json or patches_inline:
            code_meta["has_patches"] = True
        if config_map_gen:
            code_meta["config_map_generators"] = [g.get("name", "") for g in config_map_gen if isinstance(g, dict)]
        if secret_gen:
            code_meta["secret_generators"] = [g.get("name", "") for g in secret_gen if isinstance(g, dict)]

        return [ParsedChunk(
            content_text=chunk_text[:4000],
            language="yaml",
            code_metadata=code_meta,
        )]


class DockerfileParser:
    def parse(self, file_path: str, content: str) -> List[ParsedChunk]:
        component = _infer_component_name(file_path)

        from_images = re.findall(r"^FROM\s+(\S+)(?:\s+[Aa][Ss]\s+(\S+))?", content, re.MULTILINE)
        expose_ports = re.findall(r"^EXPOSE\s+(.+)", content, re.MULTILINE)
        entrypoints = re.findall(r"^(?:ENTRYPOINT|CMD)\s+(.+)", content, re.MULTILINE)
        labels = re.findall(r'^LABEL\s+(.+)', content, re.MULTILINE)
        args = re.findall(r'^ARG\s+(\S+)', content, re.MULTILINE)
        workdirs = re.findall(r'^WORKDIR\s+(\S+)', content, re.MULTILINE)
        healthchecks = re.findall(r'^HEALTHCHECK\s+(.+)', content, re.MULTILINE)
        envs = re.findall(r'^ENV\s+(\S+)', content, re.MULTILINE)
        copy_from = re.findall(r'^COPY\s+--from=(\S+)', content, re.MULTILINE)
        run_cmds = re.findall(r'^RUN\s+(.+)', content, re.MULTILINE)

        base_images = [img[0] for img in from_images]
        stage_names = [img[1] for img in from_images if img[1]]
        is_multi_stage = len(from_images) > 1

        header_parts = [
            f"# Dockerfile",
            f"# File: {file_path}",
        ]
        if component:
            header_parts.append(f"# Component: {component}")
        if base_images:
            header_parts.append(f"# Base images: {', '.join(base_images)}")
        if is_multi_stage:
            header_parts.append(f"# Multi-stage build: {len(from_images)} stages")
            if stage_names:
                header_parts.append(f"# Stage names: {', '.join(stage_names)}")
        if expose_ports:
            header_parts.append(f"# Exposed ports: {', '.join(expose_ports)}")
        if entrypoints:
            header_parts.append(f"# Entrypoint: {entrypoints[-1].strip()}")
        if workdirs:
            header_parts.append(f"# Workdir: {workdirs[-1]}")

        header = "\n".join(header_parts)
        chunk_text = f"{header}\n\n{content}"

        code_meta = {
            "code_entity_type": "dockerfile",
            "code_entity_name": base_images[0] if base_images else "",
            "component_name": component,
            "base_images": base_images,
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
        if labels:
            code_meta["labels_raw"] = labels[:5]
        if healthchecks:
            code_meta["has_healthcheck"] = True
        if envs:
            code_meta["env_vars"] = envs[:10]
        if workdirs:
            code_meta["workdir"] = workdirs[-1]

        return [ParsedChunk(
            content_text=chunk_text[:4000],
            language="dockerfile",
            code_metadata=code_meta,
        )]


class ShellScriptParser:
    def parse(self, file_path: str, content: str) -> List[ParsedChunk]:
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

        base_header_parts = [
            f"# Shell script",
            f"# File: {file_path}",
        ]
        if component:
            base_header_parts.append(f"# Component: {component}")
        if shebang:
            base_header_parts.append(f"# Shebang: {shebang}")
        if sourced_files:
            base_header_parts.append(f"# Sources: {', '.join(sourced_files[:5])}")

        base_meta = {
            "component_name": component,
        }
        if shebang:
            base_meta["shebang"] = shebang
        if sourced_files:
            base_meta["sourced_files"] = sourced_files
        if export_vars:
            base_meta["exported_variables"] = export_vars[:15]
        if set_flags:
            base_meta["set_flags"] = set_flags

        if len(function_matches) < 2:
            header = "\n".join(base_header_parts)
            func_name = None
            if function_matches:
                func_name = function_matches[0].group(1) or function_matches[0].group(2)
                base_header_parts.append(f"# Function: {func_name}")
                header = "\n".join(base_header_parts)

            chunk_text = f"{header}\n\n{content}"
            meta = {
                **base_meta,
                "code_entity_type": "function" if func_name else "script",
                "code_entity_name": func_name if func_name else file_path.split("/")[-1],
            }
            if function_matches:
                meta["functions"] = [function_matches[0].group(1) or function_matches[0].group(2)]

            return [ParsedChunk(
                content_text=chunk_text[:4000],
                language="shell",
                code_metadata=meta,
            )]

        chunks = []
        all_func_names = [m.group(1) or m.group(2) for m in function_matches]

        preamble_end = function_matches[0].start()
        preamble = content[:preamble_end].strip()
        if preamble:
            header = "\n".join(base_header_parts)
            chunk_text = f"{header}\n# Section: preamble\n\n{preamble}"
            chunks.append(ParsedChunk(
                content_text=chunk_text[:4000],
                language="shell",
                code_metadata={
                    **base_meta,
                    "code_entity_type": "script",
                    "code_entity_name": file_path.split("/")[-1],
                    "functions": all_func_names,
                }
            ))

        for i, match in enumerate(function_matches):
            func_name = match.group(1) or match.group(2)
            start = match.start()
            end = function_matches[i + 1].start() if i + 1 < len(function_matches) else len(content)
            func_body = content[start:end].strip()

            func_header_parts = base_header_parts + [f"# Function: {func_name}"]
            header = "\n".join(func_header_parts)
            chunk_text = f"{header}\n\n{func_body}"

            func_meta = {
                **base_meta,
                "code_entity_type": "function",
                "code_entity_name": func_name,
            }

            func_sources = re.findall(r'(?:source|\\.)\s+["\']?([^\s"\']+)', func_body)
            func_exports = re.findall(r'export\s+(\w+)=', func_body)
            if func_sources:
                func_meta["sourced_files"] = func_sources
            if func_exports:
                func_meta["exported_variables"] = func_exports

            chunks.append(ParsedChunk(
                content_text=chunk_text[:4000],
                language="shell",
                code_metadata=func_meta,
            ))

        return chunks if chunks else [ParsedChunk(
            content_text=content[:4000],
            language="shell",
            code_metadata={
                **base_meta,
                "code_entity_type": "script",
                "code_entity_name": file_path.split("/")[-1],
            }
        )]


def parse_file(file_path: str, content: str) -> List[ParsedChunk]:
    filename = file_path.split("/")[-1].lower()

    kustomization_names = {"kustomization.yaml", "kustomization.yml", "kustomization"}
    if filename in kustomization_names:
        return KustomizationParser().parse(file_path, content)

    if filename.startswith("dockerfile") or filename == "dockerfile":
        return DockerfileParser().parse(file_path, content)

    if filename.endswith(".sh") or filename.endswith(".bash"):
        return ShellScriptParser().parse(file_path, content)

    if filename.endswith(".yaml") or filename.endswith(".yml"):
        return YAMLManifestParser().parse(file_path, content)

    return [ParsedChunk(
        content_text=content[:4000],
        language="",
        code_metadata={
            "code_entity_type": "",
            "code_entity_name": "",
            "component_name": _infer_component_name(file_path),
        }
    )]
