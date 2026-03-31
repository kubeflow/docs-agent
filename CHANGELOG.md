# Changelog

All notable changes to the Kubeflow docs-agent are 
documented here.

## [Unreleased]

### Planned (GSoC 2026)
- Agentic architecture using LangGraph or Kagent
- Multi-source querying (docs + GitHub Issues + 
  Platform Architecture)
- KServe Scale-to-Zero deployment
- Terraform/Manifests for OCI deployment
- Evaluation framework for retrieval accuracy

---

## [0.3.0] — 2026 (Recent)

### Added
- Kagent POC with Feast as the feature store
- `kagent-feast-mcp/` directory with MCP integration
- Updated GSoC 2026 Agentic RAG Architecture Proposal
  (`gsoc2026_agentic_rag.md`)

### Fixed
- Replaced hardcoded namespace in manifests, 
  pipelines, and deployment files

---

## [0.2.0] — 2025

### Added
- HTTPS API server (`server-https/app.py`) with 
  FastAPI framework
- WebSocket API server (`server/app.py`)
- Kubeflow Pipeline for automated ETL
  (`pipelines/kubeflow-pipeline.py`)
- Milvus vector database integration
- KServe inference service configuration
- Kubernetes manifests for deployment
- CONTRIBUTING.md contributor guide

### Architecture
- Dual API: WebSocket for real-time chat, 
  HTTPS for RESTful integration
- RAG pipeline: fetch → chunk → embed → store
- Tool calling with automatic documentation lookup
- Citation tracking and deduplication

---

## [0.1.0] — 2025 (Initial Release)

### Added
- Initial project structure
- Basic RAG implementation
- Documentation assets (architecture diagrams)
- Apache 2.0 License

---

## How to Update This Changelog

When making changes, add an entry under `[Unreleased]`:

### Added
- New feature description

### Fixed  
- Bug fix description

### Changed
- Modified behavior description

### Removed
- Deprecated feature removed

Follow [Keep a Changelog](https://keepachangelog.com/) 
format and [Semantic Versioning](https://semver.org/).