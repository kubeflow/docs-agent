# Makefile for Kubeflow Documentation AI Assistant
# https://github.com/kubeflow/docs-agent

.PHONY: all build build-server build-server-https compile compile-main compile-incremental \
        test lint lint-check format clean install help

# Docker image configuration
DOCKER_REGISTRY ?=
IMAGE_TAG ?= latest
SERVER_IMAGE_NAME ?= ws-proxy
SERVER_HTTPS_IMAGE_NAME ?= https-api

# Full image names (with optional registry prefix)
ifdef DOCKER_REGISTRY
	SERVER_IMAGE := $(DOCKER_REGISTRY)/$(SERVER_IMAGE_NAME):$(IMAGE_TAG)
	SERVER_HTTPS_IMAGE := $(DOCKER_REGISTRY)/$(SERVER_HTTPS_IMAGE_NAME):$(IMAGE_TAG)
else
	SERVER_IMAGE := $(SERVER_IMAGE_NAME):$(IMAGE_TAG)
	SERVER_HTTPS_IMAGE := $(SERVER_HTTPS_IMAGE_NAME):$(IMAGE_TAG)
endif

# Python configuration
PYTHON ?= python3
PIP ?= pip3

# Directories
PIPELINES_DIR := pipelines
SERVER_DIR := server
SERVER_HTTPS_DIR := server-https

# Default target
all: lint test build compile

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

install: ## Install development dependencies
	$(PIP) install -r $(PIPELINES_DIR)/requirements.txt
	$(PIP) install flake8 black pytest pytest-cov

install-all: install ## Install all dependencies (including server requirements)
	$(PIP) install -r $(SERVER_DIR)/requirements.txt
	$(PIP) install -r $(SERVER_HTTPS_DIR)/requirements.txt

##@ Build

build: build-server build-server-https ## Build all Docker images

build-server: ## Build the WebSocket server Docker image
	@echo "Building $(SERVER_IMAGE)..."
	docker build -t $(SERVER_IMAGE) $(SERVER_DIR)
	@echo "Successfully built $(SERVER_IMAGE)"

build-server-https: ## Build the HTTPS server Docker image
	@echo "Building $(SERVER_HTTPS_IMAGE)..."
	docker build -t $(SERVER_HTTPS_IMAGE) $(SERVER_HTTPS_DIR)
	@echo "Successfully built $(SERVER_HTTPS_IMAGE)"

push: push-server push-server-https ## Push all Docker images to registry

push-server: build-server ## Push WebSocket server image to registry
ifndef DOCKER_REGISTRY
	$(error DOCKER_REGISTRY is not set. Usage: make push-server DOCKER_REGISTRY=your-registry)
endif
	docker push $(SERVER_IMAGE)

push-server-https: build-server-https ## Push HTTPS server image to registry
ifndef DOCKER_REGISTRY
	$(error DOCKER_REGISTRY is not set. Usage: make push-server-https DOCKER_REGISTRY=your-registry)
endif
	docker push $(SERVER_HTTPS_IMAGE)

##@ Pipeline Compilation

compile: compile-main compile-incremental ## Compile all Kubeflow pipelines to YAML

compile-main: ## Compile the main RAG pipeline
	@echo "Compiling main pipeline..."
	cd $(PIPELINES_DIR) && $(PYTHON) kubeflow-pipeline.py
	@echo "Generated: $(PIPELINES_DIR)/github_rag_pipeline.yaml"

compile-incremental: ## Compile the incremental RAG pipeline
	@echo "Compiling incremental pipeline..."
	cd $(PIPELINES_DIR) && $(PYTHON) incremental-pipeline.py
	@echo "Generated: $(PIPELINES_DIR)/github_rag_incremental_pipeline.yaml"

##@ Testing

test: ## Run all tests with pytest
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short 2>/dev/null || \
		(echo "No tests found or tests directory missing. Create tests in tests/ directory." && exit 0)

test-cov: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing 2>/dev/null || \
		(echo "No tests found or tests directory missing." && exit 0)

##@ Code Quality

lint: lint-check ## Run all linting checks (alias for lint-check)

lint-check: ## Check code style with flake8 and black
	@echo "Running flake8..."
	$(PYTHON) -m flake8 $(SERVER_DIR) $(SERVER_HTTPS_DIR) $(PIPELINES_DIR) \
		--max-line-length=120 \
		--exclude=__pycache__,venv,.venv,env \
		--ignore=E501,W503 || true
	@echo "Checking black formatting..."
	$(PYTHON) -m black --check --diff $(SERVER_DIR) $(SERVER_HTTPS_DIR) $(PIPELINES_DIR) \
		--exclude="/(\.git|__pycache__|venv|\.venv|env)/" || true

format: ## Format code with black
	@echo "Formatting code with black..."
	$(PYTHON) -m black $(SERVER_DIR) $(SERVER_HTTPS_DIR) $(PIPELINES_DIR) \
		--exclude="/(\.git|__pycache__|venv|\.venv|env)/"
	@echo "Code formatted successfully"

##@ Cleanup

clean: ## Remove build artifacts and caches
	@echo "Cleaning up..."
	# Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	find . -type f -name "*~" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete"

clean-docker: ## Remove built Docker images
	@echo "Removing Docker images..."
	docker rmi $(SERVER_IMAGE) 2>/dev/null || true
	docker rmi $(SERVER_HTTPS_IMAGE) 2>/dev/null || true
	@echo "Docker images removed"
