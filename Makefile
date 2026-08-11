.DEFAULT_GOAL := help
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
ROWS ?= 250000

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- setup ---------------------------------------------------------------
.PHONY: venv
venv: ## Create the virtualenv and install everything
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[serving,export,explain,genai,dev]"

.PHONY: install-streaming
install-streaming: ## Add Kafka + Feast (needed only for the streaming path)
	$(PIP) install -e ".[streaming]"

# --- pipeline ------------------------------------------------------------
.PHONY: train
train: ## Train the ensemble and register a version (ROWS=250000)
	$(PY) scripts/train.py --rows $(ROWS) --promote

.PHONY: export
export: ## Export the current model to ONNX and quantize
	$(PY) scripts/export_onnx.py

.PHONY: index
index: ## Build the RAG case index
	$(PY) scripts/build_case_index.py

.PHONY: governance
governance: ## Fairness, drift and performance pack
	$(PY) scripts/governance_report.py

.PHONY: all
all: train export index governance ## Full pipeline from scratch

# --- run -----------------------------------------------------------------
.PHONY: serve
serve: ## Run the scoring API on :8080
	.venv/bin/uvicorn fraudplat.serving.app:app --host 0.0.0.0 --port 8080 --reload

.PHONY: stream
stream: ## Run the Kafka feature writer
	$(PY) scripts/run_stream_writer.py

.PHONY: demo
demo: ## End-to-end walkthrough, no infrastructure required
	$(PY) scripts/demo.py

.PHONY: bench
bench: ## Latency benchmark against the 50ms budget
	$(PY) scripts/benchmark_inference.py

# --- quality -------------------------------------------------------------
.PHONY: test
test: ## Run the test suite
	$(PY) -m pytest

.PHONY: cov
cov: ## Test suite with coverage
	$(PY) -m pytest --cov=fraudplat --cov-report=term-missing

.PHONY: lint
lint: ## Ruff check + format check
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts

.PHONY: fmt
fmt: ## Apply formatting and autofixes
	.venv/bin/ruff check --fix src tests scripts
	.venv/bin/ruff format src tests scripts

# --- containers ----------------------------------------------------------
.PHONY: docker
docker: ## Build both images
	docker build -f docker/Dockerfile.serving  -t fraudplat-serving:local .
	docker build -f docker/Dockerfile.streaming -t fraudplat-streaming:local .

.PHONY: pipeline
pipeline: ## Compile the Kubeflow pipeline
	$(PY) pipelines/kubeflow/fraud_pipeline.py

.PHONY: clean
clean: ## Remove generated artifacts
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
