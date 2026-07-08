MODEL_DIR  ?= ./models
MODEL_NAME ?= KnightsAnalytics/all-MiniLM-L6-v2
MODEL_PATH ?= $(MODEL_DIR)/KnightsAnalytics_all-MiniLM-L6-v2

MAX_TOKENS ?= 512
THRESHOLD  ?= 0.7

PYTHON := bench/.venv/bin/python3

.PHONY: help download-model build test \
        bench-setup bench-coverage bench-metrics \
        clean

help:
	@echo "kukri — hybrid markdown-aware + semantic chunker"
	@echo ""
	@echo "Setup:"
	@echo "  make download-model   download all-MiniLM-L6-v2 ONNX weights"
	@echo "  make bench-setup      create Python venv and install bench deps"
	@echo ""
	@echo "Build & test:"
	@echo "  make build            compile the chunker binary"
	@echo "  make test             run Go tests"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench-coverage   Answer Coverage Rate benchmark"
	@echo "  make bench-metrics    intrinsic metrics on a single markdown file"
	@echo ""
	@echo "Variables:"
	@echo "  MAX_TOKENS=512        chunk size limit"
	@echo "  THRESHOLD=0.7         semantic similarity threshold"
	@echo "  CORPUS=path.jsonl     corpus for coverage benchmark"
	@echo "  ANTHROPIC_API_KEY=... required for gold generation"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

download-model:
	go run ./cmd/download -model $(MODEL_NAME) -dest $(MODEL_DIR)

bench-setup:
	python3 -m venv bench/.venv
	bench/.venv/bin/pip install --quiet --upgrade pip
	bench/.venv/bin/pip install --quiet -r bench/requirements.txt
	@echo "venv ready at bench/.venv"

# ---------------------------------------------------------------------------
# Build & test
# ---------------------------------------------------------------------------

build:
	go build -o kukri ./cmd/chunker

test:
	go test ./...

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# Usage: make bench-coverage CORPUS=corpus.jsonl ANTHROPIC_API_KEY=sk-...
bench-coverage: bench/.venv/bin/python3
ifndef CORPUS
	$(error set CORPUS=path/to/corpus.jsonl)
endif
ifndef ANTHROPIC_API_KEY
	$(error set ANTHROPIC_API_KEY)
endif
	$(PYTHON) bench/generate_gold.py \
		--input         $(CORPUS) \
		--output        bench/gold.jsonl \
		--pairs-per-doc 5
	$(PYTHON) bench/coverage_bench.py \
		--gold      bench/gold.jsonl \
		--corpus    $(CORPUS) \
		--max-tokens $(MAX_TOKENS) \
		--threshold  $(THRESHOLD)

# Usage: make bench-metrics INPUT=document.md
bench-metrics: bench/.venv/bin/python3
ifndef INPUT
	$(error set INPUT=path/to/document.md)
endif
	go run ./cmd/chunker \
		-input       $(INPUT) \
		-model       $(MODEL_PATH) \
		-output      /tmp/kukri_chunks.json \
		-max-tokens  $(MAX_TOKENS) \
		-threshold   $(THRESHOLD)
	$(PYTHON) bench/metrics.py \
		--model all-MiniLM-L6-v2 \
		"kukri:/tmp/kukri_chunks.json"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean:
	rm -f kukri /tmp/kukri_chunks.json bench/gold.jsonl
	find . -name '*.test' -delete
