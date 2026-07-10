MAX_TOKENS ?= 128
THRESHOLD  ?= 0

PYTHON       := bench/.venv/bin/python3
MARKDOWNFMT  := go run github.com/Kunde21/markdownfmt/v2@v2.1.0

.PHONY: help build test fmt-md \
        bench-setup bench-sweep \
        clean

help:
	@echo "kukri — structure-aware markdown chunker"
	@echo ""
	@echo "Build & test:"
	@echo "  make build            compile the chunker binary"
	@echo "  make test             run Go tests"
	@echo "  make fmt-md           format and align tables in all README files"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench-setup      create Python venv and install bench deps"
	@echo "  make bench-sweep      sweep thresholds and report Hit@k"
	@echo ""
	@echo "Variables:"
	@echo "  MAX_TOKENS=128        chunk size limit (words)"
	@echo "  THRESHOLD=0           scorer similarity threshold (0 = size-only)"
	@echo "  CORPUS=path.jsonl     corpus for benchmark"
	@echo "  GOLD=path.jsonl       gold QA pairs for benchmark"
	@echo "  EMBED_MODEL=...       HuggingFace embedding model for Hit@k"

# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

fmt-md:
	$(MARKDOWNFMT) -w README.md ../README.md

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

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

# Usage: make bench-sweep CORPUS=corpus.jsonl GOLD=bench/gold.jsonl EMBED_MODEL=BAAI/bge-small-en-v1.5
bench-sweep: build bench/.venv/bin/python3
ifndef CORPUS
	$(error set CORPUS=path/to/corpus.jsonl)
endif
ifndef GOLD
	$(error set GOLD=path/to/gold.jsonl)
endif
	KUKRI_BIN=./kukri $(PYTHON) bench/sweep_threshold.py \
		--gold        $(GOLD) \
		--corpus      $(CORPUS) \
		--max-tokens  $(MAX_TOKENS) \
		--thresholds  $(THRESHOLD) \
		$(if $(EMBED_MODEL),--embed-model $(EMBED_MODEL),)

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean:
	rm -f kukri /tmp/kukri_chunks.json bench/gold.jsonl
	find . -name '*.test' -delete
