# Makefile for reducelang: Shannon redundancy estimation.
# Run 'make setup' to initialize the project.

.PHONY: setup lock test lint format typecheck clean notebook english romanian paper site prep-en prep-ro prep-all clean-data estimate-en-unigram estimate-en-ngram estimate-ro-unigram estimate-ro-ngram estimate-all clean-results

setup:
	uv sync --locked
	uv pip install -e .
	uv python pin 3.12

lock:
	uv lock --upgrade

test:
	uv run pytest tests/

lint:
	uv run ruff check reducelang/ tests/

format:
	uv run ruff format reducelang/ tests/

typecheck:
	uv run mypy reducelang/

clean:
	uv run python -c "import os,shutil,glob; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','.mypy_cache','.ruff_cache'] if os.path.isdir(p)]; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(egg, ignore_errors=True) for egg in glob.glob('*.egg-info')]; print('Cleaned artifacts.')" && $(MAKE) clean-data
	$(MAKE) clean-results

notebook:
	uv run jupyter notebook notebooks/

# Future targets (placeholders)
english:
	@echo "English pipeline (to be implemented in Phase 2)"

romanian:
	@echo "Romanian pipeline (to be implemented in Phase 2)"

paper:
	@echo "Paper build (to be implemented in Phase 7)"

site:
	@echo "Documentation site (to be implemented later)"

# Corpus preparation targets
# prep-en: Download and preprocess all English corpora
prep-en:
	uv run reducelang prep --lang en --corpus all

# prep-ro: Download and preprocess all Romanian corpora
prep-ro:
	uv run reducelang prep --lang ro --corpus all

# prep-all: Download and preprocess all corpora
prep-all: prep-en prep-ro

# clean-data: Remove downloaded and processed corpora
clean-data:
	rm -rf data/corpora/

# clean-results: Remove generated results and models
clean-results:
	rm -rf results/

# estimate-en-unigram: Estimate entropy with unigram model on English text8
estimate-en-unigram:
	uv run reducelang estimate --model unigram --lang en --corpus text8

# estimate-en-ngram: Estimate entropy with n-gram models on English text8
estimate-en-ngram:
	uv run reducelang estimate --model ngram --order 2 --lang en --corpus text8
	uv run reducelang estimate --model ngram --order 3 --lang en --corpus text8
	uv run reducelang estimate --model ngram --order 5 --lang en --corpus text8
	uv run reducelang estimate --model ngram --order 8 --lang en --corpus text8

# estimate-ro-unigram: Estimate entropy with unigram model on Romanian OPUS
estimate-ro-unigram:
	uv run reducelang estimate --model unigram --lang ro --corpus opus

# estimate-ro-ngram: Estimate entropy with n-gram models on Romanian OPUS
estimate-ro-ngram:
	uv run reducelang estimate --model ngram --order 2 --lang ro --corpus opus
	uv run reducelang estimate --model ngram --order 3 --lang ro --corpus opus
	uv run reducelang estimate --model ngram --order 5 --lang ro --corpus opus
	uv run reducelang estimate --model ngram --order 8 --lang ro --corpus opus

# estimate-all: Run all entropy estimation targets
estimate-all: estimate-en-unigram estimate-en-ngram estimate-ro-unigram estimate-ro-ngram


