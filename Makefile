# Makefile for reducelang: Shannon redundancy estimation.
# Run 'make setup' to initialize the project.

.PHONY: setup lock test lint format typecheck clean notebook english romanian paper site prep-en prep-ro prep-all clean-data estimate-en-unigram estimate-en-ngram estimate-ro-unigram estimate-ro-ngram estimate-all clean-results estimate-en-ppm estimate-ro-ppm estimate-ppm-all verify-ppm huffman-en huffman-ro huffman-all compare-en compare-ro compare-all validate-en validate-ro validate-all bootstrap-en bootstrap-ro sensitivity-en sensitivity-ro report-all

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

# english: Run full English pipeline (prep, estimate, validate, report)
english: prep-en estimate-en-unigram estimate-en-ngram estimate-en-ppm huffman-en validate-en
	uv run reducelang report --lang en --format both --out paper/

# romanian: Run full Romanian pipeline (prep, estimate, validate, report)
romanian: prep-ro estimate-ro-unigram estimate-ro-ngram estimate-ro-ppm huffman-ro validate-ro
	uv run reducelang report --lang ro --format both --out paper/

# paper: Generate LaTeX/PDF proofs for both languages
paper:
	uv run reducelang report --lang both --format pdf --out paper/
	@echo "PDF proofs generated in paper/"
	@echo "Compile with: cd paper && pdflatex en_redundancy.tex && pdflatex ro_redundancy.tex"

# site: Generate HTML static site for both languages
site:
	uv run reducelang report --lang both --format html --out site/
	@echo "Static site generated in site/"
	@echo "View with: python -m http.server --directory site/"

# report-all: Generate both PDF and HTML for both languages
report-all:
	uv run reducelang report --lang both --format both --out paper/

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
estimate-all: estimate-en-unigram estimate-en-ngram estimate-ro-unigram estimate-ro-ngram estimate-ppm-all huffman-all

# estimate-en-ppm: Estimate entropy with PPM models (depths 3,5,8) on English text8
estimate-en-ppm:
	uv run reducelang estimate --model ppm --order 3 --lang en --corpus text8
	uv run reducelang estimate --model ppm --order 5 --lang en --corpus text8
	uv run reducelang estimate --model ppm --order 8 --lang en --corpus text8 --verify-codelength

# estimate-ro-ppm: Estimate entropy with PPM models on Romanian OPUS
estimate-ro-ppm:
	uv run reducelang estimate --model ppm --order 3 --lang ro --corpus opus
	uv run reducelang estimate --model ppm --order 5 --lang ro --corpus opus
	uv run reducelang estimate --model ppm --order 8 --lang ro --corpus opus --verify-codelength

# estimate-ppm-all: Run all PPM estimation targets
estimate-ppm-all: estimate-en-ppm estimate-ro-ppm

# verify-ppm: Run PPM estimation with codelength verification on both languages
verify-ppm:
	uv run reducelang estimate --model ppm --order 8 --lang en --corpus text8 --verify-codelength
	uv run reducelang estimate --model ppm --order 8 --lang ro --corpus opus --verify-codelength

# huffman-en: Train Huffman model on English text8 and generate comparison table
huffman-en:
	uv run reducelang huffman --lang en --corpus text8 --compare

# huffman-ro: Train Huffman model on Romanian OPUS and generate comparison table
huffman-ro:
	uv run reducelang huffman --lang ro --corpus opus --compare

# huffman-all: Run Huffman on both languages
huffman-all: huffman-en huffman-ro

# compare-en: Generate comparison table for English (Markdown format)
compare-en:
	uv run reducelang huffman --lang en --corpus text8 --compare --compare-format markdown > results/entropy/en/text8/2025-10-01/comparison.md

# compare-ro: Generate comparison table for Romanian (Markdown format)
compare-ro:
	uv run reducelang huffman --lang ro --corpus opus --compare --compare-format markdown > results/entropy/ro/opus/2025-10-01/comparison.md

# compare-all: Generate comparison tables for both languages
compare-all: compare-en compare-ro

# validate-en: Run PPM with bootstrap CI and sensitivity analysis for English
validate-en:
	uv run reducelang estimate --model ppm --order 8 --lang en --corpus text8 --bootstrap --sensitivity

# validate-ro: Run PPM with bootstrap CI and sensitivity analysis for Romanian
validate-ro:
	uv run reducelang estimate --model ppm --order 8 --lang ro --corpus opus --bootstrap --sensitivity --ablations no_diacritics,no_space

# validate-all: Run validation on both languages
validate-all: validate-en validate-ro

# bootstrap-en: Compute bootstrap CIs for English
bootstrap-en:
	uv run reducelang estimate --model ppm --order 8 --lang en --corpus text8 --bootstrap

# bootstrap-ro: Compute bootstrap CIs for Romanian
bootstrap-ro:
	uv run reducelang estimate --model ppm --order 8 --lang ro --corpus opus --bootstrap

# sensitivity-en: Run sensitivity only for English
sensitivity-en:
	uv run reducelang estimate --model ppm --order 8 --lang en --corpus text8 --sensitivity

# sensitivity-ro: Run sensitivity only for Romanian
sensitivity-ro:
	uv run reducelang estimate --model ppm --order 8 --lang ro --corpus opus --sensitivity --ablations no_diacritics,no_space

