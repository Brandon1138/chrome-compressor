# Makefile for reducelang: Shannon redundancy estimation.
# Run 'make setup' to initialize the project.

.PHONY: setup lock test lint format typecheck clean notebook english romanian paper site

setup:
	uv sync
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
	uv run python -c "import os,shutil,glob; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','.mypy_cache','.ruff_cache'] if os.path.isdir(p)]; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(egg, ignore_errors=True) for egg in glob.glob('*.egg-info')]; print('Cleaned artifacts.')"

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


