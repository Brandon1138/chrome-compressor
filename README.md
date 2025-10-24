# reducelang: Shannon Redundancy Estimation for Natural Languages

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-CI%20coming%20soon-lightgrey.svg)](https://example.org)

## Overview
`reducelang` aims to reproduce and extend Claude Shannon's classic estimate (~68%) of English redundancy by estimating the entropy rate of natural language. We implement information-theoretic models (n-grams, PPM) and compression-based baselines (e.g., Huffman) to compute redundancy R = 1 - H/log₂M, where M is the alphabet size. We start with English and Romanian alphabets and provide both CLI and Jupyter interfaces. The project emphasizes academic rigor and reproducibility.

References include Shannon (1951), Huffman (1952), and Cover & Thomas. We verify results via block bootstrap confidence intervals, corpus sensitivity, and ablations.

## Features
- Alphabet definitions for English (M=27) and Romanian (M=32) with variants (space, punctuation, diacritics)
- Entropy estimation: unigram, n-gram (Kneser-Ney), PPM (PPM-C/D), Huffman baseline
- Redundancy computation: R = 1 - H / log₂M
- Statistical validation: block bootstrap CIs, corpus sensitivity, ablations
- Reproducible: deterministic seeds, `uv.lock`, SHA256-verified corpora
- Dual interface: CLI (`reducelang`) and Jupyter notebooks
- Academic output: LaTeX/PDF proofs, Markdown/MathJax site

## Installation
1. Ensure Python 3.12 and [`uv`](https://astral.sh) are installed.
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Alternative: `pip install uv`
2. Clone the repo and enter it:
   ```bash
   git clone <repo-url> && cd chrome-compressor
   ```
3. Setup the environment:
   ```bash
   make setup
   # or
   uv sync && uv pip install -e .
   ```
4. Verify the installation (CLI scaffold to be added in later phases):
   ```bash
   uv run reducelang --version
   ```

## Quick Start
- Launch Jupyter for exploration:
  ```bash
  make notebook
  ```
- (Future) Prepare corpora:
  ```bash
  reducelang prep --lang en --corpus wikipedia
  ```
- (Future) Estimate entropy:
  ```bash
  reducelang estimate --model ppm --order 8 --lang en
  ```
- (Future) Full English pipeline:
  ```bash
  make english
  ```
- (Future) Build paper:
  ```bash
  make paper
  ```

## Project Structure
```
chrome-compressor/
├── reducelang/          # Main library
│   ├── __init__.py
│   ├── alphabet.py      # Alphabet definitions, log₂M
│   ├── config.py        # Configuration constants
│   ├── corpus/          # (Phase 2) Corpus download, preprocessing
│   ├── models/          # (Phase 3-4) Entropy models (n-gram, PPM, Huffman)
│   ├── validation/      # (Phase 6) Bootstrap, sensitivity analysis
│   └── proofs/          # (Phase 7) LaTeX/Markdown proof generation
├── notebooks/           # Jupyter exploration
├── tests/               # Pytest unit tests
├── paper/               # LaTeX source and figures
├── data/                # Corpora (gitignored)
├── results/             # JSON/CSV outputs (gitignored)
├── pyproject.toml       # Project metadata, dependencies
├── uv.lock              # Deterministic lockfile
├── Makefile             # Automation
└── README.md
```

## Development
- Run tests: `make test`
- Lint: `make lint`
- Format: `make format`
- Type check: `make typecheck`
- Regenerate lockfile: `make lock`

## Reproducibility
- Python pinned to 3.12 via `.python-version` and `pyproject.toml`
- Dependencies locked in `uv.lock`
- Random seeds fixed in `reducelang/config.py`
- Corpora verified via SHA256 (placeholders in this phase)
- Experiments runnable via `make` targets

## References
1. Shannon, C. E. (1951). Prediction and entropy of printed English.
2. Huffman, D. A. (1952). A method for the construction of minimum-redundancy codes.
3. Cover, T. M., & Thomas, J. A. Elements of Information Theory.

## License
MIT License. See [`LICENSE`](LICENSE).

## Acknowledgments
Inspired by Claude Shannon, David Huffman, and modern compression research.

## Contact
Maintainer: <author@example.com> (placeholder)
