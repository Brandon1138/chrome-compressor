# reducelang: Shannon Redundancy Estimation for Natural Languages

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-CI%20coming%20soon-lightgrey.svg)](https://example.org)

## Overview
`reducelang` aims to reproduce and extend Claude Shannon's classic estimate (~68%) of English redundancy by estimating the entropy rate of natural language. We implement information-theoretic models (n-grams, PPM) and compression-based baselines (e.g., Huffman) to compute redundancy R = 1 - H/log₂M, where M is the alphabet size. We start with English and Romanian alphabets and provide both CLI and Jupyter interfaces. The project emphasizes academic rigor and reproducibility.

References include Shannon (1951), Huffman (1952), and Cover & Thomas. We verify results via block bootstrap confidence intervals, corpus sensitivity, and ablations.

## Features
- Alphabet definitions for English (M=27) and Romanian (M=32) with variants (space, punctuation, diacritics)
- Entropy estimation: unigram (empirical frequencies), n-gram (Kneser-Ney smoothing), PPM (adaptive contexts), and Huffman (optimal single-character codes)
- Arithmetic coding for codelength verification (validates entropy estimates)
- Context depths up to 12 for capturing long-range dependencies (PPM)
- Redundancy computation: R = 1 - H / log₂M, comparison tables across models
- Statistical validation: block bootstrap CIs, corpus sensitivity, ablations
- Reproducible: deterministic seeds, `uv.lock`, SHA256-verified corpora
- Dual interface: CLI (`reducelang`) and Jupyter notebooks
- Academic output: LaTeX/PDF proofs, Markdown/MathJax site
- Automated corpus download: Wikipedia (EN/RO), Brown, Gutenberg, text8, OPUS, Europarl
- Format handlers: Wikipedia XML, NLTK, zip, gzip, tar.gz, parallel corpora
- Data cards: JSON metadata for reproducibility (source, license, hashes, dates)

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
4. Verify the installation:
   ```bash
   uv run reducelang --version
   ```

## Quick Start
- Launch Jupyter for exploration:
  ```bash
  make notebook
  ```
- Prepare corpora:
  ```bash
  reducelang prep --lang en --corpus wikipedia
  reducelang prep --lang ro --corpus all
  make prep-en
  ```

### Huffman coding and comparison tables

Train Huffman and generate a comparison table:

```bash
reducelang huffman --lang en --corpus text8
reducelang huffman --lang en --corpus text8 --compare
make huffman-en
```

Generate comparison tables only (Markdown):

```bash
make compare-all
```

### User-provided corpora

You can supply your own plain-text file and bypass the registry and downloader:

```bash
reducelang prep --lang en --corpus corpus --corpus-path /path/to/my.txt
```

This uses an inline CorpusSpec with `format` set to `plain_text` and processes
the file using the `PlainTextExtractor` (UTF-8 read, normalized with the chosen
alphabet, and written to the processed path). The data card will record the
alphabet name and computed hashes.

### Wikipedia snapshots

For Wikipedia corpora, you can specify a snapshot:

```bash
reducelang prep --lang en --corpus wikipedia --snapshot 2025-10-01
```

Snapshot values accepted: `latest`, `YYYY-MM-DD`, or `YYYYMMDD`. The downloader
constructs URLs like `https://dumps.wikimedia.org/enwiki/{yyyymmdd}/enwiki-{yyyymmdd}-pages-articles.xml.bz2`.

### Gutenberg corpus

You can select the NLTK Gutenberg corpus explicitly:

```bash
reducelang prep --lang en --corpus gutenberg
```

This requires `nltk` and the `gutenberg` dataset (`nltk.download('gutenberg')`).
- Estimate entropy:
  ```bash
  reducelang estimate --model unigram --lang en --corpus text8
  reducelang estimate --model ngram --order 5 --lang ro --corpus opus
  reducelang estimate --model ppm --order 8 --lang en --corpus text8
  reducelang estimate --model ppm --order 5 --lang ro --corpus opus --verify-codelength
  make estimate-en-ngram
  make estimate-en-ppm
  make verify-ppm
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
│   ├── huffman.py       # Huffman coding (single-character baseline)
│   ├── redundancy.py    # Redundancy computation and comparison tables
│   ├── corpus/          # Corpus download, extraction, preprocessing
│   │   ├── registry.py      # Corpus specifications (URLs, formats, licenses)
│   │   ├── downloader.py    # HTTP downloads with progress, resume, SHA256
│   │   ├── extractors.py    # Format-specific text extraction
│   │   ├── preprocessor.py  # Normalization pipeline
│   │   └── datacard.py      # JSON metadata generation
│   ├── commands/        # CLI subcommands
│   │   ├── prep.py          # Corpus preparation command
│   │   ├── estimate.py      # Entropy estimation (unigram, n-gram, PPM)
│   │   └── huffman.py       # Huffman coding and comparison
│   └── utils.py         # Shared utilities (SHA256, file ops)
│   ├── models/          # Entropy estimation models
│   │   ├── base.py          # Abstract LanguageModel base class
│   │   ├── unigram.py       # Empirical character frequencies
│   │   └── ngram.py         # Kneser-Ney n-gram models (NLTK)
│   │   └── ppm.py           # Prediction by Partial Matching
│   ├── coding/              # Arithmetic coding for verification
│   │   └── arithmetic.py    # Range coder, codelength computation
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
- Corpora verified via SHA256 (first run computes and records)
- Experiments runnable via `make` targets

## Dependencies
- `nltk`: N-gram language models with Kneser-Ney smoothing

## Entropy Estimation
- Unigram model: Empirical character frequencies with Laplace smoothing; computes H₁ = -Σ p(c) log₂ p(c).
- N-gram model: Kneser-Ney interpolated smoothing via NLTK; captures character dependencies (orders 2-8 recommended, enforced).
- Train/test split: 80/20 split (configurable via `--test-split`) for cross-entropy measurement.
- Results: JSON in `results/entropy/{lang}/{corpus}/{snapshot}/` with bits/char, metadata, and model paths. Models serialized to `results/models/` (pickle).

### Huffman model
Optimal single-character prefix coding. Builds a binary tree from character frequencies and computes average code length. Achieves the entropy lower bound for single-character coding (H₁), but cannot exploit dependencies. Serves as a baseline to demonstrate that context models (n-gram, PPM) capture much more redundancy.

### Results Directory Layout
```
results/
├── entropy/
│   ├── en/
│   │   └── text8/
│   │       └── 2025-10-01/
│   │           ├── unigram_order1.json
│   │           ├── ngram_order3.json
│   │           ├── ngram_order5.json
│   │           ├── ppm_order5.json
│   │           ├── ppm_order8.json
│   │           ├── huffman_order1.json
│   │           ├── comparison.md
│   │           └── comparison.json
└── models/
    ├── en/
    │   └── text8/
    │       └── 2025-10-01/
    │           ├── unigram_order1.pkl
    │           └── ngram_order3.pkl
```

## Examples
```bash
# 1. Download and preprocess corpus
reducelang prep --lang en --corpus text8

# 2. Estimate entropy with unigram
reducelang estimate --model unigram --lang en --corpus text8

# 3. Estimate entropy with n-grams (orders 2-8)
for order in 2 3 5 8; do
  reducelang estimate --model ngram --order $order --lang en --corpus text8
done

# 4. Estimate entropy with PPM and verify codelength
reducelang estimate --model ppm --order 8 --lang en --corpus text8 --verify-codelength

# 5. Train Huffman and generate comparison table
reducelang huffman --lang en --corpus text8 --compare

# 6. View results
cat results/entropy/en/text8/2025-10-01/ppm_order8.json
```

## Redundancy and Comparison Tables

- Definition: R = 1 - H / log₂M, where H is entropy estimate (bits/char) and M is alphabet size.
- Interpretation: Fraction of predictable information that can be compressed away. For English (M=27), log₂M ≈ 4.755 bpc.
- Single-character redundancy: Huffman and unigram capture ~10–15% redundancy from frequency imbalances.
- Dependency redundancy: N-gram and PPM capture ~60–75% redundancy by exploiting spelling, grammar, and context.
- Usage: `reducelang huffman --lang en --corpus text8 --compare --compare-format markdown`

### Expected Results (illustrative)
- English (M=27, log₂M ≈ 4.755): Huffman ≈ 4.2 bpc (~12%), N-gram(5) ≈ 2.5 bpc (~47%), PPM(8) ≈ 1.5 bpc (~68%).
- Romanian (M=32, log₂M = 5.000): Huffman ≈ 4.3 bpc (~14%), N-gram(5) ≈ 2.7 bpc (~46%), PPM(8) ≈ 1.6 bpc (~68%).

## References
1. Shannon, C. E. (1951). Prediction and entropy of printed English.
2. Huffman, D. A. (1952). A method for the construction of minimum-redundancy codes.
3. Cover, T. M., & Thomas, J. A. Elements of Information Theory.
4. Cleary, J. G., & Witten, I. H. (1984). Data compression using adaptive coding and partial string matching. IEEE Transactions on Communications, 32(4), 396–402.
5. Moffat, A. (1990). Implementing the PPM data compression scheme. IEEE Transactions on Communications, 38(11), 1917–1921.
6. Teahan, W. J., & Cleary, J. G. (1997). The entropy of English using PPM-based models. Data Compression Conference.

## License
MIT License. See [`LICENSE`](LICENSE).

Note: `wikiextractor` is AGPL-3.0 licensed. If you distribute modified versions of
`reducelang` that include `wikiextractor`, you must comply with AGPL-3.0 terms. The
Wikipedia extractor attempts to call `wikiextractor` and falls back to `python -m wikiextractor`.

## Acknowledgments
Inspired by Claude Shannon, David Huffman, and modern compression research.

## Contact
Maintainer: <author@example.com> (placeholder)
