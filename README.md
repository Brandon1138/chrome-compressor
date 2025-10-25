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
- Bootstrap confidence intervals: block bootstrap (default 1000 resamples) for 95% CIs on bits/char and redundancy
- Sensitivity analysis: ablation studies for alphabet variants (with/without space, punctuation, diacritics)
- Statistical rigor: quantify uncertainty and test robustness of entropy estimates
- Reproducible: deterministic seeds, `uv.lock`, SHA256-verified corpora
- Dual interface: CLI (`reducelang`) and Jupyter notebooks
- Academic output: LaTeX/PDF proofs, Markdown/MathJax site
- Proof generation: LaTeX/PDF and Markdown/HTML with theorem blocks, citations, and figures
- Template-driven: Jinja2 templates for academic-quality proofs following Shannon's framework
- Automated compilation: pdflatex/latexmk for PDF, Quarto for static sites
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
#### Reporting and Proof Generation

- Generate PDF proof (English) to `paper/` with figures in `paper/figs`:
  ```bash
  reducelang report --lang en --format pdf --out paper/ --figures-dir paper/figs
  ```
- Choose LaTeX style: `article` (default), `acm`, or `arxiv`:
  ```bash
  # ACM acmart style (non-ACM, sigconf)
  reducelang report --lang en --format pdf --out paper/ --figures-dir paper/figs --latex-style acm

  # arXiv-friendly article style (geometry + natbib)
  reducelang report --lang en --format pdf --out paper/ --figures-dir paper/figs --latex-style arxiv
  ```
  If a requested style template is missing, the renderer falls back to the default article template.
- Generate HTML site (English). Figures will be referenced relatively; if needed they are copied to `site/figs`:
  ```bash
  reducelang report --lang en --format html --out site/ --figures-dir paper/figs
  ```
- Generate all outputs for both languages:
  ```bash
  reducelang report --lang both --format both
  ```
- Make targets:
  ```bash
  make english
  make romanian
  make paper
  make site
  make report-all
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
│   │   ├── estimate.py      # Entropy estimation (now with --bootstrap, --sensitivity)
│   │   └── huffman.py       # Huffman coding and comparison
│   └── utils.py         # Shared utilities (SHA256, file ops)
│   ├── models/          # Entropy estimation models
│   │   ├── base.py          # Abstract LanguageModel base class
│   │   ├── unigram.py       # Empirical character frequencies
│   │   └── ngram.py         # Kneser-Ney n-gram models (NLTK)
│   │   └── ppm.py           # Prediction by Partial Matching
│   ├── coding/              # Arithmetic coding for verification
│   │   └── arithmetic.py    # Range coder, codelength computation
│   ├── validation/      # Statistical validation
│   │   ├── bootstrap.py     # Block bootstrap for confidence intervals
│   │   └── sensitivity.py   # Ablation studies for alphabet variants
│   └── proofs/          # Proof generation
│       ├── generator.py     # Load results, build context
│       ├── renderer.py      # Render templates, compile outputs
│       └── templates/       # Jinja2 templates (LaTeX, Markdown)
│           ├── latex/       # ACM/arXiv style templates
│           └── markdown/    # Quarto-compatible templates
│   ├── report.py        # Orchestration: load → render → compile
│   ├── commands/
│   │   └── report.py    # CLI command for proof generation
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
 - `jinja2`: Template rendering
 - `matplotlib`: Figure generation
 - `seaborn`: Optional, for prettier plots
 - External tools (optional): `pdflatex`/`latexmk` for PDF, `quarto` for HTML

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

## Bootstrap Confidence Intervals

Purpose: Quantify uncertainty in entropy estimates due to finite test set size. We use block bootstrap (1–2k character blocks) to preserve temporal dependencies, with 1000 resamples by default and 95% percentile CIs.

Usage:

```bash
reducelang estimate --model ppm --order 8 --lang en --corpus text8 --bootstrap
```

Results JSON includes `bootstrap` with `mean_bpc`, `ci_lower_bpc`, `ci_upper_bpc`, and `redundancy_ci`. Typical CI width for ~20k test chars: ±0.02–0.05 bpc.

## Sensitivity Analysis

Purpose: Test robustness of entropy estimates to preprocessing choices (alphabet definition). For each ablation, we create an alphabet variant, re-normalize train/test, retrain, and report deltas.

Ablations:
- `no_space`: Remove space from alphabet
- `no_diacritics`: Remove diacritics (Romanian)
- `with_punctuation`: Include punctuation symbols

Usage:

```bash
reducelang estimate --model ppm --order 8 --lang ro --corpus opus --sensitivity --ablations no_diacritics,no_space
```

Interpretation:
- Positive ΔH means worse compression; negative ΔR means less redundancy
- For Romanian, removing diacritics typically increases entropy by ~0.1–0.3 bpc
- Removing space often increases entropy by ~0.5–1.0 bpc

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

# 7. Validation via Makefile
make validate-all

# 8. Generate PDF proof
reducelang report --lang en --format pdf

# 9. Generate HTML site
reducelang report --lang en --format html --out site/
```
## Proof Generation

Purpose: Generate publication-ready proofs from experimental results, with theorem blocks, citations, and figures.

Templates: LaTeX (ACM/arXiv style) and Markdown (Quarto-compatible with MathJax).

Content: Entropy rate definition, redundancy definition, source coding theorem, Shannon's guessing game, finite-order bounds, methodology, results tables, sensitivity analysis.

Figures: Automatically generated comparison plots (entropy vs. order, redundancy comparison).

Compilation: Requires `pdflatex` or `latexmk` for PDF, `quarto` for HTML (graceful fallback if not installed).

Usage: `reducelang report --lang en --format pdf --out paper/ --figures-dir paper/figs`

### Output Structure

```
paper/
├── en_redundancy.tex        # LaTeX source (English)
├── en_redundancy.pdf        # Compiled PDF (English)
├── ro_redundancy.tex        # LaTeX source (Romanian)
├── ro_redundancy.pdf        # Compiled PDF (Romanian)
├── references.bib           # BibTeX references
└── figs/                    # Generated figures
    ├── entropy_vs_order_en.pdf
    ├── entropy_vs_order_en.png
    ├── redundancy_comparison_en.pdf
    └── ...

site/
├── en_redundancy.html       # HTML proof (English)
├── ro_redundancy.html       # HTML proof (Romanian)
└── _site/                   # Quarto build artifacts
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
4. Cleary, J. G., & Witten, I. H. (1984). Data compression using adaptive coding and partial string matching.
5. Moffat, A. (1990). Implementing the PPM data compression scheme.
6. Teahan, W. J., & Cleary, J. G. (1997). The entropy of English using PPM-based models.
7. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
8. Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012). An Empirical Investigation of Statistical Significance in NLP.

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
