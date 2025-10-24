"""Block bootstrap for confidence intervals on entropy estimates.

Implements block bootstrap resampling suitable for time-series/text data,
computing confidence intervals for bits per character (bpc) and mapping those
to redundancy confidence intervals.

References
----------
- Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
- Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012). An Empirical
  Investigation of Statistical Significance in NLP. EMNLP.

Examples
--------
>>> from reducelang.validation import block_bootstrap, compute_bootstrap_ci
>>> from reducelang.models import PPMModel
>>> from reducelang.alphabet import ENGLISH_ALPHABET
>>> model = PPMModel(ENGLISH_ALPHABET, depth=3)
>>> model.fit("hello world ")
>>> res = block_bootstrap("hello world ", model, 2000, 10, 0.95, 42)
>>> _ = compute_bootstrap_ci(res.get("mean_bpc", 1.0), ENGLISH_ALPHABET.log2_size, res)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reducelang.models.base import LanguageModel
from reducelang.config import Config
from reducelang.redundancy import compute_redundancy


def _split_into_blocks(text: str, block_size: int) -> list[str]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if not text:
        raise ValueError("text must be non-empty for bootstrap resampling")
    n_full = len(text) // block_size
    blocks: list[str] = [text[i * block_size : (i + 1) * block_size] for i in range(n_full)]
    remainder = len(text) % block_size
    if remainder > 0:
        blocks.append(text[n_full * block_size :])
    return blocks


def block_bootstrap(
    text: str,
    model: LanguageModel,
    block_size: int | None = None,
    n_resamples: int | None = None,
    confidence_level: float | None = None,
    seed: int | None = None,
) -> dict[str, float | int]:
    """Compute block bootstrap CI for bits per character using a trained model.

    Parameters
    ----------
    text:
        Test text to resample (should be the held-out test set).
    model:
        Trained language model implementing ``evaluate(text) -> float``.
    block_size:
        Contiguous block size for resampling. Defaults to Config.BOOTSTRAP_BLOCK_SIZE.
    n_resamples:
        Number of bootstrap resamples. Defaults to Config.BOOTSTRAP_N_RESAMPLES.
    confidence_level:
        CI level (e.g., 0.95). Defaults to Config.BOOTSTRAP_CONFIDENCE_LEVEL.
    seed:
        Random seed for reproducibility. Defaults to Config.RANDOM_SEED.

    Returns
    -------
    dict
        Dictionary containing mean/std/percentile CI bounds and config metadata.
    """

    bs = int(block_size if block_size is not None else Config.BOOTSTRAP_BLOCK_SIZE)
    n = int(n_resamples if n_resamples is not None else Config.BOOTSTRAP_N_RESAMPLES)
    cl = float(confidence_level if confidence_level is not None else Config.BOOTSTRAP_CONFIDENCE_LEVEL)
    rng_seed = int(seed if seed is not None else Config.RANDOM_SEED)

    if not (0.0 < cl < 1.0):
        raise ValueError("confidence_level must be in (0,1)")
    if n <= 0:
        raise ValueError("n_resamples must be > 0")

    rng = np.random.default_rng(rng_seed)
    blocks = _split_into_blocks(text, bs)
    if not blocks:
        raise ValueError("No blocks available for bootstrap; check text and block_size")

    n_blocks = len(blocks)
    bpc_samples: list[float] = []
    for _ in range(n):
        choices = rng.choice(blocks, size=n_blocks, replace=True)
        resampled_text = "".join(list(choices))
        bpc = float(model.evaluate(resampled_text))
        bpc_samples.append(bpc)

    samples_np = np.array(bpc_samples, dtype=float)
    mean_bpc = float(np.mean(samples_np))
    std_bpc = float(np.std(samples_np, ddof=1)) if len(samples_np) > 1 else 0.0
    alpha = 1.0 - cl
    lower = float(np.percentile(samples_np, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(samples_np, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean_bpc": mean_bpc,
        "std_bpc": std_bpc,
        "ci_lower_bpc": lower,
        "ci_upper_bpc": upper,
        "n_resamples": n,
        "block_size": bs,
        "confidence_level": cl,
    }


def compute_bootstrap_ci(
    *,
    bits_per_char: float,
    log2_alphabet_size: float,
    bootstrap_results: dict[str, Any],
) -> dict[str, float]:
    """Map bpc bootstrap CI to redundancy CI and include point estimate.

    Redundancy is R = 1 - H / log₂M, which is a decreasing function of H.
    Thus, bounds invert: lower_R uses upper_H, and upper_R uses lower_H.
    """

    h = float(bits_per_char)
    log2m = float(log2_alphabet_size)
    r_point = compute_redundancy(h, log2m)

    ci_lo_h = float(bootstrap_results.get("ci_lower_bpc", h))
    ci_hi_h = float(bootstrap_results.get("ci_upper_bpc", h))

    # Invert bounds
    r_lo = 1.0 - (ci_hi_h / log2m) if log2m > 0.0 else 0.0
    r_hi = 1.0 - (ci_lo_h / log2m) if log2m > 0.0 else 0.0

    # Clamp
    def _clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    return {
        "redundancy": _clamp01(float(r_point)),
        "ci_lower_redundancy": _clamp01(float(r_lo)),
        "ci_upper_redundancy": _clamp01(float(r_hi)),
    }


