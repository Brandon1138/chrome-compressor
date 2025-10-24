"""Statistical validation: bootstrap confidence intervals and sensitivity analysis.

This package provides utilities for:
- Block bootstrap confidence intervals for bits/char and redundancy
- Sensitivity analysis via alphabet ablations (space, punctuation, diacritics)

Public API:
- block_bootstrap, compute_bootstrap_ci
- run_sensitivity_analysis, run_ablation_study, format_sensitivity_results
"""

from __future__ import annotations

from reducelang.validation.bootstrap import block_bootstrap, compute_bootstrap_ci
from reducelang.validation.sensitivity import (
    run_sensitivity_analysis,
    run_ablation_study,
    format_sensitivity_results,
)

__all__ = [
    "block_bootstrap",
    "compute_bootstrap_ci",
    "run_sensitivity_analysis",
    "run_ablation_study",
    "format_sensitivity_results",
]


