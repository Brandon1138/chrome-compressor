"""Proof generation: load results, render LaTeX/Markdown templates with theorem
blocks, compile to PDF/HTML.

Public API
----------
- ProofGenerator, load_results_for_language
- ProofRenderer, compile_latex_to_pdf, build_static_site
"""

from __future__ import annotations

from typing import Any

# Re-export public API for convenience
from reducelang.proofs.generator import ProofGenerator, load_results_for_language  # noqa: F401
from reducelang.proofs.renderer import (  # noqa: F401
    ProofRenderer,
    compile_latex_to_pdf,
    build_static_site,
)

__all__ = [
    "ProofGenerator",
    "load_results_for_language",
    "ProofRenderer",
    "compile_latex_to_pdf",
    "build_static_site",
]


