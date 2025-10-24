"""Arithmetic and range coding for codelength verification.

Used to validate that measured cross-entropy matches actual compression
codelength under an arithmetic coder using model-provided probabilities.

Public API:
- ArithmeticCoder
- encode_with_model
- compute_codelength
- verify_codelength
"""

from __future__ import annotations

from reducelang.coding.arithmetic import (
    ArithmeticCoder,
    encode_with_model,
    compute_codelength,
    verify_codelength,
)

__all__ = [
    "ArithmeticCoder",
    "encode_with_model",
    "compute_codelength",
    "verify_codelength",
]


