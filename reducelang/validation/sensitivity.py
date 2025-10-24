"""Sensitivity analysis and ablation studies for alphabet variants.

This module trains a baseline model and then re-trains/evaluates variants
with modified alphabets (e.g., removing space or diacritics) to quantify the
effect on bits/char and redundancy.
"""

from __future__ import annotations

from typing import Any

from reducelang.models.base import LanguageModel
from reducelang.alphabet import Alphabet
from reducelang.redundancy import compute_redundancy


def run_sensitivity_analysis(
    *,
    model_class: type[LanguageModel],
    base_alphabet: Alphabet,
    train_text: str,
    test_text: str,
    model_kwargs: dict,
    variants: list[dict],
) -> dict[str, Any]:
    """Run sensitivity analysis by evaluating alphabet variants.

    Returns a dictionary with baseline metrics and per-variant deltas.
    """

    # Train baseline
    base_train = base_alphabet.normalize(train_text)
    base_test = base_alphabet.normalize(test_text)
    baseline_model = model_class(base_alphabet, **(model_kwargs or {}))
    baseline_model.fit(base_train)
    baseline_bpc = float(baseline_model.evaluate(base_test))
    baseline_r = compute_redundancy(baseline_bpc, base_alphabet.log2_size)

    out: dict[str, Any] = {
        "baseline": {
            "alphabet_name": base_alphabet.name,
            "alphabet_size": base_alphabet.size,
            "log2_alphabet_size": base_alphabet.log2_size,
            "bits_per_char": baseline_bpc,
            "redundancy": baseline_r,
        },
        "variants": [],
    }

    for spec in variants:
        name = str(spec.get("name", "variant"))
        kwargs = dict(spec.get("alphabet_kwargs", {}))
        variant_alphabet = base_alphabet.variant(**kwargs)
        v_train = variant_alphabet.normalize(train_text)
        v_test = variant_alphabet.normalize(test_text)
        v_model = model_class(variant_alphabet, **(model_kwargs or {}))
        v_model.fit(v_train)
        v_bpc = float(v_model.evaluate(v_test))
        v_r = compute_redundancy(v_bpc, variant_alphabet.log2_size)

        delta_bpc = v_bpc - baseline_bpc
        delta_r = v_r - baseline_r
        rel_change = (delta_bpc / baseline_bpc) if baseline_bpc > 0.0 else 0.0

        out["variants"].append(
            {
                "name": name,
                "alphabet_name": variant_alphabet.name,
                "alphabet_size": variant_alphabet.size,
                "log2_alphabet_size": variant_alphabet.log2_size,
                "bits_per_char": v_bpc,
                "redundancy": v_r,
                "delta_bpc": delta_bpc,
                "delta_redundancy": delta_r,
                "relative_change_bpc": rel_change,
            }
        )

    return out


def run_ablation_study(
    *,
    model_class: type[LanguageModel],
    alphabet: Alphabet,
    train_text: str,
    test_text: str,
    model_kwargs: dict,
    ablations: list[str],
) -> dict[str, Any]:
    """Run predefined ablation variants by name and return sensitivity results."""

    name_to_spec = {
        "no_space": {"name": "no_space", "alphabet_kwargs": {"include_space": False}},
        "no_diacritics": {"name": "no_diacritics", "alphabet_kwargs": {"include_diacritics": False}},
        "with_punctuation": {"name": "with_punctuation", "alphabet_kwargs": {"include_punctuation": True}},
    }

    variants: list[dict] = []
    for ab in ablations or []:
        spec = name_to_spec.get(str(ab).strip())
        if spec is not None:
            variants.append(spec)

    return run_sensitivity_analysis(
        model_class=model_class,
        base_alphabet=alphabet,
        train_text=train_text,
        test_text=test_text,
        model_kwargs=model_kwargs,
        variants=variants,
    )


def format_sensitivity_results(results: dict, output_format: str = "table") -> str | dict:
    """Format sensitivity results as ASCII table, Markdown, CSV, or JSON."""

    baseline = results.get("baseline", {})
    variants = results.get("variants", [])

    headers = [
        "Variant",
        "M",
        "log2M",
        "H (bpc)",
        "Redundancy (%)",
        "ΔH (bpc)",
        "ΔR (pp)",
        "Rel. change (%)",
    ]

    rows: list[list[str]] = []
    h_base = float(baseline.get("bits_per_char", 0.0))
    r_base = float(baseline.get("redundancy", 0.0))
    for v in variants:
        rows.append(
            [
                str(v.get("name", "")),
                str(int(v.get("alphabet_size", 0))),
                f"{float(v.get(\"log2_alphabet_size\", 0.0)):.3f}",
                f"{float(v.get(\"bits_per_char\", 0.0)):.4f}",
                f"{float(v.get(\"redundancy\", 0.0))*100:.2f}",
                f"{float(v.get(\"delta_bpc\", 0.0)):+.4f}",
                f"{float(v.get(\"delta_redundancy\", 0.0))*100:+.2f}",
                (
                    f"{((float(v.get(\"bits_per_char\", 0.0)) - h_base) / h_base * 100.0):+.2f}"
                    if h_base > 0.0
                    else "+0.00"
                ),
            ]
        )

    if output_format == "json":
        return {
            "baseline": baseline,
            "variants": variants,
        }

    if output_format == "csv":
        lines = [",".join(["variant","M","log2M","H_bpc","redundancy","delta_H","delta_R","relative_change"])]
        for r in rows:
            lines.append(",".join(r))
        return "\n".join(lines)

    if output_format == "markdown":
        out_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for r in rows:
            out_lines.append("| " + " | ".join(r) + " |")
        return "\n".join(out_lines)

    # Default ASCII table
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    def _fmt_row(cols: list[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))
    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt_row(headers), sep]
    for r in rows:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


