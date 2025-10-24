"""Redundancy computation and comparison table utilities.

This module loads model result JSON files and computes redundancy
R = 1 - H / log₂M, where H is bits/char and M is the alphabet size.

It can also format comparison tables across models (unigram, n-gram, PPM,
Huffman) in several output formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import warnings

from reducelang.config import Config


def compute_redundancy(bits_per_char: float, log2_alphabet_size: float) -> float:
    """Return redundancy R = 1 - H / log₂M.

    If inputs are invalid (e.g., H > log₂M), returns 0.0 and emits a warning.
    """

    if log2_alphabet_size <= 0.0:
        return 0.0
    if bits_per_char < 0.0:
        bits_per_char = 0.0
    if bits_per_char > log2_alphabet_size + 1e-9:
        warnings.warn(
            f"bits_per_char ({bits_per_char}) exceeds log2_alphabet_size ({log2_alphabet_size}); clipping to 0 redundancy.",
            RuntimeWarning,
        )
        return 0.0
    r = 1.0 - (bits_per_char / log2_alphabet_size)
    if r < 0.0:
        r = 0.0
    if r > 1.0:
        r = 1.0
    return r


def _results_dir_for(lang: str, corpus: str, snapshot: str) -> Path:
    return Config.RESULTS_DIR / "entropy" / lang / corpus / snapshot


def load_model_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all result JSON files from ``results_dir``.

    Ignores non-JSON files and malformed entries.
    """

    if not results_dir.exists() or not results_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Attach filename for later reference
            data["__file_name__"] = p.name
            out.append(data)
        except Exception:
            continue
    return out


def _normalize_result_entry(entry: dict[str, Any]) -> dict[str, Any]:
    model = str(entry.get("model_choice") or entry.get("model_type") or "?")
    order = int(entry.get("order") or entry.get("depth") or 0)
    h = float(entry.get("bits_per_char", 0.0))
    log2m = float(entry.get("log2_alphabet_size", 0.0))
    m = int(entry.get("alphabet_size", 0))
    redundancy = compute_redundancy(h, log2m) if log2m > 0 else 0.0
    compression_ratio = (log2m / h) if (h > 0 and log2m > 0) else 0.0
    return {
        "model": model,
        "order": order,
        "bits_per_char": h,
        "log2_alphabet_size": log2m,
        "alphabet_size": m,
        "redundancy": redundancy,
        "compression_ratio": compression_ratio,
        "__file_name__": entry.get("__file_name__", ""),
    }


def analyze_redundancy_gain(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute redundancy gains across model families.

    Returns keys: huffman_redundancy, ngram_redundancy, ppm_redundancy,
    huffman_to_ngram_gain, huffman_to_ppm_gain, ngram_to_ppm_gain
    """

    huffman_vals = [r["redundancy"] for r in results if r.get("model", "").startswith("huffman") or r.get("model") == "huffman"]
    ngram_vals = [r["redundancy"] for r in results if str(r.get("model", "")).startswith("ngram")]
    ppm_vals = [r["redundancy"] for r in results if str(r.get("model", "")) == "ppm"]

    huffman_r = max(huffman_vals) if huffman_vals else 0.0
    ngram_r = max(ngram_vals) if ngram_vals else 0.0
    ppm_r = max(ppm_vals) if ppm_vals else 0.0

    return {
        "huffman_redundancy": float(huffman_r),
        "ngram_redundancy": float(ngram_r),
        "ppm_redundancy": float(ppm_r),
        "huffman_to_ngram_gain": float(ngram_r - huffman_r),
        "huffman_to_ppm_gain": float(ppm_r - huffman_r),
        "ngram_to_ppm_gain": float(ppm_r - ngram_r),
    }


def _sorted_results(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    # Sort by bits_per_char descending (higher to lower) to show progression
    return sorted(entries, key=lambda r: float(r.get("bits_per_char", 0.0)), reverse=True)


def format_table_ascii(results: list[dict[str, Any]]) -> str:
    headers = ["Model", "Order", "H (bpc)", "log₂M", "Redundancy", "Comp. Ratio"]
    rows: list[list[str]] = []
    for r in results:
        model_name = {
            "unigram": "Unigram",
            "huffman": "Huffman",
            "ppm": f"PPM (depth={r.get('order', 0)})",
        }.get(str(r.get("model")), str(r.get("model")))
        if str(r.get("model", "")).startswith("ngram"):
            model_name = f"N-gram (order={r.get('order', 0)})"
        rows.append(
            [
                model_name,
                str(r.get("order", "")),
                f"{float(r.get('bits_per_char', 0.0)):.4f}",
                f"{float(r.get('log2_alphabet_size', 0.0)):.4f}",
                f"{float(r.get('redundancy', 0.0))*100:.2f}%",
                f"{float(r.get('compression_ratio', 0.0)):.3f}",
            ]
        )

    # Column widths
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def format_table_markdown(results: list[dict[str, Any]]) -> str:
    headers = ["Model", "Order", "H (bpc)", "log₂M", "Redundancy", "Comp. Ratio"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in results:
        model_name = {
            "unigram": "Unigram",
            "huffman": "Huffman",
            "ppm": f"PPM (depth={r.get('order', 0)})",
        }.get(str(r.get("model")), str(r.get("model")))
        if str(r.get("model", "")).startswith("ngram"):
            model_name = f"N-gram (order={r.get('order', 0)})"
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    str(r.get("order", "")),
                    f"{float(r.get('bits_per_char', 0.0)):.4f}",
                    f"{float(r.get('log2_alphabet_size', 0.0)):.4f}",
                    f"{float(r.get('redundancy', 0.0))*100:.2f}%",
                    f"{float(r.get('compression_ratio', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_table_csv(results: list[dict[str, Any]]) -> str:
    headers = ["model", "order", "bits_per_char", "log2_alphabet_size", "redundancy", "compression_ratio"]
    out_lines = [",".join(headers)]
    for r in results:
        out_lines.append(
            ",".join(
                [
                    str(r.get("model", "")),
                    str(r.get("order", "")),
                    f"{float(r.get('bits_per_char', 0.0)):.6f}",
                    f"{float(r.get('log2_alphabet_size', 0.0)):.6f}",
                    f"{float(r.get('redundancy', 0.0)):.6f}",
                    f"{float(r.get('compression_ratio', 0.0)):.6f}",
                ]
            )
        )
    return "\n".join(out_lines)


def save_comparison_table(table: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")


def generate_comparison_table(
    lang: str,
    corpus: str,
    snapshot: str,
    output_format: str = "table",
) -> str | dict:
    """Load results and return a formatted comparison table.

    ``output_format`` may be one of {"table", "json", "csv", "markdown"}.
    """

    results_dir = _results_dir_for(lang, corpus, snapshot)
    entries_raw = load_model_results(results_dir)
    normalized = [_normalize_result_entry(e) for e in entries_raw]
    normalized = [e for e in normalized if e.get("bits_per_char", 0.0) > 0 and e.get("log2_alphabet_size", 0.0) > 0]
    normalized = _sorted_results(normalized)

    if output_format == "json":
        return {
            "lang": lang,
            "corpus": corpus,
            "snapshot": snapshot,
            "results": normalized,
            "gains": analyze_redundancy_gain(normalized),
        }
    if output_format == "csv":
        return format_table_csv(normalized)
    if output_format == "markdown":
        return format_table_markdown(normalized)
    # Default ASCII table
    return format_table_ascii(normalized)


__all__ = [
    "compute_redundancy",
    "generate_comparison_table",
    "load_model_results",
    "format_table_ascii",
    "format_table_markdown",
    "format_table_csv",
    "analyze_redundancy_gain",
]


