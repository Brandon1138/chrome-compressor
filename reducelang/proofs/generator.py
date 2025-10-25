"""Context generator for proof rendering.

Loads model result JSON files, extracts key metrics (entropy, redundancy,
bootstrap CIs, sensitivity), and constructs a context dictionary suitable for
injection into LaTeX/Markdown Jinja2 templates.

Example
-------
>>> from reducelang.proofs import load_results_for_language
>>> ctx = load_results_for_language("en", "text8", "2025-10-01")
>>> round(ctx["ppm_bpc"], 3)
1.500  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging

from reducelang.config import Config
from reducelang.alphabet import ENGLISH_ALPHABET, ROMANIAN_ALPHABET
from reducelang.redundancy import (
    load_model_results,
    analyze_redundancy_gain,
    compute_redundancy,
)
from reducelang.corpus.datacard import load_datacard


_LOGGER = logging.getLogger(__name__)


def _ensure_results_dir(lang: str, corpus: str, snapshot: str) -> Path:
    results_dir = Config.RESULTS_DIR / "entropy" / lang / corpus / snapshot
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}. Run estimation first."
        )
    return results_dir


@dataclass
class ProofGenerator:
    """Generate a template context for a given language/corpus/snapshot.

    Parameters
    ----------
    lang:
        Language code ("en" or "ro").
    corpus:
        Corpus name (e.g., "text8", "opus").
    snapshot:
        Snapshot identifier (e.g., "2025-10-01").
    """

    lang: str
    corpus: str
    snapshot: str

    def __post_init__(self) -> None:
        self._results_dir: Path = Config.RESULTS_DIR / "entropy" / self.lang / self.corpus / self.snapshot
        self._alphabet = ENGLISH_ALPHABET if self.lang == "en" else ROMANIAN_ALPHABET

    def _find_result_by_model(self, results: list[dict[str, Any]], model: str, order: int | None = None) -> dict[str, Any] | None:
        model = model.lower()
        for r in results:
            m = str(r.get("model_choice") or r.get("model_type") or r.get("model") or "").lower()
            ord_val = r.get("order") or r.get("depth")
            if m == model and (order is None or int(ord_val or 0) == int(order)):
                return r
        return None

    def _extract_bootstrap_ci(self, result: dict[str, Any]) -> dict[str, float] | None:
        boot = result.get("bootstrap")
        if not isinstance(boot, dict):
            return None
        try:
            ci_lower = float(boot.get("ci_lower_bpc"))
            ci_upper = float(boot.get("ci_upper_bpc"))
            mean_bpc = float(boot.get("mean_bpc", result.get("bits_per_char", 0.0)))
            ci_width = float(abs(ci_upper - ci_lower) / 2.0)
            log2m = float(result.get("log2_alphabet_size", self._alphabet.log2_size))
            red_lower = compute_redundancy(ci_upper, log2m)  # worst-case redundancy at upper H
            red_upper = compute_redundancy(ci_lower, log2m)  # best-case redundancy at lower H
            red_width = float(abs(red_upper - red_lower) / 2.0)
            return {
                "ci_lower_bpc": ci_lower,
                "ci_upper_bpc": ci_upper,
                "ci_width": ci_width,
                "redundancy_ci_lower": red_lower,
                "redundancy_ci_upper": red_upper,
                "redundancy_ci_width": red_width,
                "mean_bpc": mean_bpc,
            }
        except Exception:
            return None

    def _extract_sensitivity(self, result: dict[str, Any]) -> dict[str, Any] | None:
        sens = result.get("sensitivity")
        if not isinstance(sens, dict):
            return None
        variants = sens.get("variants") or []
        baseline = sens.get("baseline") or {}
        return {
            "baseline": baseline,
            "variants": variants,
        }

    def _normalize_all(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for e in entries:
            model = str(e.get("model_choice") or e.get("model_type") or e.get("model") or "?")
            order = int(e.get("order") or e.get("depth") or 0)
            h = float(e.get("bits_per_char", 0.0))
            log2m = float(e.get("log2_alphabet_size", self._alphabet.log2_size))
            m = int(e.get("alphabet_size", self._alphabet.size))
            redundancy = compute_redundancy(h, log2m) if log2m > 0 else 0.0
            compression_ratio = (log2m / h) if (h > 0 and log2m > 0) else 0.0
            normalized.append(
                {
                    "model": model,
                    "order": order,
                    "bits_per_char": h,
                    "log2_alphabet_size": log2m,
                    "alphabet_size": m,
                    "redundancy": redundancy,
                    "compression_ratio": compression_ratio,
                }
            )
        return normalized

    def _best_ppm(self, entries: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, float]]]:
        ppm_entries = [e for e in entries if str(e.get("model") or e.get("model_choice") or e.get("model_type")) in {"ppm", "PPM"}]
        if not ppm_entries:
            return None, None
        # Choose by lowest bits_per_char, break ties by highest order/depth
        best = sorted(ppm_entries, key=lambda r: (float(r.get("bits_per_char", 1e9)), -int(r.get("order") or r.get("depth") or 0)))[0]
        return best, self._extract_bootstrap_ci(best)

    def _best_ngram(self, entries: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        ngram_entries = [e for e in entries if str(e.get("model") or e.get("model_choice") or e.get("model_type", "")).startswith("ngram")]
        if not ngram_entries:
            return None
        return sorted(ngram_entries, key=lambda r: (float(r.get("bits_per_char", 1e9)), -int(r.get("order") or 0)))[0]

    def generate_context(self) -> dict[str, Any]:
        """Load results and build a context dictionary for templates.

        Raises FileNotFoundError if the results directory is missing, or
        ValueError if no results are present.
        """

        results_dir = _ensure_results_dir(self.lang, self.corpus, self.snapshot)
        entries_raw = load_model_results(results_dir)
        if not entries_raw:
            raise ValueError(
                f"No results found in {results_dir}. Run 'reducelang estimate' and 'reducelang huffman' first."
            )

        normalized_all = self._normalize_all(entries_raw)

        # Identify key results
        huffman = None
        for r in entries_raw:
            m = str(r.get("model_choice") or r.get("model_type") or r.get("model") or "").lower()
            if m == "huffman":
                huffman = r
                break

        ppm_best, ppm_ci = self._best_ppm(entries_raw)
        ngram_best = self._best_ngram(entries_raw)

        # Gains
        gains = analyze_redundancy_gain(self._normalize_all(entries_raw))

        # Datacard for corpus metadata
        datacard_path = Config.DEFAULT_CORPUS_DIR / self.lang / self.snapshot / "processed" / f"{self.corpus}_datacard.json"
        corpus_size = None
        if datacard_path.exists():
            try:
                datacard = load_datacard(datacard_path)
                corpus_size = datacard.get("char_count")
            except Exception:
                pass

        language_name = "English" if self.lang == "en" else "Romanian"

        context: dict[str, Any] = {
            "language_name": language_name,
            "lang": self.lang,
            "corpus_name": self.corpus,
            "snapshot": self.snapshot,
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "alphabet_size": self._alphabet.size,
            "log2_alphabet_size": self._alphabet.log2_size,
            "alphabet_name": self._alphabet.name,
            "corpus_size": corpus_size,
            "comparison_table_data": normalized_all,
            # defaults for key metrics
            "ppm_bpc": float(ppm_best.get("bits_per_char", 0.0)) if ppm_best else 0.0,
            "ppm_order": int(ppm_best.get("order") or ppm_best.get("depth") or 0) if ppm_best else 0,
            "ppm_redundancy": compute_redundancy(
                float(ppm_best.get("bits_per_char", 0.0)) if ppm_best else 0.0,
                self._alphabet.log2_size,
            ),
            "ppm_ci_width": float(ppm_ci.get("ci_width", 0.0)) if ppm_ci else 0.0,
            "redundancy_ci_width": float(ppm_ci.get("redundancy_ci_width", 0.0)) if ppm_ci else 0.0,
            "huffman_bpc": float(huffman.get("bits_per_char", 0.0)) if huffman else 0.0,
            "huffman_redundancy": compute_redundancy(
                float(huffman.get("bits_per_char", 0.0)) if huffman else 0.0,
                self._alphabet.log2_size,
            ),
            "ngram_best_bpc": float(ngram_best.get("bits_per_char", 0.0)) if ngram_best else 0.0,
            "ngram_best_order": int(ngram_best.get("order") or 0) if ngram_best else 0,
            "ppm_to_huffman_gain": float(gains.get("huffman_to_ppm_gain", 0.0) * 100.0),
            "sensitivity_table_data": [],
            "ablations_list": "",
            "diacritics_delta": 0.0,
            "space_delta": 0.0,
            "has_bootstrap": bool(ppm_ci is not None),
            "has_sensitivity": False,
        }

        # Sensitivity extraction (if present on any relevant entry)
        sens_result = None
        for r in entries_raw:
            if r.get("sensitivity"):
                sens_result = r
                break
        if sens_result is not None:
            sens = self._extract_sensitivity(sens_result)
            if sens is not None:
                variants = sens.get("variants", [])
                context["sensitivity_table_data"] = []
                ablations: list[str] = []
                baseline_bpc = float(sens.get("baseline", {}).get("bits_per_char", 0.0))
                for v in variants:
                    name = str(v.get("name") or v.get("variant") or "variant")
                    ablations.append(name)
                    vbpc = float(v.get("bits_per_char", 0.0))
                    delta_h = float(vbpc - baseline_bpc)
                    v_log2m = float(v.get("log2_alphabet_size", self._alphabet.log2_size))
                    v_red = compute_redundancy(vbpc, v_log2m)
                    context["sensitivity_table_data"].append(
                        {
                            "variant": name,
                            "alphabet_size": int(v.get("alphabet_size", self._alphabet.size)),
                            "log2_alphabet_size": v_log2m,
                            "bits_per_char": vbpc,
                            "redundancy": v_red,
                            "delta_h": delta_h,
                            "delta_r": v_red - compute_redundancy(baseline_bpc, v_log2m),
                        }
                    )
                context["ablations_list"] = ", ".join(ablations)
                context["has_sensitivity"] = True
                # Specific deltas if present
                for v in variants:
                    name = str(v.get("name") or v.get("variant") or "")
                    if name == "no_diacritics":
                        context["diacritics_delta"] = float(v.get("bits_per_char", 0.0) - baseline_bpc)
                    if name == "no_space":
                        context["space_delta"] = float(v.get("bits_per_char", 0.0) - baseline_bpc)

        return context


def load_results_for_language(lang: str, corpus: str, snapshot: str) -> dict[str, Any]:
    """Convenience function to generate a proof context for templates.

    Example
    -------
    >>> from reducelang.proofs import load_results_for_language
    >>> context = load_results_for_language("en", "text8", "2025-10-01")
    >>> print(f"PPM: {context['ppm_bpc']:.4f} bpc, {context['ppm_redundancy']:.2%} redundancy")
    PPM: 1.5000 bpc, 68.00% redundancy  # doctest: +SKIP
    """

    gen = ProofGenerator(lang=lang, corpus=corpus, snapshot=snapshot)
    return gen.generate_context()


__all__ = [
    "ProofGenerator",
    "load_results_for_language",
]


