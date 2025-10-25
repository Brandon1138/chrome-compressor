"""Report orchestration: load results, render templates, generate figures, and
optionally compile to PDF/HTML.

This module provides a high-level interface used by the CLI and notebooks.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any
import logging

from reducelang.config import Config
from reducelang.proofs.generator import load_results_for_language
from reducelang.proofs.renderer import ProofRenderer, compile_latex_to_pdf, build_static_site


_LOGGER = logging.getLogger(__name__)


def generate_report(
    *,
    lang: str,
    corpus: str,
    snapshot: str,
    output_format: str,
    output_dir: Path,
    figures_dir: Path | None = None,
    latex_style: str | None = None,
) -> dict[str, Path]:
    """Generate a report for a single language.

    Parameters
    ----------
    lang: "en" or "ro"
    corpus: e.g., "text8", "opus"
    snapshot: e.g., "2025-10-01"
    output_format: "pdf", "html", or "both"
    output_dir: destination directory for outputs
    """

    context = load_results_for_language(lang, corpus, snapshot)
    renderer = ProofRenderer()

    # Figures
    figs_dir = figures_dir or Config.FIGURES_DIR
    _ = renderer.generate_figures(context, figs_dir)

    outputs: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = output_format.lower()
    if fmt in {"pdf", "both"}:
        tex_path = output_dir / f"{lang}_redundancy.tex"
        # Set LaTeX figure path relative to the LaTeX output directory
        try:
            rel_tex = os.path.relpath(str(figs_dir), str(tex_path.parent))
        except Exception:
            rel_tex = "figs"
        context["figs_rel_path_tex"] = Path(rel_tex).as_posix()
        renderer.render_latex(context, tex_path, style=latex_style)
        pdf_path = compile_latex_to_pdf(tex_path, output_dir=output_dir)
        if pdf_path is not None:
            outputs["pdf"] = pdf_path

    if fmt in {"html", "both"}:
        md_path = output_dir / f"{lang}_redundancy.md"
        # Decide where HTML will live and how to reference figures
        html_out_dir = output_dir if output_dir.name == "site" else (output_dir / "site")
        # If figures are not inside html_out_dir, compute relative path or copy
        try:
            rel_md = os.path.relpath(str(figs_dir), str(html_out_dir))
            context["figs_rel_path_md"] = Path(rel_md).as_posix()
        except Exception:
            # Copy figures into site/figs and point there
            site_figs = html_out_dir / "figs"
            site_figs.mkdir(parents=True, exist_ok=True)
            try:
                from shutil import copy2
                for p in figs_dir.glob("*_*_" + lang + "*.png"):
                    copy2(p, site_figs / p.name)
                for p in figs_dir.glob("*_*_" + lang + "*.pdf"):
                    copy2(p, site_figs / p.name)
            except Exception:
                pass
            context["figs_rel_path_md"] = "figs"
        renderer.render_markdown(context, md_path)
        html_path = build_static_site(md_path, output_dir=html_out_dir)
        if html_path is not None:
            outputs["html"] = html_path

    return outputs


def generate_combined_report(
    *,
    langs: list[str],
    corpus_map: dict[str, str],
    snapshot: str,
    output_format: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate reports for multiple languages.

    Note: For Phase 7 we generate per-language reports independently and return
    a merged mapping of outputs. Combined templating can be added later.
    """

    outputs: dict[str, Path] = {}
    for lg in langs:
        corp = corpus_map.get(lg)
        if not corp:
            _LOGGER.warning("No corpus specified for language: %s", lg)
            continue
        paths = generate_report(
            lang=lg,
            corpus=corp,
            snapshot=snapshot,
            output_format=output_format,
            output_dir=output_dir,
        )
        # Use keys like "pdf_en" to avoid collisions
        for kind, p in paths.items():
            outputs[f"{kind}_{lg}"] = p
    return outputs


__all__ = [
    "generate_report",
    "generate_combined_report",
]


