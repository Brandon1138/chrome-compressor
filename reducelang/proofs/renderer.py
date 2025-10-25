"""Rendering utilities for proofs: Jinja2 templates, figure generation, and
optional compilation to PDF/HTML via external tools (LaTeX/Quarto).

This module is designed to fail gracefully when external tools are not
available: PDF/HTML compilation will return None instead of raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import shutil
import subprocess

import jinja2
import matplotlib.pyplot as plt
import numpy as np


_LOGGER = logging.getLogger(__name__)


def _escape_latex_filter(text: str) -> str:
    """Escape LaTeX special characters for safe insertion in templates."""

    if text is None:
        return ""
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def _check_tool_available(tool: str) -> bool:
    return shutil.which(tool) is not None


class ProofRenderer:
    def __init__(self, template_dir: Path | None = None) -> None:
        base = template_dir if template_dir is not None else Path(__file__).parent / "templates"
        self._template_dir = Path(base)
        self._env = jinja2.Environment(loader=jinja2.FileSystemLoader(self._template_dir))
        self._env.filters["escape_latex"] = _escape_latex_filter

    def render_latex(self, context: dict[str, Any], output_path: Path, *, style: str | None = None) -> None:
        template_name = "latex/main.tex.jinja"
        if style:
            candidate = self._template_dir / "latex" / f"{style}.tex.jinja"
            if candidate.exists():
                template_name = f"latex/{style}.tex.jinja"
        template = self._env.get_template(template_name)
        rendered = template.render(**context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        # Copy references.bib next to the LaTeX output
        src_bib = self._template_dir / "latex" / "references.bib"
        dst_bib = output_path.parent / "references.bib"
        try:
            if src_bib.exists():
                dst_bib.write_text(src_bib.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            _LOGGER.warning("Failed to copy references.bib: %s", e)

    def render_markdown(self, context: dict[str, Any], output_path: Path) -> None:
        template = self._env.get_template("markdown/main.md.jinja")
        rendered = template.render(**context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    def generate_figures(self, context: dict[str, Any], output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Figure 1: Entropy vs. order
        fig1_path_pdf = output_dir / f"entropy_vs_order_{context['lang']}.pdf"
        fig1_path_png = output_dir / f"entropy_vs_order_{context['lang']}.png"

        # Prepare data series from comparison_table_data
        table = context.get("comparison_table_data", []) or []
        ngram = sorted([r for r in table if str(r.get("model", "")).startswith("ngram")], key=lambda r: int(r.get("order", 0)))
        ppm = sorted([r for r in table if str(r.get("model", "")).lower() == "ppm"], key=lambda r: int(r.get("order") or r.get("depth") or 0))
        huff = [r for r in table if str(r.get("model", "")).lower() == "huffman"]

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        if ngram:
            n_orders = [int(r.get("order", 0)) for r in ngram]
            n_label = f"N-gram (n={min(n_orders)}..{max(n_orders)})" if n_orders else "N-gram"
            ax.plot(n_orders, [float(r.get("bits_per_char", 0.0)) for r in ngram], marker="o", linestyle="-", label=n_label)
        if ppm:
            p_orders = [int(r.get("order") or r.get("depth") or 0) for r in ppm]
            p_label = f"PPM (d={min(p_orders)}..{max(p_orders)})" if p_orders else "PPM"
            ax.plot(p_orders, [float(r.get("bits_per_char", 0.0)) for r in ppm], marker="s", linestyle="-", label=p_label)
        if huff:
            ax.scatter([1], [float(huff[0].get("bits_per_char", 0.0))], marker="^", label="Huffman")
        ax.axhline(float(context.get("log2_alphabet_size", 0.0)), color="#888888", linestyle="--", label="log2 M")
        ax.set_xlabel("Model order / depth")
        ax.set_ylabel("Bits per character (bpc)")
        ax.set_title(f"Entropy vs. order ({context.get('language_name','')})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig1_path_pdf)
        fig.savefig(fig1_path_png, dpi=200)
        plt.close(fig)

        # Figure 2: Redundancy comparison
        fig2_path_pdf = output_dir / f"redundancy_comparison_{context['lang']}.pdf"
        fig2_path_png = output_dir / f"redundancy_comparison_{context['lang']}.png"

        models: list[str] = []
        redundancies: list[float] = []
        for row in table:
            models.append(str(row.get("model")))
            redundancies.append(float(row.get("redundancy", 0.0)) * 100.0)

        if models:
            fig2, ax2 = plt.subplots(figsize=(7.0, 4.0))
            idx = np.arange(len(models))
            ax2.bar(idx, redundancies, color="#2ca02c")
            ax2.set_xticks(idx, models, rotation=45, ha="right")
            ax2.set_ylabel("Redundancy (%)")
            ax2.set_title(f"Redundancy comparison ({context.get('language_name','')})")
            fig2.tight_layout()
            fig2.savefig(fig2_path_pdf)
            fig2.savefig(fig2_path_png, dpi=200)
            plt.close(fig2)

        return {
            "entropy_vs_order_pdf": fig1_path_pdf,
            "entropy_vs_order_png": fig1_path_png,
            "redundancy_comparison_pdf": fig2_path_pdf,
            "redundancy_comparison_png": fig2_path_png,
        }


def compile_latex_to_pdf(tex_path: Path, output_dir: Path | None = None, engine: str = "pdflatex") -> Path | None:
    """Compile a LaTeX file into PDF using the given engine.

    Returns the PDF path on success, else None.
    """

    if not tex_path.exists():
        _LOGGER.warning("LaTeX source not found: %s", tex_path)
        return None

    if engine == "latexmk":
        tool = "latexmk"
    else:
        tool = engine

    if not _check_tool_available(tool):
        _LOGGER.warning("Tool not available: %s", tool)
        return None

    workdir = tex_path.parent
    out_dir = output_dir or workdir
    try:
        if engine == "latexmk":
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode"]
            if out_dir != workdir:
                cmd.append(f"-outdir={out_dir}")
            cmd.append(tex_path.name)
            subprocess.run(cmd, cwd=workdir, check=True)
        else:
            # Run twice to resolve refs
            base_cmd = [engine, "-interaction=nonstopmode"]
            if out_dir != workdir:
                base_cmd.append(f"-output-directory={out_dir}")
            subprocess.run(base_cmd + [tex_path.name], cwd=workdir, check=True)
            # Bibtex pass (best-effort)
            if (workdir / "references.bib").exists() and _check_tool_available("bibtex"):
                # bibtex expects aux in the output dir when -output-directory is used
                subprocess.run(["bibtex", str(out_dir / tex_path.stem)], cwd=workdir, check=False)
            subprocess.run(base_cmd + [tex_path.name], cwd=workdir, check=True)
        pdf_path = out_dir / f"{tex_path.stem}.pdf"
        return pdf_path if pdf_path.exists() else None
    except Exception as e:
        _LOGGER.warning("LaTeX compilation failed: %s", e)
        return None


def build_static_site(markdown_path: Path, output_dir: Path, tool: str = "quarto") -> Path | None:
    """Build a static HTML from a Markdown file using Quarto or Jupyter Book."""

    if not markdown_path.exists():
        _LOGGER.warning("Markdown source not found: %s", markdown_path)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    tool = tool.lower()
    try:
        if tool == "quarto":
            if not _check_tool_available("quarto"):
                _LOGGER.warning("Quarto not available on PATH.")
                return None
            subprocess.run(["quarto", "render", str(markdown_path), "--output-dir", str(output_dir)], check=True)
            html_path = output_dir / f"{markdown_path.stem}.html"
            return html_path if html_path.exists() else None
        elif tool == "jupyter-book":
            if not _check_tool_available("jupyter-book"):
                _LOGGER.warning("jupyter-book not available on PATH.")
                return None
            # jupyter-book expects a directory; build into output_dir
            subprocess.run(["jupyter-book", "build", str(markdown_path.parent), "--path-output", str(output_dir)], check=True)
            # Best-effort: return an index.html under _build
            candidates = list(output_dir.rglob("*.html"))
            return candidates[0] if candidates else None
        else:
            _LOGGER.warning("Unknown site tool: %s", tool)
            return None
    except Exception as e:
        _LOGGER.warning("Static site build failed: %s", e)
        return None


__all__ = [
    "ProofRenderer",
    "compile_latex_to_pdf",
    "build_static_site",
    "_escape_latex_filter",
    "_check_tool_available",
]


