from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from reducelang.proofs.renderer import (
    ProofRenderer,
    compile_latex_to_pdf,
    build_static_site,
    _escape_latex_filter,
    _check_tool_available,
)


@pytest.fixture()
def sample_context() -> dict:
    return {
        "language_name": "English",
        "lang": "en",
        "corpus_name": "text8",
        "generation_date": "2025-10-01",
        "alphabet_size": 27,
        "log2_alphabet_size": 4.755,
        "comparison_table_data": [
            {"model": "huffman", "order": 1, "bits_per_char": 4.2, "log2_alphabet_size": 4.755, "redundancy": 0.116, "compression_ratio": 1.132},
            {"model": "ngram", "order": 5, "bits_per_char": 2.5, "log2_alphabet_size": 4.755, "redundancy": 0.474, "compression_ratio": 1.902},
            {"model": "ppm", "order": 8, "bits_per_char": 1.5, "log2_alphabet_size": 4.755, "redundancy": 0.684, "compression_ratio": 3.170},
        ],
        "ppm_bpc": 1.5,
        "ppm_order": 8,
        "ppm_redundancy": 0.684,
        "ppm_ci_width": 0.05,
        "redundancy_ci_width": 0.02,
        "huffman_bpc": 4.2,
        "huffman_redundancy": 0.116,
        "ppm_to_huffman_gain": 57.0,
        "has_sensitivity": False,
        "sensitivity_table_data": [],
        "ablations_list": "",
    }


def test_proof_renderer_init() -> None:
    r = ProofRenderer()
    assert r is not None


def test_render_latex(tmp_path: Path, sample_context: dict) -> None:
    r = ProofRenderer()
    out = tmp_path / "paper.tex"
    r.render_latex(sample_context, out)
    assert out.exists()
    assert "Shannon Redundancy Estimation" in out.read_text(encoding="utf-8")


def test_render_markdown(tmp_path: Path, sample_context: dict) -> None:
    r = ProofRenderer()
    out = tmp_path / "paper.md"
    r.render_markdown(sample_context, out)
    assert out.exists()
    assert "Shannon Redundancy Estimation" in out.read_text(encoding="utf-8")


def test_generate_figures(tmp_path: Path, sample_context: dict) -> None:
    r = ProofRenderer()
    figs = r.generate_figures(sample_context, tmp_path)
    for p in figs.values():
        assert Path(p).exists()


def test_render_latex_acm_style(tmp_path: Path, sample_context: dict) -> None:
    r = ProofRenderer()
    out = tmp_path / "paper_acm.tex"
    # Ensure figure path token exists for rendering
    sample_context_with_figs = dict(sample_context)
    sample_context_with_figs["figs_rel_path_tex"] = "figs"
    r.render_latex(sample_context_with_figs, out, style="acm")
    s = out.read_text(encoding="utf-8")
    assert "\\documentclass[sigconf,review=false,nonacm]{acmart}" in s
    assert "\\bibliographystyle{ACM-Reference-Format}" in s


def test_render_latex_arxiv_style(tmp_path: Path, sample_context: dict) -> None:
    r = ProofRenderer()
    out = tmp_path / "paper_arxiv.tex"
    sample_context_with_figs = dict(sample_context)
    sample_context_with_figs["figs_rel_path_tex"] = "figs"
    r.render_latex(sample_context_with_figs, out, style="arxiv")
    s = out.read_text(encoding="utf-8")
    assert "\\documentclass[11pt]{article}" in s
    assert "\\bibliographystyle{plainnat}" in s


def test_render_latex_fallback_to_main_when_style_missing(tmp_path: Path, sample_context: dict) -> None:
    r = ProofRenderer()
    out = tmp_path / "paper_missingstyle.tex"
    sample_context_with_figs = dict(sample_context)
    sample_context_with_figs["figs_rel_path_tex"] = "figs"
    r.render_latex(sample_context_with_figs, out, style="doesnotexist")
    s = out.read_text(encoding="utf-8")
    # Fallback should produce the generic article template, not acmart
    assert "\\documentclass[11pt]{article}" in s
    assert "acmart" not in s


def test_escape_latex_filter() -> None:
    s = _escape_latex_filter("50% & 1_2{3}#$")
    assert "\\%" in s and "\\&" in s and "\\_" in s and "\\{" in s and "\\#" in s and "\\$" in s


def test_compile_latex_to_pdf_missing_tool(tmp_path: Path, sample_context: dict) -> None:
    # Create minimal tex
    tex = tmp_path / "x.tex"
    tex.write_text("\\documentclass{article}\\begin{document}Hi\\end{document}", encoding="utf-8")
    with patch("reducelang.proofs.renderer._check_tool_available", return_value=False):
        assert compile_latex_to_pdf(tex) is None


def test_build_static_site_missing_tool(tmp_path: Path) -> None:
    md = tmp_path / "x.md"
    md.write_text("# x", encoding="utf-8")
    with patch("reducelang.proofs.renderer._check_tool_available", return_value=False):
        assert build_static_site(md, tmp_path, tool="quarto") is None


