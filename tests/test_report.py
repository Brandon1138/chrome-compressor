from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from reducelang.report import generate_report, generate_combined_report


@patch("reducelang.report.load_results_for_language")
@patch("reducelang.report.ProofRenderer")
@patch("reducelang.report.compile_latex_to_pdf")
@patch("reducelang.report.build_static_site")
def test_generate_report_pdf(mock_site, mock_pdf, MockRenderer, mock_load, tmp_path: Path) -> None:
    mock_load.return_value = {"lang": "en", "language_name": "English", "comparison_table_data": [], "log2_alphabet_size": 4.755}
    r = Mock()
    r.generate_figures.return_value = {}
    r.render_latex.return_value = None
    MockRenderer.return_value = r
    mock_pdf.return_value = tmp_path / "en_redundancy.pdf"
    out = generate_report(lang="en", corpus="text8", snapshot="2025-10-01", output_format="pdf", output_dir=tmp_path)
    assert "pdf" in out


@patch("reducelang.report.load_results_for_language")
@patch("reducelang.report.ProofRenderer")
@patch("reducelang.report.compile_latex_to_pdf")
@patch("reducelang.report.build_static_site")
def test_generate_report_html(mock_site, mock_pdf, MockRenderer, mock_load, tmp_path: Path) -> None:
    mock_load.return_value = {"lang": "en", "language_name": "English", "comparison_table_data": [], "log2_alphabet_size": 4.755}
    r = Mock()
    r.generate_figures.return_value = {}
    r.render_markdown.return_value = None
    MockRenderer.return_value = r
    mock_site.return_value = tmp_path / "en_redundancy.html"
    out = generate_report(lang="en", corpus="text8", snapshot="2025-10-01", output_format="html", output_dir=tmp_path)
    assert "html" in out


@patch("reducelang.report.load_results_for_language")
@patch("reducelang.report.ProofRenderer")
@patch("reducelang.report.compile_latex_to_pdf")
@patch("reducelang.report.build_static_site")
def test_generate_report_both(mock_site, mock_pdf, MockRenderer, mock_load, tmp_path: Path) -> None:
    mock_load.return_value = {"lang": "en", "language_name": "English", "comparison_table_data": [], "log2_alphabet_size": 4.755}
    r = Mock()
    r.generate_figures.return_value = {}
    r.render_latex.return_value = None
    r.render_markdown.return_value = None
    MockRenderer.return_value = r
    mock_pdf.return_value = tmp_path / "en_redundancy.pdf"
    mock_site.return_value = tmp_path / "en_redundancy.html"
    out = generate_report(lang="en", corpus="text8", snapshot="2025-10-01", output_format="both", output_dir=tmp_path)
    assert "pdf" in out and "html" in out


def test_generate_report_missing_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        # Will fail in load_results_for_language due to missing results
        generate_report(lang="en", corpus="text8", snapshot="2099-01-01", output_format="pdf", output_dir=tmp_path)


@patch("reducelang.report.generate_report")
def test_generate_combined_report(MockGen, tmp_path: Path) -> None:
    MockGen.return_value = {"pdf": tmp_path / "en_redundancy.pdf"}
    out = generate_combined_report(langs=["en", "ro"], corpus_map={"en": "text8", "ro": "opus"}, snapshot="2025-10-01", output_format="pdf", output_dir=tmp_path)
    assert "pdf_en" in out


