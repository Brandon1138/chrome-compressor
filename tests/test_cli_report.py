from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch
import pytest

from reducelang.commands.report import report


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@patch("reducelang.commands.report.generate_report")
def test_report_success_pdf(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"pdf": tmp_path / "en_redundancy.pdf"}
    result = runner.invoke(report, ["--lang", "en", "--format", "pdf", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "Generating report for en" in result.output


@patch("reducelang.commands.report.generate_report")
def test_report_success_html(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"html": tmp_path / "en_redundancy.html"}
    result = runner.invoke(report, ["--lang", "en", "--format", "html", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "Generated html" in result.output


@patch("reducelang.commands.report.generate_report")
def test_report_threads_latex_style_acm(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"pdf": tmp_path / "en_redundancy.pdf"}
    result = runner.invoke(
        report,
        ["--lang", "en", "--format", "pdf", "--out", str(tmp_path), "--latex-style", "acm"],
    )
    assert result.exit_code == 0
    # Ensure CLI passes None for article, or style name for non-default
    kwargs = mock_gen.call_args.kwargs
    assert kwargs.get("latex_style") == "acm"


@patch("reducelang.commands.report.generate_report")
def test_report_threads_latex_style_arxiv(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"pdf": tmp_path / "en_redundancy.pdf"}
    result = runner.invoke(
        report,
        ["--lang", "en", "--format", "pdf", "--out", str(tmp_path), "--latex-style", "arxiv"],
    )
    assert result.exit_code == 0
    kwargs = mock_gen.call_args.kwargs
    assert kwargs.get("latex_style") == "arxiv"


@patch("reducelang.commands.report.generate_report")
def test_report_threads_latex_style_article_as_none(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"pdf": tmp_path / "en_redundancy.pdf"}
    result = runner.invoke(
        report,
        ["--lang", "en", "--format", "pdf", "--out", str(tmp_path), "--latex-style", "article"],
    )
    assert result.exit_code == 0
    kwargs = mock_gen.call_args.kwargs
    assert kwargs.get("latex_style") is None


@patch("reducelang.commands.report.generate_report")
def test_report_success_both(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"pdf": tmp_path / "en_redundancy.pdf", "html": tmp_path / "en_redundancy.html"}
    result = runner.invoke(report, ["--lang", "en", "--format", "both", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "Generated pdf" in result.output and "Generated html" in result.output


@patch("reducelang.commands.report.generate_report")
def test_report_both_languages(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"pdf": tmp_path / "en_redundancy.pdf"}
    result = runner.invoke(report, ["--lang", "both", "--format", "pdf", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert "Generating report for en" in result.output
    assert "Generating report for ro" in result.output


def test_report_invalid_language(runner: CliRunner) -> None:
    # Click will catch invalid choice
    result = runner.invoke(report, ["--lang", "xx", "--format", "pdf"])
    assert result.exit_code != 0


@patch("reducelang.commands.report.generate_report")
def test_report_custom_output_dir(mock_gen, runner: CliRunner, tmp_path: Path) -> None:
    mock_gen.return_value = {"pdf": tmp_path / "custom.pdf"}
    result = runner.invoke(report, ["--lang", "en", "--format", "pdf", "--out", str(tmp_path)])
    assert result.exit_code == 0
    mock_gen.assert_called()


