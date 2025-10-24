from pathlib import Path
from unittest.mock import patch
import json

import pytest
from click.testing import CliRunner

from reducelang.cli import cli
from reducelang.config import Config


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def mock_corpus(tmp_path: Path):
    # Create processed directory structure
    lang = "en"
    snapshot = Config.DEFAULT_SNAPSHOT_DATE
    processed_dir = tmp_path / "data" / "corpora" / lang / snapshot / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    corpus = "test"
    corpus_file = processed_dir / f"{corpus}.txt"
    datacard_file = processed_dir / f"{corpus}_datacard.json"
    text = "hello world this is a test " * 50
    corpus_file.write_text(text, encoding="utf-8")
    datacard = {
        "alphabet_name": "English-27",
        "preprocessing_hash": "deadbeef",
    }
    datacard_file.write_text(json.dumps(datacard), encoding="utf-8")
    return {
        "lang": lang,
        "snapshot": snapshot,
        "corpus": corpus,
        "base": tmp_path,
        "processed_dir": processed_dir,
    }


def test_estimate_unigram_success(cli_runner, mock_corpus, monkeypatch):
    """Unigram estimation should produce a results JSON file and success output."""

    base = mock_corpus["base"]
    # Point Config.DEFAULT_CORPUS_DIR to tmp data directory during test
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)

    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.2345
    ):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "unigram",
                "--lang",
                mock_corpus["lang"],
                "--corpus",
                mock_corpus["corpus"],
                "--snapshot",
                mock_corpus["snapshot"],
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["bits_per_char"] == 1.2345


def test_estimate_ngram_success(cli_runner, mock_corpus, monkeypatch):
    """N-gram estimation should run and write results."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.ngram.NGramModel.fit", return_value=None), patch(
        "reducelang.models.ngram.NGramModel.evaluate", return_value=0.9876
    ):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "ngram",
                "--order",
                "3",
                "--lang",
                mock_corpus["lang"],
                "--corpus",
                mock_corpus["corpus"],
                "--snapshot",
                mock_corpus["snapshot"],
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "ngram_order3.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["bits_per_char"] == 0.9876


def test_estimate_force_flag(cli_runner, mock_corpus, monkeypatch):
    """--force should overwrite existing results."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)

    out_dir = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "unigram_order1.json"
    out_file.write_text("{}", encoding="utf-8")

    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "unigram",
                "--lang",
                mock_corpus["lang"],
                "--corpus",
                mock_corpus["corpus"],
                "--snapshot",
                mock_corpus["snapshot"],
                "--force",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["bits_per_char"] == 1.0


def test_estimate_skip_existing(cli_runner, mock_corpus, monkeypatch):
    """Without --force, should skip training if results exist."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)

    out_dir = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "unigram_order1.json"
    out_file.write_text(json.dumps({"bits_per_char": 9.999}), encoding="utf-8")

    result = cli_runner.invoke(
        cli,
        [
            "estimate",
            "--model",
            "unigram",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--snapshot",
            mock_corpus["snapshot"],
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Output contains the existing JSON
    assert "9.999" in result.output


def test_estimate_invalid_model(cli_runner):
    """Invalid model choice should be caught by Click."""

    result = cli_runner.invoke(
        cli,
        ["estimate", "--model", "invalid", "--lang", "en", "--corpus", "test"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0


def test_estimate_invalid_order(cli_runner, mock_corpus, monkeypatch):
    """ngram with order=1 should error."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    result = cli_runner.invoke(
        cli,
        [
            "estimate",
            "--model",
            "ngram",
            "--order",
            "1",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--snapshot",
            mock_corpus["snapshot"],
        ],
        catch_exceptions=False,
    )
    assert result.exit_code != 0


def test_estimate_missing_corpus(cli_runner, tmp_path: Path, monkeypatch):
    """Missing corpus should raise an error."""

    base = tmp_path
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    result = cli_runner.invoke(
        cli,
        ["estimate", "--model", "unigram", "--lang", "en", "--corpus", "missing"],
        catch_exceptions=False,
    )
    assert result.exit_code != 0


def test_estimate_invalid_test_split(cli_runner, mock_corpus, monkeypatch):
    """Invalid test split should error out."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    result = cli_runner.invoke(
        cli,
        [
            "estimate",
            "--model",
            "unigram",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--snapshot",
            mock_corpus["snapshot"],
            "--test-split",
            "1.5",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code != 0


def test_estimate_output_path(cli_runner, mock_corpus, monkeypatch, tmp_path: Path):
    """Custom output path should be honored."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=2.5
    ):
      
        custom = tmp_path / "custom.json"
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "unigram",
                "--lang",
                mock_corpus["lang"],
                "--corpus",
                mock_corpus["corpus"],
                "--snapshot",
                mock_corpus["snapshot"],
                "--output",
                str(custom),
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert custom.exists()
    data = json.loads(custom.read_text(encoding="utf-8"))
    assert data["bits_per_char"] == 2.5


