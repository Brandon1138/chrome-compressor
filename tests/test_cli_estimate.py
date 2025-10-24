from pathlib import Path
from unittest.mock import patch
import json

import pytest
from click.testing import CliRunner

from reducelang.cli import cli
from reducelang.config import Config
from reducelang.models import PPMModel
from reducelang.coding import verify_codelength as _verify


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


def test_estimate_ppm_success(cli_runner, mock_corpus, monkeypatch):
    """PPM estimation should produce results JSON and success output."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.ppm.PPMModel.fit", return_value=None), patch(
        "reducelang.models.ppm.PPMModel.evaluate", return_value=0.7654
    ):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "ppm",
                "--order",
                "5",
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
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "ppm_order5.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["bits_per_char"] == 0.7654
    assert data.get("escape_method", "A") in {"A", "B", "C", "D"}


def test_estimate_ppm_with_verification(cli_runner, mock_corpus, monkeypatch):
    """PPM with verification flag should print verification details."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.ppm.PPMModel.fit", return_value=None), patch(
        "reducelang.models.ppm.PPMModel.evaluate", return_value=0.7654
    ), patch("reducelang.coding.arithmetic.verify_codelength", return_value={
        "cross_entropy_bpc": 0.7654,
        "codelength_bpc": 0.7654,
        "delta_bpc": 0.0,
        "matches": True,
    }):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "ppm",
                "--order",
                "5",
                "--lang",
                mock_corpus["lang"],
                "--corpus",
                mock_corpus["corpus"],
                "--snapshot",
                mock_corpus["snapshot"],
                "--verify-codelength",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "Verifying codelength" in result.output


def test_estimate_ppm_escape_method(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.ppm.PPMModel.fit", return_value=None), patch(
        "reducelang.models.ppm.PPMModel.evaluate", return_value=0.5
    ):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "ppm",
                "--order",
                "5",
                "--escape-method",
                "D",
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
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "ppm_order5.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    # Effective method is A, but requested is preserved for transparency
    assert data.get("escape_method") == "A"
    assert data.get("escape_method_requested") == "D"


def test_estimate_ppm_update_exclusion(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.ppm.PPMModel.fit", return_value=None), patch(
        "reducelang.models.ppm.PPMModel.evaluate", return_value=0.5
    ):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "ppm",
                "--order",
                "5",
                "--update-exclusion",
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
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "ppm_order5.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data.get("update_exclusion") is True


def test_estimate_ppm_invalid_depth(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    result = cli_runner.invoke(
        cli,
        [
            "estimate",
            "--model",
            "ppm",
            "--order",
            "0",
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


def test_estimate_ppm_depth_warning(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    with patch("reducelang.models.ppm.PPMModel.fit", return_value=None), patch(
        "reducelang.models.ppm.PPMModel.evaluate", return_value=1.0
    ):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "ppm",
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
    assert result.exit_code == 0
    assert "Warning: PPM with depth < 2" in result.output


def test_estimate_ppm_output_structure(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.ppm.PPMModel.fit", return_value=None), patch(
        "reducelang.models.ppm.PPMModel.evaluate", return_value=0.5
    ), patch("reducelang.coding.arithmetic.verify_codelength", return_value={
        "cross_entropy_bpc": 0.5,
        "codelength_bpc": 0.5,
        "delta_bpc": 0.0,
        "matches": True,
    }):
        result = cli_runner.invoke(
            cli,
            [
                "estimate",
                "--model",
                "ppm",
                "--order",
                "5",
                "--lang",
                mock_corpus["lang"],
                "--corpus",
                mock_corpus["corpus"],
                "--snapshot",
                mock_corpus["snapshot"],
                "--verify-codelength",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "ppm_order5.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    for k in [
        "model_choice",
        "order",
        "language",
        "corpus",
        "snapshot",
        "bits_per_char",
        "escape_method",
        "update_exclusion",
    ]:
        assert k in data
    assert "codelength_verification" in data


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


def test_estimate_with_bootstrap(cli_runner, mock_corpus, monkeypatch):
    """CLI with --bootstrap attaches bootstrap results to JSON and prints notice."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)

    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.bootstrap.block_bootstrap", return_value={
        "mean_bpc": 1.0,
        "std_bpc": 0.1,
        "ci_lower_bpc": 0.9,
        "ci_upper_bpc": 1.1,
        "n_resamples": 10,
        "block_size": 2000,
        "confidence_level": 0.95,
    }):
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
                "--bootstrap",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "Computing bootstrap confidence intervals" in result.output
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "bootstrap" in data and "mean_bpc" in data["bootstrap"]


def test_estimate_with_sensitivity(cli_runner, mock_corpus, monkeypatch):
    """CLI with --sensitivity attaches sensitivity results to JSON."""

    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)

    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.sensitivity.run_ablation_study", return_value={
        "baseline": {"bits_per_char": 1.0, "redundancy": 0.5, "alphabet_name": "English-27", "alphabet_size": 27, "log2_alphabet_size": 4.755},
        "variants": [
            {"name": "no_space", "bits_per_char": 1.2, "redundancy": 0.45, "delta_bpc": 0.2, "delta_redundancy": -0.05, "alphabet_name": "English-26", "alphabet_size": 26, "log2_alphabet_size": 4.700},
        ],
    }):
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
                "--sensitivity",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "Running sensitivity analysis" in result.output
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "sensitivity" in data and "baseline" in data["sensitivity"]


def test_estimate_with_both_flags(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.bootstrap.block_bootstrap", return_value={
        "mean_bpc": 1.0,
        "std_bpc": 0.1,
        "ci_lower_bpc": 0.9,
        "ci_upper_bpc": 1.1,
        "n_resamples": 10,
        "block_size": 2000,
        "confidence_level": 0.95,
    }), patch("reducelang.validation.sensitivity.run_ablation_study", return_value={
        "baseline": {"bits_per_char": 1.0, "redundancy": 0.5, "alphabet_name": "English-27", "alphabet_size": 27, "log2_alphabet_size": 4.755},
        "variants": [],
    }):
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
                "--bootstrap",
                "--sensitivity",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert "Warning: Both --bootstrap and --sensitivity" in result.output
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "bootstrap" in data and "sensitivity" in data


def test_estimate_with_ablations(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.sensitivity.run_ablation_study", return_value={
        "baseline": {"bits_per_char": 1.0, "redundancy": 0.5, "alphabet_name": "English-27", "alphabet_size": 27, "log2_alphabet_size": 4.755},
        "variants": [
            {"name": "no_space", "bits_per_char": 1.2, "redundancy": 0.45, "delta_bpc": 0.2, "delta_redundancy": -0.05, "alphabet_name": "English-26", "alphabet_size": 26, "log2_alphabet_size": 4.700},
            {"name": "with_punctuation", "bits_per_char": 0.9, "redundancy": 0.55, "delta_bpc": -0.1, "delta_redundancy": 0.05, "alphabet_name": "English-??", "alphabet_size": 40, "log2_alphabet_size": 5.322},
        ],
    }):
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
                "--sensitivity",
                "--ablations",
                "no_space,with_punctuation",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data.get("sensitivity", {}).get("variants", [])) == 2


def test_estimate_bootstrap_ci_structure(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.bootstrap.block_bootstrap", return_value={
        "mean_bpc": 1.0,
        "std_bpc": 0.1,
        "ci_lower_bpc": 0.9,
        "ci_upper_bpc": 1.1,
        "n_resamples": 10,
        "block_size": 2000,
        "confidence_level": 0.95,
    }):
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
                "--bootstrap",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    b = data.get("bootstrap", {})
    for k in ["mean_bpc", "ci_lower_bpc", "ci_upper_bpc"]:
        assert k in b


def test_estimate_sensitivity_structure(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.sensitivity.run_ablation_study", return_value={
        "baseline": {"bits_per_char": 1.0, "redundancy": 0.5, "alphabet_name": "English-27", "alphabet_size": 27, "log2_alphabet_size": 4.755},
        "variants": [],
    }):
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
                "--sensitivity",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    out_file = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1.json"
    data = json.loads(out_file.read_text(encoding="utf-8"))
    s = data.get("sensitivity", {})
    assert "baseline" in s and "variants" in s


def test_estimate_bootstrap_failure_graceful(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.bootstrap.block_bootstrap", side_effect=RuntimeError("boom")):
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
                "--bootstrap",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert "Bootstrap failed" in result.output


def test_estimate_sensitivity_failure_graceful(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.sensitivity.run_ablation_study", side_effect=RuntimeError("fail")):
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
                "--sensitivity",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert "Sensitivity analysis failed" in result.output


def test_estimate_sensitivity_markdown_output(cli_runner, mock_corpus, monkeypatch):
    base = mock_corpus["base"]
    monkeypatch.setattr(Config, "DEFAULT_CORPUS_DIR", base / "data" / "corpora", raising=False)
    monkeypatch.setattr(Config, "RESULTS_DIR", base / "results", raising=False)
    with patch("reducelang.models.unigram.UnigramModel.fit", return_value=None), patch(
        "reducelang.models.unigram.UnigramModel.evaluate", return_value=1.0
    ), patch("reducelang.validation.sensitivity.run_ablation_study", return_value={
        "baseline": {"bits_per_char": 1.0, "redundancy": 0.5, "alphabet_name": "English-27", "alphabet_size": 27, "log2_alphabet_size": 4.755},
        "variants": [],
    }):
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
                "--sensitivity",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    sens_md = base / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "unigram_order1_sensitivity.md"
    assert sens_md.exists()


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


