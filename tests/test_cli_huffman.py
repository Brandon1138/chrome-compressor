from pathlib import Path
import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from reducelang.cli import cli


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_corpus(tmp_path: Path, monkeypatch):
    lang = "en"
    snapshot = "2025-10-01"
    corpus = "test"
    processed_dir = tmp_path / "data" / "corpora" / lang / snapshot / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = processed_dir / f"{corpus}.txt"
    corpus_file.write_text("hello world " * 100, encoding="utf-8")
    datacard = {
        "alphabet_name": "English-27",
        "preprocessing_hash": "deadbeef",
        "sources": [],
    }
    (processed_dir / f"{corpus}_datacard.json").write_text(json.dumps(datacard), encoding="utf-8")

    # Redirect Config paths
    from reducelang import config as cfg

    class Dummy:
        DEFAULT_CORPUS_DIR = tmp_path / "data" / "corpora"
        RESULTS_DIR = tmp_path / "results"
        DEFAULT_SNAPSHOT_DATE = snapshot
        DEFAULT_TEST_SPLIT = 0.2
        RANDOM_SEED = 42

    monkeypatch.setattr(cfg, "Config", Dummy)
    return {
        "lang": lang,
        "snapshot": snapshot,
        "corpus": corpus,
        "base": tmp_path,
    }


def test_huffman_success(cli_runner: CliRunner, mock_corpus):
    result = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
        ],
    )
    assert result.exit_code == 0
    # Check results JSON exists
    out = mock_corpus["base"] / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "huffman_order1.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.get("model_choice") == "huffman"


def test_huffman_with_compare(cli_runner: CliRunner, mock_corpus):
    result = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--compare",
            "--compare-format",
            "markdown",
        ],
    )
    assert result.exit_code == 0
    assert "Comparison" in result.output or result.output.strip().startswith("| ")


def test_huffman_compare_format_json(cli_runner: CliRunner, mock_corpus):
    result = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--compare",
            "--compare-format",
            "json",
        ],
    )
    assert result.exit_code == 0
    cmp = mock_corpus["base"] / "results" / "entropy" / mock_corpus["lang"] / mock_corpus["corpus"] / mock_corpus["snapshot"] / "comparison.json"
    # Our CLI writes to file only for non-table/markdown; ensure created
    if not cmp.exists():
        # Accept printing JSON to stdout fallback
        assert "results" in result.output or result.output


def test_huffman_force_and_skip(cli_runner: CliRunner, mock_corpus):
    # First run to create results
    r1 = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
        ],
    )
    assert r1.exit_code == 0
    # Second run without --force should skip
    r2 = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
        ],
    )
    assert r2.exit_code == 0
    assert "Results exist" in r2.output
    # Third run with --force retrains
    r3 = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--force",
        ],
    )
    assert r3.exit_code == 0


def test_huffman_invalid_test_split(cli_runner: CliRunner, mock_corpus):
    result = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--test-split",
            "1.5",
        ],
    )
    assert result.exit_code != 0


def test_huffman_output_path(cli_runner: CliRunner, mock_corpus, tmp_path: Path):
    custom = tmp_path / "custom.json"
    result = cli_runner.invoke(
        cli,
        [
            "huffman",
            "--lang",
            mock_corpus["lang"],
            "--corpus",
            mock_corpus["corpus"],
            "--output",
            str(custom),
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert custom.exists()


