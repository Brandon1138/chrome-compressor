from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from reducelang.commands.prep import prep


def test_prep_single_corpus(tmp_path: Path):
    runner = CliRunner()
    raw = tmp_path / "raw.zip"
    out_file = tmp_path / "out.txt"
    meta = {"char_count": 5, "unique_chars": {"a"}, "coverage": 1.0, "raw_size_bytes": 1, "processed_size_bytes": 5, "preprocessing_hash": "sha256:00"}

    with patch("reducelang.commands.prep.download_corpus", return_value=raw) as mock_dl, \
         patch("reducelang.commands.prep.preprocess_corpus", return_value=meta) as mock_pre,
         patch("reducelang.commands.prep.get_corpus_spec") as mock_spec,
         patch("reducelang.commands.prep.get_extractor") as mock_get_ex,
         patch("reducelang.commands.prep.generate_datacard") as mock_card:

        class _Spec:
            name = "text8"
            url = "http://example.com/text8.zip"
            format = "text_zip"
            license = "Public Domain"
            description = ""
            sha256 = None
            extractor_class = "ZipTextExtractor"

        mock_spec.return_value = _Spec()
        mock_get_ex.return_value = object()

        result = runner.invoke(prep, ["--lang", "en", "--corpus", "text8", "--snapshot", "latest"])
        assert result.exit_code == 0


def test_prep_with_user_provided_corpus_path(tmp_path: Path):
    runner = CliRunner()
    # Create a local text file to serve as user-provided corpus
    local = tmp_path / "my.txt"
    local.write_text("Hello WORLD!", encoding="utf-8")

    meta = {
        "char_count": 12,
        "unique_chars": {"h", "e", "l", "o", " ", "w", "r", "d", "!"},
        "coverage": 1.0,
        "raw_size_bytes": 12,
        "processed_size_bytes": 12,
        "preprocessing_hash": "sha256:00",
        "alphabet_name": "English-27",
        "raw_sha256": None,
    }

    with patch("reducelang.commands.prep.get_extractor") as mock_get_ex, \
         patch("reducelang.commands.prep.preprocess_corpus", return_value=meta) as mock_pre, \
         patch("reducelang.commands.prep.generate_datacard") as mock_card:

        class _Extractor:
            pass

        mock_get_ex.return_value = _Extractor()

        result = runner.invoke(
            prep,
            [
                "--lang",
                "en",
                "--corpus",
                "corpus",
                "--corpus-path",
                str(local),
            ],
        )
        assert result.exit_code == 0
        # Ensure preprocess called and datacard generated
        assert mock_pre.called
        assert mock_card.called



