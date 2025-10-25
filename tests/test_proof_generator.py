import json
from pathlib import Path
import pytest

from reducelang.proofs.generator import ProofGenerator, load_results_for_language
from reducelang.config import Config


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def test_proof_generator_basic() -> None:
    pg = ProofGenerator(lang="en", corpus="text8", snapshot=Config.DEFAULT_SNAPSHOT_DATE)
    assert pg.lang == "en"
    assert pg.corpus == "text8"


def test_generate_context_missing_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point results dir to a non-existing location by overriding Config.RESULTS_DIR via monkeypatching module attribute is tricky.
    # Instead, create an instance with a non-existing snapshot to trigger FileNotFoundError.
    pg = ProofGenerator(lang="en", corpus="text8", snapshot="2099-01-01")
    with pytest.raises(FileNotFoundError):
        pg.generate_context()


def test_find_result_by_model(tmp_path: Path) -> None:
    pg = ProofGenerator(lang="en", corpus="text8", snapshot=Config.DEFAULT_SNAPSHOT_DATE)
    results = [
        {"model_choice": "unigram", "order": 1},
        {"model_choice": "ngram", "order": 5},
        {"model_choice": "ppm", "depth": 8},
    ]
    assert pg._find_result_by_model(results, "ppm", order=8) == results[2]
    assert pg._find_result_by_model(results, "ngram", order=5) == results[1]
    assert pg._find_result_by_model(results, "huffman") is None


def test_extract_bootstrap_ci() -> None:
    from reducelang.alphabet import ENGLISH_ALPHABET

    pg = ProofGenerator(lang="en", corpus="text8", snapshot=Config.DEFAULT_SNAPSHOT_DATE)
    res = {
        "bits_per_char": 1.5,
        "log2_alphabet_size": ENGLISH_ALPHABET.log2_size,
        "bootstrap": {
            "mean_bpc": 1.5,
            "ci_lower_bpc": 1.45,
            "ci_upper_bpc": 1.55,
        },
    }
    ci = pg._extract_bootstrap_ci(res)
    assert ci is not None
    assert pytest.approx(ci["ci_width"], rel=1e-6) == 0.05
    assert "redundancy_ci_width" in ci


def test_generate_context_with_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create fake results
    results_dir = tmp_path / "results" / "entropy" / "en" / "text8" / Config.DEFAULT_SNAPSHOT_DATE
    _write_json(
        results_dir / "huffman_order1.json",
        {"model_choice": "huffman", "order": 1, "bits_per_char": 4.2, "log2_alphabet_size": 4.755, "alphabet_size": 27},
    )
    _write_json(
        results_dir / "ngram_order5.json",
        {"model_choice": "ngram", "order": 5, "bits_per_char": 2.5, "log2_alphabet_size": 4.755, "alphabet_size": 27},
    )
    _write_json(
        results_dir / "ppm_order8.json",
        {
            "model_choice": "ppm",
            "depth": 8,
            "bits_per_char": 1.5,
            "log2_alphabet_size": 4.755,
            "alphabet_size": 27,
            "bootstrap": {"mean_bpc": 1.5, "ci_lower_bpc": 1.45, "ci_upper_bpc": 1.55},
        },
    )

    # Monkeypatch Config.RESULTS_DIR used inside ProofGenerator helper
    from reducelang import redundancy as red_mod
    from reducelang import proofs as proofs_mod

    # Temporarily redirect the default results path by creating symlink-like behavior
    # Here, we set the environment by changing working directory expectations via function override
    orig_config_results = Config.RESULTS_DIR
    try:
        Config.RESULTS_DIR = tmp_path / "results"  # type: ignore[misc]
        ctx = load_results_for_language("en", "text8", Config.DEFAULT_SNAPSHOT_DATE)
        assert ctx["language_name"] == "English"
        assert ctx["ppm_order"] == 8
        assert ctx["ppm_bpc"] == 1.5
        assert ctx["huffman_bpc"] == 4.2
        assert ctx["ngram_best_order"] == 5
        assert ctx["comparison_table_data"]
    finally:
        Config.RESULTS_DIR = orig_config_results  # type: ignore[misc]


def test_load_results_for_language_integration(tmp_path: Path) -> None:
    # Minimal existence check using FileNotFoundError on missing dir
    with pytest.raises(FileNotFoundError):
        load_results_for_language("en", "text8", "2099-01-01")


