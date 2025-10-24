from pathlib import Path
import json

import pytest

from reducelang.redundancy import (
    compute_redundancy,
    generate_comparison_table,
    load_model_results,
    analyze_redundancy_gain,
    format_table_ascii,
    format_table_markdown,
    format_table_csv,
)


def test_compute_redundancy_basic():
    r = compute_redundancy(1.5, 4.755)
    assert 0.6 <= r <= 0.8


def test_compute_redundancy_zero():
    r = compute_redundancy(4.755, 4.755)
    assert r == pytest.approx(0.0)


def test_compute_redundancy_one():
    r = compute_redundancy(0.0, 4.755)
    assert r == pytest.approx(1.0)


def test_compute_redundancy_invalid_cap():
    r = compute_redundancy(5.0, 4.755)
    assert r == 0.0


@pytest.fixture()
def mock_results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results" / "entropy" / "en" / "text8" / "2025-10-01"
    d.mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "file": d / "unigram_order1.json",
            "data": {
                "model_choice": "unigram",
                "order": 1,
                "bits_per_char": 4.200,
                "log2_alphabet_size": 4.755,
                "alphabet_size": 27,
            },
        },
        {
            "file": d / "ngram_order5.json",
            "data": {
                "model_choice": "ngram-kn",
                "order": 5,
                "bits_per_char": 2.500,
                "log2_alphabet_size": 4.755,
                "alphabet_size": 27,
            },
        },
        {
            "file": d / "ppm_order8.json",
            "data": {
                "model_choice": "ppm",
                "order": 8,
                "bits_per_char": 1.500,
                "log2_alphabet_size": 4.755,
                "alphabet_size": 27,
            },
        },
        {
            "file": d / "huffman_order1.json",
            "data": {
                "model_choice": "huffman",
                "order": 1,
                "bits_per_char": 4.220,
                "log2_alphabet_size": 4.755,
                "alphabet_size": 27,
            },
        },
    ]
    for s in samples:
        s["file"].write_text(json.dumps(s["data"]), encoding="utf-8")
    return d


def test_load_model_results(mock_results_dir: Path):
    entries = load_model_results(mock_results_dir)
    assert len(entries) == 4


def test_generate_comparison_table_formats(mock_results_dir: Path, monkeypatch):
    # Redirect Config.RESULTS_DIR temporarily
    from reducelang import config as cfg

    class Dummy:
        RESULTS_DIR = mock_results_dir.parent.parent.parent

    monkeypatch.setattr(cfg, "Config", Dummy)
    t_ascii = generate_comparison_table("en", "text8", "2025-10-01", output_format="table")
    assert isinstance(t_ascii, str) and "Model" in t_ascii
    t_md = generate_comparison_table("en", "text8", "2025-10-01", output_format="markdown")
    assert isinstance(t_md, str) and t_md.startswith("| ")
    t_csv = generate_comparison_table("en", "text8", "2025-10-01", output_format="csv")
    assert isinstance(t_csv, str) and t_csv.splitlines()[0].startswith("model,")
    t_json = generate_comparison_table("en", "text8", "2025-10-01", output_format="json")
    assert isinstance(t_json, dict) and "results" in t_json


def test_sorting_order(mock_results_dir: Path, monkeypatch):
    from reducelang import config as cfg

    class Dummy:
        RESULTS_DIR = mock_results_dir.parent.parent.parent

    monkeypatch.setattr(cfg, "Config", Dummy)
    t_json = generate_comparison_table("en", "text8", "2025-10-01", output_format="json")
    results = t_json["results"]
    # Sorted by H descending, so first should be close to unigram/huffman
    assert results[0]["bits_per_char"] >= results[-1]["bits_per_char"]


def test_analyze_redundancy_gain():
    results = [
        {"model": "huffman", "redundancy": 0.12},
        {"model": "ngram-kn", "redundancy": 0.47},
        {"model": "ppm", "redundancy": 0.68},
    ]
    gains = analyze_redundancy_gain(results)
    assert pytest.approx(gains["huffman_to_ngram_gain"], rel=1e-6) == 0.35
    assert pytest.approx(gains["huffman_to_ppm_gain"], rel=1e-6) == 0.56
    assert pytest.approx(gains["ngram_to_ppm_gain"], rel=1e-6) == 0.21


def test_formatters_basic():
    mock = [
        {"model": "huffman", "order": 1, "bits_per_char": 4.2, "log2_alphabet_size": 4.755, "redundancy": 0.12, "compression_ratio": 1.13},
        {"model": "ngram-kn", "order": 5, "bits_per_char": 2.5, "log2_alphabet_size": 4.755, "redundancy": 0.47, "compression_ratio": 1.90},
    ]
    ascii_tbl = format_table_ascii(mock)
    assert "Model" in ascii_tbl and "H (bpc)" in ascii_tbl
    md_tbl = format_table_markdown(mock)
    assert md_tbl.startswith("| Model |")
    csv_tbl = format_table_csv(mock)
    assert csv_tbl.splitlines()[0].startswith("model,")


