from typing import Any

import pytest

from reducelang.validation.sensitivity import (
    run_sensitivity_analysis,
    run_ablation_study,
    format_sensitivity_results,
)
from reducelang.models import UnigramModel
from reducelang.alphabet import ENGLISH_ALPHABET, ROMANIAN_ALPHABET


@pytest.fixture
def sample_text_small() -> str:
    return ("ana are mere si pere ") * 150


def test_run_sensitivity_analysis_basic(sample_text_small: str):
    """Single variant structure and fields exist."""

    results = run_sensitivity_analysis(
        model_class=UnigramModel,
        base_alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        variants=[{"name": "no_space", "alphabet_kwargs": {"include_space": False}}],
    )
    assert "baseline" in results and "variants" in results
    assert len(results["variants"]) == 1
    v = results["variants"][0]
    for k in [
        "name",
        "alphabet_name",
        "alphabet_size",
        "log2_alphabet_size",
        "bits_per_char",
        "redundancy",
        "delta_bpc",
        "delta_redundancy",
        "relative_change_bpc",
    ]:
        assert k in v


def test_run_sensitivity_analysis_multiple_variants(sample_text_small: str):
    results = run_sensitivity_analysis(
        model_class=UnigramModel,
        base_alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        variants=[
            {"name": "no_space", "alphabet_kwargs": {"include_space": False}},
            {"name": "with_punctuation", "alphabet_kwargs": {"include_punctuation": True}},
        ],
    )
    names = [v["name"] for v in results["variants"]]
    assert "no_space" in names and "with_punctuation" in names


def test_run_ablation_study_basic(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_space"],
    )
    assert len(results["variants"]) == 1


def test_run_ablation_study_romanian_diacritics(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ROMANIAN_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_diacritics"],
    )
    baseline_h = results["baseline"]["bits_per_char"]
    variant_h = results["variants"][0]["bits_per_char"]
    assert pytest.approx(variant_h - baseline_h, rel=1e-3) == results["variants"][0]["delta_bpc"]


def test_run_ablation_study_no_space_increases_entropy(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_space"],
    )
    delta = results["variants"][0]["delta_bpc"]
    # Removing space typically increases entropy or leaves similar for unigram; allow non-strict
    assert delta >= 0.0 or pytest.approx(delta, abs=1e-6) == 0.0


def test_sensitivity_delta_computation(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_space"],
    )
    v = results["variants"][0]
    assert pytest.approx(v["bits_per_char"] - results["baseline"]["bits_per_char"], rel=1e-6) == v["delta_bpc"]


def test_format_sensitivity_results_table(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_space"],
    )
    table = format_sensitivity_results(results, output_format="table")
    assert "Variant" in table and "H (bpc)" in table


def test_format_sensitivity_results_markdown(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_space"],
    )
    md = format_sensitivity_results(results, output_format="markdown")
    assert md.startswith("| ") and "---" in md


def test_format_sensitivity_results_csv(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_space"],
    )
    csv = format_sensitivity_results(results, output_format="csv")
    assert csv.splitlines()[0].startswith("variant,M,log2M,H_bpc,redun")


def test_format_sensitivity_results_json(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=["no_space"],
    )
    js = format_sensitivity_results(results, output_format="json")
    assert isinstance(js, dict) and "baseline" in js and "variants" in js


def test_sensitivity_empty_ablations(sample_text_small: str):
    results = run_ablation_study(
        model_class=UnigramModel,
        alphabet=ENGLISH_ALPHABET,
        train_text=sample_text_small,
        test_text=sample_text_small,
        model_kwargs={},
        ablations=[],
    )
    assert results["variants"] == []


