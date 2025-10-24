"""CLI command for Huffman coding and redundancy comparison.

Train a Huffman model on a processed corpus, evaluate cross-entropy
in bits/char, save results, and optionally generate comparison tables
across models (unigram, n-gram, PPM, Huffman).

Examples
--------
  reducelang huffman --lang en --corpus text8
  reducelang huffman --lang ro --corpus opus --compare
  reducelang huffman --lang en --corpus text8 --compare --compare-format markdown
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import click

from reducelang.config import Config
from reducelang.alphabet import ENGLISH_ALPHABET, ROMANIAN_ALPHABET
from reducelang.corpus.datacard import load_datacard
from reducelang.huffman import HuffmanModel
from reducelang.redundancy import compute_redundancy, generate_comparison_table
from reducelang.utils import ensure_dir


@click.command(name="huffman")
@click.option(
    "lang",
    "--lang",
    type=click.Choice(["en", "ro"], case_sensitive=False),
    required=True,
    help="Language code",
)
@click.option(
    "corpus",
    "--corpus",
    type=str,
    required=True,
    help="Corpus name (must exist in processed/)",
)
@click.option(
    "snapshot",
    "--snapshot",
    type=str,
    default=Config.DEFAULT_SNAPSHOT_DATE,
    show_default=True,
    help="Snapshot date (YYYY-MM-DD) or 'latest'",
)
@click.option(
    "test_split",
    "--test-split",
    type=float,
    default=Config.DEFAULT_TEST_SPLIT,
    show_default=True,
    help="Fraction of data for testing",
)
@click.option(
    "output",
    "--output",
    type=click.Path(path_type=Path),
    required=False,
    help="Output path for results JSON (default: auto-generated in results/)",
)
@click.option(
    "force",
    "--force",
    is_flag=True,
    help="Re-train even if results exist",
)
@click.option(
    "compare",
    "--compare",
    is_flag=True,
    help="Generate comparison table with all models (unigram, n-gram, PPM, Huffman)",
)
@click.option(
    "compare_format",
    "--compare-format",
    type=click.Choice(["table", "json", "csv", "markdown"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Comparison table format",
)
def huffman(
    lang: str,
    corpus: str,
    snapshot: str,
    test_split: float,
    output: Path | None,
    force: bool,
    compare: bool,
    compare_format: str,
) -> None:
    """Train Huffman model and optionally generate comparison tables."""

    try:
        lang = lang.lower()
        if not (0.0 < test_split < 1.0):
            raise click.ClickException("--test-split must be between 0 and 1 (exclusive).")

        # Locate processed corpus
        processed_dir = Config.DEFAULT_CORPUS_DIR / lang / snapshot / "processed"
        corpus_file = processed_dir / f"{corpus}.txt"
        datacard_file = processed_dir / f"{corpus}_datacard.json"
        if not corpus_file.exists():
            raise click.ClickException(f"Missing corpus file: {corpus_file}")
        if not datacard_file.exists():
            raise click.ClickException(f"Missing datacard file: {datacard_file}")

        # Load corpus and datacard
        text = corpus_file.read_text(encoding="utf-8")
        datacard = load_datacard(datacard_file)
        corpus_hash = datacard.get("preprocessing_hash")
        alphabet_name = datacard.get("alphabet_name")

        # Determine alphabet
        alphabet = ENGLISH_ALPHABET if lang == "en" else ROMANIAN_ALPHABET
        if alphabet.name != alphabet_name:
            raise click.ClickException(
                f"Alphabet mismatch: datacard has {alphabet_name}, code uses {alphabet.name}"
            )

        # Stratified train/test split (same logic style as estimate)
        def _segments_from_datacard() -> list[str]:
            segments: list[str] = []
            sources = datacard.get("sources") or datacard.get("documents")
            if not isinstance(sources, list):
                return []
            for item in sources:
                try:
                    if isinstance(item, dict):
                        if all(k in item for k in ("start", "end")):
                            start = int(item["start"])  # inclusive
                            end = int(item["end"])      # exclusive
                            if 0 <= start < end <= len(text):
                                segments.append(text[start:end])
                            continue
                        if "span" in item and isinstance(item["span"], (list, tuple)) and len(item["span"]) == 2:
                            start = int(item["span"][0])
                            end = int(item["span"][1])
                            if 0 <= start < end <= len(text):
                                segments.append(text[start:end])
                            continue
                        file_key = item.get("file") or item.get("path")
                        if isinstance(file_key, str):
                            p = Path(file_key)
                            if not p.is_absolute():
                                p = processed_dir / p
                            if p.exists() and p.is_file():
                                segments.append(p.read_text(encoding="utf-8"))
                                continue
                except Exception:
                    continue
            return segments

        def _segments_from_processed() -> list[str]:
            candidates: list[Path] = []
            candidates.extend(sorted(processed_dir.glob(f"{corpus}_*.txt")))
            subdir = processed_dir / corpus
            if subdir.exists() and subdir.is_dir():
                candidates.extend(sorted(subdir.rglob("*.txt")))
            candidates = [p for p in candidates if p.name != f"{corpus}.txt"]
            segments: list[str] = []
            for p in candidates:
                try:
                    segments.append(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
            return segments

        raw_segments = _segments_from_datacard()
        if not raw_segments:
            raw_segments = _segments_from_processed()
        if not raw_segments:
            raw_segments = [text]

        train_parts: list[str] = []
        test_parts: list[str] = []
        for seg in raw_segments:
            if not seg:
                continue
            idx = int(len(seg) * (1 - test_split))
            idx = max(0, min(idx, len(seg)))
            train_parts.append(seg[:idx])
            test_parts.append(seg[idx:])

        train_text = "".join(train_parts)
        test_text = "".join(test_parts)

        if len(train_text) == 0 or len(test_text) == 0:
            raise click.ClickException(
                "After stratified splitting, one of the splits is empty. The corpus may be too small for the chosen --test-split. Try decreasing --test-split or using a larger corpus."
            )

        click.echo(f"Sources: {len(raw_segments)} | Train: {len(train_text)} chars, Test: {len(test_text)} chars")

        # Output path handling
        if output is None:
            output_dir = Config.RESULTS_DIR / "entropy" / lang / corpus / snapshot
            output_file = output_dir / "huffman_order1.json"
        else:
            output_dir = output.parent
            output_file = output

        if output_file.exists() and not force:
            click.echo(f"Results exist: {output_file}. Use --force to re-train.")
            try:
                parsed = json.loads(output_file.read_text(encoding="utf-8"))
                click.echo(json.dumps(parsed, indent=2))
            except Exception:
                pass
            # Optionally generate comparison table
            if compare:
                click.echo("\nGenerating comparison table...")
                table = generate_comparison_table(lang, corpus, snapshot, output_format=compare_format.lower())
                if compare_format.lower() in {"table", "markdown"}:
                    click.echo("\n" + (table if isinstance(table, str) else json.dumps(table, indent=2)))
                else:
                    comparison_file = output_dir / f"comparison.{compare_format.lower()}"
                    if isinstance(table, dict):
                        comparison_file.write_text(json.dumps(table, indent=2), encoding="utf-8")
                    else:
                        comparison_file.write_text(str(table), encoding="utf-8")
                    click.echo(f"Comparison table saved: {comparison_file}")
            return

        # Train Huffman
        lm = HuffmanModel(alphabet)
        click.echo(f"Training Huffman on {lang}/{corpus}...")
        lm.fit(train_text)

        # Evaluate
        click.echo("Evaluating on test set...")
        bits_per_char = lm.evaluate(test_text)
        click.echo(f"Average code length: {bits_per_char:.4f} bits/char")

        # Redundancy
        redundancy = compute_redundancy(bits_per_char, alphabet.log2_size)
        click.echo(f"Redundancy: {redundancy:.2%}")
        click.echo(f"Maximum entropy (log₂M): {alphabet.log2_size:.4f} bits/char")

        # Save model
        model_dir = Config.RESULTS_DIR / "models" / lang / corpus / snapshot
        model_file = model_dir / "huffman_order1.pkl"
        ensure_dir(model_dir)
        lm.save(model_file)
        click.echo(f"Model saved: {model_file}")

        # Save results JSON
        ensure_dir(output_dir)
        results = {
            "model_choice": "huffman",
            "order": 1,
            "language": lang,
            "corpus": corpus,
            "snapshot": snapshot,
            "alphabet_name": alphabet.name,
            "alphabet_size": alphabet.size,
            "log2_alphabet_size": alphabet.log2_size,
            "bits_per_char": bits_per_char,
            "redundancy": redundancy,
            "train_chars": len(train_text),
            "test_chars": len(test_text),
            "test_split": test_split,
            "seed": Config.RANDOM_SEED,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_file),
            "corpus_hash": corpus_hash,
        }
        results.update(lm.to_dict())

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        click.echo(f"Results saved: {output_file}")

        # Optional comparison table
        if compare:
            click.echo("\nGenerating comparison table...")
            table = generate_comparison_table(lang, corpus, snapshot, output_format=compare_format.lower())
            if compare_format.lower() in {"table", "markdown"}:
                click.echo("\n" + (table if isinstance(table, str) else json.dumps(table, indent=2)))
            else:
                comparison_file = output_dir / f"comparison.{compare_format.lower()}"
                if isinstance(table, dict):
                    comparison_file.write_text(json.dumps(table, indent=2), encoding="utf-8")
                else:
                    comparison_file.write_text(str(table), encoding="utf-8")
                click.echo(f"Comparison table saved: {comparison_file}")

        click.secho(
            f"OK: Huffman on {lang}/{corpus}: {bits_per_char:.4f} bits/char, {redundancy:.2%} redundancy",
            fg="green",
        )
        if compare:
            click.echo("\nComparison shows Huffman (single-char) vs. context models (n-gram, PPM):")
            click.echo("  - Huffman captures ~10-15% redundancy from character frequencies")
            click.echo("  - N-gram/PPM capture ~60-75% redundancy from dependencies")
    except click.ClickException:
        raise
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)



