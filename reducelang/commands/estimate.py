"""Train and evaluate language models for entropy estimation.

Examples
--------
  reducelang estimate --model unigram --lang en --corpus text8
  reducelang estimate --model ngram --order 5 --lang ro --corpus opus
  reducelang estimate --model ngram --order 8 --lang en --corpus text8 --force
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from itertools import chain
import json

import click

from reducelang.config import Config
from reducelang.alphabet import ENGLISH_ALPHABET, ROMANIAN_ALPHABET
from reducelang.corpus.datacard import load_datacard
from reducelang.models import UnigramModel, NGramModel
from reducelang.utils import ensure_dir


@click.command(name="estimate")
@click.option(
    "model",
    "--model",
    type=click.Choice(["unigram", "ngram"], case_sensitive=False),
    required=True,
    help="Model type (unigram or ngram with Kneser-Ney)",
)
@click.option(
    "order",
    "--order",
    type=int,
    default=Config.DEFAULT_NGRAM_ORDER,
    show_default=True,
    help="N-gram order (1 for unigram, 2-8 for ngram)",
)
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
def estimate(
    model: str,
    order: int,
    lang: str,
    corpus: str,
    snapshot: str,
    test_split: float,
    output: Path | None,
    force: bool,
) -> None:
    """Train and evaluate language models for entropy estimation."""

    try:
        # Normalize choices
        model = model.lower()
        lang = lang.lower()

        # Validate inputs
        if model == "unigram" and order != 1:
            click.secho("Warning: unigram forces order=1; overriding.", fg="yellow")
            order = 1
        if model == "ngram" and not (2 <= order <= 8):
            raise click.ClickException("For ngram, --order must be between 2 and 8 inclusive.")
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

        # Train/test split (source-stratified)
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
                    # Skip malformed entries
                    continue
            return segments

        def _segments_from_processed() -> list[str]:
            candidates: list[Path] = []
            # processed/{corpus}_*.txt
            candidates.extend(sorted(processed_dir.glob(f"{corpus}_*.txt")))
            # processed/{corpus}/**/*.txt
            subdir = processed_dir / corpus
            if subdir.exists() and subdir.is_dir():
                candidates.extend(sorted(subdir.rglob("*.txt")))
            # Exclude the main concatenated file if present
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
            # Fallback to contiguous split if no boundaries found
            raw_segments = [text]

        train_parts: list[str] = []
        test_parts: list[str] = []
        for seg in raw_segments:
            if not seg:
                continue
            idx = int(len(seg) * (1 - test_split))
            # Ensure within bounds
            idx = max(0, min(idx, len(seg)))
            train_parts.append(seg[:idx])
            test_parts.append(seg[idx:])

        train_text = "".join(train_parts)
        test_text = "".join(test_parts)

        # Guard against empty splits
        if len(train_text) == 0 or len(test_text) == 0:
            raise click.ClickException(
                "After stratified splitting, one of the splits is empty. "
                "The corpus may be too small for the chosen --test-split. "
                "Try decreasing --test-split or using a larger corpus."
            )

        click.echo(f"Sources: {len(raw_segments)} | Train: {len(train_text)} chars, Test: {len(test_text)} chars")

        # Output path handling
        if output is None:
            output_dir = Config.RESULTS_DIR / "entropy" / lang / corpus / snapshot
            output_file = output_dir / f"{model}_order{order}.json"
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
            return

        # Instantiate model
        if model == "unigram":
            lm = UnigramModel(alphabet)
        elif model == "ngram":
            # Validate order range for n-gram (2-8)
            if not (2 <= order <= 8):
                raise click.ClickException("For ngram, --order must be between 2 and 8 inclusive.")
            lm = NGramModel(alphabet, order=order)
        else:
            raise click.ClickException(f"Unsupported model: {model}")

        click.echo(f"Training {model} (order={order}) on {lang}/{corpus}...")
        lm.fit(train_text)

        click.echo("Evaluating on test set...")
        bits_per_char = lm.evaluate(test_text)
        click.echo(f"Cross-entropy: {bits_per_char:.4f} bits/char")

        # Save model
        model_dir = Config.RESULTS_DIR / "models" / lang / corpus / snapshot
        model_file = model_dir / f"{model}_order{order}.pkl"
        ensure_dir(model_dir)
        lm.save(model_file)
        click.echo(f"Model saved: {model_file}")

        # Results JSON
        ensure_dir(output_dir)
        results: dict[str, Any] = {
            "model_choice": model,
            "order": order,
            "language": lang,
            "corpus": corpus,
            "snapshot": snapshot,
            "alphabet_name": alphabet.name,
            "alphabet_size": alphabet.size,
            "log2_alphabet_size": alphabet.log2_size,
            "bits_per_char": bits_per_char,
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

        click.secho(
            f"OK: {model} (order={order}) on {lang}/{corpus}: {bits_per_char:.4f} bits/char",
            fg="green",
        )
    except click.ClickException:
        raise
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)


