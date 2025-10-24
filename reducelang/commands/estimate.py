"""Train and evaluate language models for entropy estimation.

Examples
--------
  reducelang estimate --model unigram --lang en --corpus text8
  reducelang estimate --model ngram --order 5 --lang ro --corpus opus
  reducelang estimate --model ngram --order 8 --lang en --corpus text8 --force
  reducelang estimate --model ppm --order 8 --lang en --corpus text8 --bootstrap
  reducelang estimate --model ppm --order 8 --lang ro --corpus opus --sensitivity
  reducelang estimate --model ngram --order 5 --lang ro --corpus opus --bootstrap --sensitivity --ablations no_diacritics,no_space
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from itertools import chain
import json
import gzip
import zipfile
import tarfile
import bz2

import click

from reducelang.config import Config
from reducelang.alphabet import ENGLISH_ALPHABET, ROMANIAN_ALPHABET
from reducelang.corpus.datacard import load_datacard
from reducelang.models import UnigramModel, NGramModel, PPMModel
from reducelang.coding import verify_codelength
from reducelang.utils import ensure_dir


@click.command(name="estimate")
@click.option(
    "model",
    "--model",
    type=click.Choice(["unigram", "ngram", "ppm"], case_sensitive=False),
    required=True,
    help="Model type (unigram, ngram with Kneser-Ney, or ppm)",
)
@click.option(
    "order",
    "--order",
    type=int,
    default=Config.DEFAULT_NGRAM_ORDER,
    show_default=True,
    help="Model order: 1 for unigram, 2-8 for ngram, depth for ppm",
)
@click.option(
    "escape_method",
    "--escape-method",
    type=click.Choice(["A", "B", "C", "D"], case_sensitive=False),
    default="A",
    show_default=True,
    help="PPM escape method (A=default, B/C/D=experimental)",
)
@click.option(
    "update_exclusion",
    "--update-exclusion",
    is_flag=True,
    help="Use update exclusion (PPM-C style)",
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
@click.option(
    "verify_codelength_flag",
    "--verify-codelength",
    is_flag=True,
    help="Verify that codelength matches cross-entropy (PPM only)",
)
@click.option(
    "bootstrap_flag",
    "--bootstrap",
    is_flag=True,
    help="Compute bootstrap confidence intervals (1000 resamples, 95% CI)",
)
@click.option(
    "sensitivity_flag",
    "--sensitivity",
    is_flag=True,
    help="Run sensitivity analysis with alphabet variants",
)
@click.option(
    "ablations",
    "--ablations",
    type=str,
    default="",
    help="Comma-separated list of ablations (no_space,no_diacritics,with_punctuation)",
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
    escape_method: str,
    update_exclusion: bool,
    verify_codelength_flag: bool,
    bootstrap_flag: bool,
    sensitivity_flag: bool,
    ablations: str,
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
        if model == "ppm":
            if not (1 <= order <= 12):
                raise click.ClickException(
                    "For ppm, --order (depth) must be between 1 and 12 inclusive."
                )
            if order < 2:
                click.secho(
                    "Warning: PPM with depth < 2 is very shallow; consider --model unigram",
                    fg="yellow",
                )
            # Warn and normalize unsupported escape methods to A at runtime
            if escape_method.upper() != "A":
                click.secho(
                    f"Warning: PPM escape method {escape_method.upper()} is not implemented; using A.",
                    fg="yellow",
                )
                # Keep user's requested method so the model can record it; model enforces A
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
        elif model == "ppm":
            if not (1 <= order <= 12):
                raise click.ClickException(
                    "For ppm, --order (depth) must be between 1 and 12 inclusive."
                )
            lm = PPMModel(
                alphabet,
                depth=order,
                escape_method=escape_method.upper(),
                update_exclusion=update_exclusion,
            )
        else:
            raise click.ClickException(f"Unsupported model: {model}")

        click.echo(f"Training {model} (order={order}) on {lang}/{corpus}...")
        lm.fit(train_text)

        if bootstrap_flag and sensitivity_flag:
            click.secho("Warning: Both --bootstrap and --sensitivity are enabled. This may take a long time.", fg="yellow")

        click.echo("Evaluating on test set...")
        bits_per_char = lm.evaluate(test_text)
        click.echo(f"Cross-entropy: {bits_per_char:.4f} bits/char")

        # Per-corpus (per-segment) evaluation and meta-analytic means
        per_corpus: list[dict[str, Any]] = []
        try:
            total_chars = 0
            weighted_sum = 0.0
            per_segment_bpcs: list[float] = []
            for i, seg_test in enumerate(test_parts):
                if not seg_test:
                    continue
                h_i = float(lm.evaluate(seg_test))
                n_i = int(len(seg_test))
                per_corpus.append({
                    "source_id": i,
                    "test_chars": n_i,
                    "bits_per_char": h_i,
                })
                total_chars += n_i
                weighted_sum += h_i * n_i
                per_segment_bpcs.append(h_i)

            if per_corpus:
                unweighted_mean = float(sum(per_segment_bpcs) / len(per_segment_bpcs))
                weighted_mean = float(weighted_sum / total_chars) if total_chars > 0 else float('nan')
            else:
                unweighted_mean = float('nan')
                weighted_mean = float('nan')
        except Exception:
            per_corpus = []
            unweighted_mean = float('nan')
            weighted_mean = float('nan')

        # Codelength verification (PPM only, optional)
        codelength_verification: dict[str, Any] | None = None
        if model == "ppm" and verify_codelength_flag:
            click.echo("Verifying codelength matches cross-entropy...")
            try:
                codelength_verification = verify_codelength(test_text, lm, tolerance=1e-3)
                click.echo(f"  Cross-entropy: {codelength_verification['cross_entropy_bpc']:.6f} bpc")
                click.echo(f"  Codelength:    {codelength_verification['codelength_bpc']:.6f} bpc")
                click.echo(f"  Delta:         {codelength_verification['delta_bpc']:.6f} bpc")
                if codelength_verification.get("matches"):
                    click.secho("  ✓ Verification passed (delta < 0.001 bpc)", fg="green")
                else:
                    click.secho("  ✗ Verification failed (delta >= 0.001 bpc)", fg="yellow")
            except Exception as _e:
                click.secho("  ⚠ Verification failed due to an unexpected error", fg="yellow")

        # Bootstrap confidence intervals (optional)
        bootstrap_results: dict[str, Any] | None = None
        if bootstrap_flag:
            from reducelang.validation import block_bootstrap, compute_bootstrap_ci
            click.echo("Computing bootstrap confidence intervals...")
            try:
                bootstrap_results = block_bootstrap(
                    text=test_text,
                    model=lm,
                    block_size=Config.BOOTSTRAP_BLOCK_SIZE,
                    n_resamples=Config.BOOTSTRAP_N_RESAMPLES,
                    confidence_level=Config.BOOTSTRAP_CONFIDENCE_LEVEL,
                    seed=Config.RANDOM_SEED,
                )
                click.echo(f"  Mean: {bootstrap_results['mean_bpc']:.4f} bpc")
                click.echo(
                    f"  95% CI: [{bootstrap_results['ci_lower_bpc']:.4f}, {bootstrap_results['ci_upper_bpc']:.4f}]"
                )

                # Compute redundancy CI
                redundancy_ci = compute_bootstrap_ci(
                    bits_per_char=bits_per_char,
                    log2_alphabet_size=alphabet.log2_size,
                    bootstrap_results=bootstrap_results,
                )
                bootstrap_results["redundancy_ci"] = redundancy_ci
                click.echo(f"  Redundancy: {redundancy_ci['redundancy']:.2%}")
                click.echo(
                    f"  95% CI: [{redundancy_ci['ci_lower_redundancy']:.2%}, {redundancy_ci['ci_upper_redundancy']:.2%}]"
                )

                # Optional: per-segment bootstrap CIs
                per_segment_bootstrap: list[dict[str, Any]] = []
                for i, seg_test in enumerate(test_parts):
                    if not seg_test:
                        continue
                    try:
                        seg_boot = block_bootstrap(
                            text=seg_test,
                            model=lm,
                            block_size=Config.BOOTSTRAP_BLOCK_SIZE,
                            n_resamples=Config.BOOTSTRAP_N_RESAMPLES,
                            confidence_level=Config.BOOTSTRAP_CONFIDENCE_LEVEL,
                            seed=Config.RANDOM_SEED,
                        )
                        per_segment_bootstrap.append({
                            "source_id": i,
                            **seg_boot,
                        })
                    except Exception:
                        continue
                if per_segment_bootstrap and per_corpus:
                    # Attach per-segment bootstrap to matching per_corpus entries
                    seg_boot_map = {b.get("source_id"): b for b in per_segment_bootstrap}
                    for item in per_corpus:
                        sid = item.get("source_id")
                        if sid in seg_boot_map:
                            item["bootstrap"] = seg_boot_map[sid]
            except Exception as e:
                click.secho(f"  ⚠ Bootstrap failed: {e}", fg="yellow")
                bootstrap_results = None

        # Sensitivity analysis (optional)
        sensitivity_results: dict[str, Any] | None = None
        if sensitivity_flag or ablations:
            from reducelang.validation import run_ablation_study, format_sensitivity_results
            click.echo("Running sensitivity analysis...")
            try:
                # Parse ablations
                ablation_list = [a.strip() for a in ablations.split(",") if a.strip()] if ablations else []

                # Default ablations if none specified
                if not ablation_list:
                    if lang == "ro":
                        ablation_list = ["no_diacritics", "no_space"]
                    else:
                        ablation_list = ["no_space"]

                # Load raw pre-normalized text segments from raw/ directory if available
                base_dir = Config.DEFAULT_CORPUS_DIR / lang / snapshot
                raw_dir = base_dir / "raw" / corpus

                def _read_raw_file(p: Path) -> str:
                    try:
                        name = p.name.lower()
                        if name.endswith(".gz") and not name.endswith(".tar.gz") and not name.endswith(".tgz"):
                            with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                                return f.read()
                        if name.endswith(".zip"):
                            texts: list[str] = []
                            with zipfile.ZipFile(p) as zf:
                                for info in zf.infolist():
                                    if info.is_dir():
                                        continue
                                    with zf.open(info) as zf_f:
                                        data = zf_f.read()
                                        try:
                                            texts.append(data.decode("utf-8"))
                                        except UnicodeDecodeError:
                                            texts.append(data.decode("latin-1", errors="ignore"))
                            return "\n".join(texts)
                        if name.endswith(".tar.gz") or name.endswith(".tgz"):
                            texts: list[str] = []
                            with tarfile.open(p, "r:gz") as tf:
                                for m in tf.getmembers():
                                    if m.isfile():
                                        fobj = tf.extractfile(m)
                                        if fobj is None:
                                            continue
                                        try:
                                            texts.append(fobj.read().decode("utf-8", errors="ignore"))
                                        except Exception:
                                            continue
                            return "\n".join(texts)
                        if name.endswith(".bz2"):
                            try:
                                with bz2.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                                    return f.read()
                            except Exception:
                                return p.read_text(encoding="utf-8", errors="ignore")
                        # default: treat as text
                        return p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        return ""

                raw_source_segments: list[str] = []
                if raw_dir.exists() and raw_dir.is_dir():
                    candidates = sorted([x for x in raw_dir.rglob("*") if x.is_file() and not x.name.endswith(".sha256")])
                    if candidates:
                        for p in candidates:
                            content = _read_raw_file(p)
                            if content:
                                raw_source_segments.append(content)

                # Fall back to processed text if raw not available
                if not raw_source_segments:
                    try:
                        raw_text_fallback = corpus_file.read_text(encoding="utf-8")
                    except Exception:
                        raw_text_fallback = text
                    raw_split_idx = int(len(raw_text_fallback) * (1 - test_split))
                    raw_train, raw_test = raw_text_fallback[:raw_split_idx], raw_text_fallback[raw_split_idx:]
                else:
                    # Build train/test using the same source-stratified strategy as baseline
                    raw_train_parts: list[str] = []
                    raw_test_parts: list[str] = []
                    for seg in raw_source_segments:
                        if not seg:
                            continue
                        idx = int(len(seg) * (1 - test_split))
                        idx = max(0, min(idx, len(seg)))
                        raw_train_parts.append(seg[:idx])
                        raw_test_parts.append(seg[idx:])
                    raw_train = "".join(raw_train_parts)
                    raw_test = "".join(raw_test_parts)

                # Model kwargs per model family
                if model == "ngram":
                    model_kwargs: dict[str, Any] = {"order": order}
                    model_class = NGramModel
                elif model == "ppm":
                    model_kwargs = {
                        "depth": order,
                        "escape_method": escape_method.upper(),
                        "update_exclusion": update_exclusion,
                    }
                    model_class = PPMModel
                else:
                    model_kwargs = {}
                    model_class = UnigramModel

                sensitivity_results = run_ablation_study(
                    model_class=model_class,
                    alphabet=alphabet,
                    train_text=raw_train,
                    test_text=raw_test,
                    model_kwargs=model_kwargs,
                    ablations=ablation_list,
                )

                click.echo(f"  Baseline: {sensitivity_results['baseline']['bits_per_char']:.4f} bpc")
                for variant in sensitivity_results.get("variants", []):
                    click.echo(
                        f"  {variant['name']}: {variant['bits_per_char']:.4f} bpc (Δ={variant['delta_bpc']:+.4f}, ΔR={variant['delta_redundancy']:+.2%})"
                    )

                # Save formatted table
                sensitivity_table = format_sensitivity_results(sensitivity_results, output_format="markdown")
                sensitivity_file = output_dir / f"{model}_order{order}_sensitivity.md"
                ensure_dir(sensitivity_file.parent)
                sensitivity_file.write_text(sensitivity_table, encoding="utf-8")
                # Also write CSV artifact
                sensitivity_csv = format_sensitivity_results(sensitivity_results, output_format="csv")
                sensitivity_csv_file = output_dir / f"{model}_order{order}_sensitivity.csv"
                ensure_dir(sensitivity_csv_file.parent)
                sensitivity_csv_file.write_text(sensitivity_csv, encoding="utf-8")
                click.echo(f"  Sensitivity table saved: {sensitivity_file}")
            except Exception as e:
                click.secho(f"  ⚠ Sensitivity analysis failed: {e}", fg="yellow")
                sensitivity_results = None

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

        # Add per-corpus aggregation keys (backward-compatible additions)
        if per_corpus:
            results["per_corpus"] = per_corpus
            results["per_corpus_unweighted_mean_bpc"] = unweighted_mean
            results["per_corpus_weighted_mean_bpc"] = weighted_mean

        # Include PPM-specific metadata
        if model == "ppm":
            # Use model metadata to reflect effective escape method and requested one
            results["escape_method"] = getattr(lm, "escape_method", "A")
            requested = getattr(lm, "escape_method_requested", results.get("escape_method", "A"))
            results["escape_method_requested"] = requested
            results["update_exclusion"] = getattr(lm, "update_exclusion", update_exclusion)
            if codelength_verification is not None:
                results["codelength_verification"] = codelength_verification
        if bootstrap_results is not None:
            results["bootstrap"] = bootstrap_results
            # Write bootstrap CSV artifact
            try:
                bootstrap_csv_path = output_dir / f"{model}_order{order}_bootstrap.csv"
                ensure_dir(bootstrap_csv_path.parent)
                red = bootstrap_results.get("redundancy_ci", {}) if isinstance(bootstrap_results, dict) else {}
                lines = [
                    ",".join([
                        "mean_bpc",
                        "ci_lower_bpc",
                        "ci_upper_bpc",
                        "redundancy",
                        "ci_lower_redundancy",
                        "ci_upper_redundancy",
                        "n_resamples",
                        "block_size",
                        "confidence_level",
                    ])
                ]
                lines.append(
                    ",".join([
                        f"{float(bootstrap_results.get('mean_bpc', float('nan'))):.6f}",
                        f"{float(bootstrap_results.get('ci_lower_bpc', float('nan'))):.6f}",
                        f"{float(bootstrap_results.get('ci_upper_bpc', float('nan'))):.6f}",
                        f"{float(red.get('redundancy', float('nan'))):.6f}",
                        f"{float(red.get('ci_lower_redundancy', float('nan'))):.6f}",
                        f"{float(red.get('ci_upper_redundancy', float('nan'))):.6f}",
                        str(int(bootstrap_results.get('n_resamples', 0))),
                        str(int(bootstrap_results.get('block_size', 0))),
                        f"{float(bootstrap_results.get('confidence_level', 0.0)):.2f}",
                    ])
                )
                bootstrap_csv_path.write_text("\n".join(lines), encoding="utf-8")
                click.echo(f"Bootstrap CSV saved: {bootstrap_csv_path}")
            except Exception:
                pass

        if sensitivity_results is not None:
            results["sensitivity"] = sensitivity_results

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


