from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from reducelang.config import Config
from reducelang.alphabet import ENGLISH_ALPHABET, ROMANIAN_ALPHABET
from reducelang.corpus.registry import CorpusSpec, get_corpus_spec, list_corpora, build_wiki_url
from reducelang.corpus.downloader import download_corpus
from reducelang.corpus.extractors import get_extractor
from reducelang.corpus.preprocessor import preprocess_corpus
from reducelang.corpus.datacard import generate_datacard
from reducelang.utils import ensure_dir, compute_sha256


@click.command(name="prep")
@click.option("--lang", required=True, type=click.Choice(["en", "ro"]))
@click.option("--corpus", required=True, type=str)
@click.option(
    "--snapshot",
    required=False,
    type=str,
    default=Config.DEFAULT_SNAPSHOT_DATE,
    show_default=True,
    help="Snapshot date (YYYY-MM-DD) or 'latest'",
)
@click.option(
    "--corpus-path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a local UTF-8 text file to use as the corpus. "
        "Skips registry lookup and download; uses a minimal inline spec and PlainTextExtractor."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "Force re-download and re-processing. When used with --corpus-path, forces recopy to raw."
    ),
)
def prep(lang: str, corpus: str, snapshot: str, corpus_path: Optional[Path], force: bool) -> None:
    """Download and preprocess text corpora for entropy estimation.

    Examples:
      reducelang prep --lang en --corpus wikipedia
      reducelang prep --lang ro --corpus opus --snapshot 2025-10-01
      reducelang prep --lang en --corpus all --force
    """

    if corpus == "all" and corpus_path is not None:
        raise click.ClickException("--corpus-path cannot be used with --corpus all")
    if corpus_path is not None and corpus == "all":
        raise click.ClickException("Provide a single corpus name when using --corpus-path")

    # Determine list of corpora
    targets = list_corpora(lang) if corpus == "all" else [(lang, corpus)]

    any_fail = False
    for lg, name in targets:
        click.echo(f"Processing {lg}/{name} (snapshot={snapshot})...")
        try:
            # Build corpus spec: either inline for user-provided path, or from registry
            if corpus_path is not None:
                spec = CorpusSpec(
                    name=name,
                    url=None,
                    format="plain_text",
                    license="User-provided",
                    description="User-provided plain text corpus",
                    sha256=None,
                    extractor_class="PlainTextExtractor",
                )
            else:
                spec = get_corpus_spec(lg, name)

            alphabet = ENGLISH_ALPHABET if lg == "en" else ROMANIAN_ALPHABET

            base_dir = Config.DEFAULT_CORPUS_DIR / lg / snapshot
            raw_dir = base_dir / "raw" / name
            processed_dir = base_dir / "processed"
            ensure_dir(raw_dir)
            ensure_dir(processed_dir)

            # Determine raw filename
            if corpus_path is not None:
                raw_file = raw_dir / (corpus_path.name)
                if force and raw_file.exists():
                    raw_file.unlink()
                # Copy contents
                raw_file.write_bytes(corpus_path.read_bytes())
                download_date = datetime.now(timezone.utc).isoformat()
            else:
                # Possibly override URL for Wikipedia corpora with snapshot-specific URL
                effective_spec_url = spec.url
                if spec.name == "wikipedia" and spec.format == "wikipedia_xml":
                    snap_used, wiki_url = build_wiki_url(lg, snapshot)
                    effective_spec_url = wiki_url
                    snapshot = snap_used

                # Derive filename from URL or name
                filename = (effective_spec_url.rsplit("/", 1)[-1] if effective_spec_url else f"{name}.txt")
                raw_file = raw_dir / filename
                if effective_spec_url is None:
                    # NLTK or gated datasets (handled by extractor); create placeholder file name
                    download_date = datetime.now(timezone.utc).isoformat()
                else:
                    raw_file = download_corpus(effective_spec_url, raw_file, spec.sha256, force=force)
                    download_date = datetime.now(timezone.utc).isoformat()

            processed_file = processed_dir / f"{name}.txt"
            extractor = get_extractor(spec.extractor_class)
            metadata = preprocess_corpus(raw_file, processed_file, alphabet, extractor)
            # Attach additional metadata: alphabet name for reproducibility
            metadata["alphabet_name"] = alphabet.name
            # Compute or read raw SHA256 and attach to metadata (prefixed)
            sidecar = raw_file.with_suffix(raw_file.suffix + ".sha256")
            if sidecar.exists():
                try:
                    raw_digest = sidecar.read_text(encoding="utf-8").strip()
                except Exception:
                    raw_digest = compute_sha256(raw_file)
            else:
                raw_digest = compute_sha256(raw_file)
            metadata["raw_sha256"] = f"sha256:{raw_digest}"
            # Record the final source URL if any (e.g., Wikipedia with a snapshot)
            if corpus_path is not None:
                metadata["source_url"] = None
            else:
                # Use effective_spec_url if defined above, else spec.url
                try:
                    metadata["source_url"] = effective_spec_url  # type: ignore[name-defined]
                except NameError:
                    metadata["source_url"] = spec.url
            datacard_path = processed_dir / f"{name}_datacard.json"
            generate_datacard(
                corpus_spec=spec,
                metadata=metadata,
                output_path=datacard_path,
                snapshot_date=snapshot,
                download_date=download_date,
                language=lg,
            )

            click.secho(
                f"OK: {lg}/{name} -> {processed_file} ({metadata['char_count']} chars)",
                fg="green",
            )
        except Exception as e:
            any_fail = True
            click.secho(f"ERROR processing {lg}/{name}: {e}", fg="red", err=True)

    if any_fail:
        raise SystemExit(1)


