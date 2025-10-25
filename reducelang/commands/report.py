from __future__ import annotations

from pathlib import Path
from typing import Optional
import click

from reducelang.config import Config
from reducelang.report import generate_report


@click.command(name="report")
@click.option("lang", "--lang", type=click.Choice(["en", "ro", "both"], case_sensitive=False), default="both", show_default=True, help="Language(s) to generate report for")
@click.option("corpus", "--corpus", type=str, required=False, help="Corpus name (default: text8 for en, opus for ro)")
@click.option("snapshot", "--snapshot", type=str, default=Config.DEFAULT_SNAPSHOT_DATE, show_default=True, help="Snapshot date")
@click.option("fmt", "--format", type=click.Choice(["pdf", "html", "both"], case_sensitive=False), default="both", show_default=True, help="Output format")
@click.option("out", "--out", type=click.Path(path_type=Path), default=Path("paper"), show_default=True, help="Output directory for main artifacts (PDF/HTML/MD)")
@click.option("figures_dir", "--figures-dir", type=click.Path(path_type=Path), default=Config.FIGURES_DIR, show_default=True, help="Directory for generated figures (default: paper/figs). For HTML, figures may be copied under OUT/site/figs for portability.")
@click.option("latex_style", "--latex-style", type=click.Choice(["article", "acm", "arxiv"], case_sensitive=False), default="article", show_default=True, help="LaTeX template style")
def report(lang: str, corpus: Optional[str], snapshot: str, fmt: str, out: Path, figures_dir: Path, latex_style: str) -> None:
    """Generate LaTeX/PDF proofs and Markdown/HTML static sites from results.

    Examples:
      reducelang report --lang en --format pdf
      reducelang report --lang ro --format html --out site/
      reducelang report --lang both --format both
      reducelang report --lang en --corpus text8 --format pdf --out paper/
    """

    try:
        out.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        langs: list[str]
        if lang == "both":
            langs = ["en", "ro"]
        else:
            langs = [lang]

        default_map = {"en": "text8", "ro": "opus"}
        for lg in langs:
            corp = corpus if corpus else default_map.get(lg, "text8")
            click.echo(f"Generating report for {lg} ({corp})...")
            paths = generate_report(
                lang=lg,
                corpus=corp,
                snapshot=snapshot,
                output_format=fmt,
                output_dir=out,
                figures_dir=figures_dir,
                latex_style=(None if latex_style.lower() == "article" else latex_style.lower()),
            )
            if paths:
                for kind, p in paths.items():
                    click.echo(f"Generated {kind}: {p}")
            else:
                click.echo("No outputs generated (check tool availability and results presence)")
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        click.secho(
            "Hint: run 'reducelang estimate' and 'reducelang huffman' to create results first.",
            fg="yellow",
        )
        raise SystemExit(1)
    except ValueError as e:
        click.secho(str(e), fg="red", err=True)
        click.secho(
            "Hint: run 'reducelang estimate' and 'reducelang huffman' to create results first.",
            fg="yellow",
        )
        raise SystemExit(1)
    except Exception as e:
        click.secho(f"Report generation failed: {e}", fg="red", err=True)
        raise SystemExit(1)


