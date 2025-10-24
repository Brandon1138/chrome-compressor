"""Command-line interface for reducelang using Click command groups."""

from __future__ import annotations

from typing import NoReturn
import click
from reducelang import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """reducelang: Shannon redundancy estimation for natural languages."""
    pass


# Register subcommands
from reducelang.commands.prep import prep  # noqa: E402

cli.add_command(prep)


def main() -> NoReturn:
    """Entry point for the CLI."""
    cli()


