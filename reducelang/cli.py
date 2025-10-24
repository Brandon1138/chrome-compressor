"""Command-line interface for reducelang.

Provides a minimal entry point that prints the package version.
"""

from __future__ import annotations

import sys
from typing import NoReturn

try:
    from reducelang import __version__
except Exception:  # pragma: no cover - fallback if import side-effects fail
    __version__ = "0.0.0"


def main() -> NoReturn:
    """CLI entry point.

    - Prints the package version by default.
    - Supports the conventional "--version" flag.
    """

    argv = sys.argv[1:]
    if any(arg in ("--version", "-V", "-v") for arg in argv):
        print(__version__)
        raise SystemExit(0)

    # Default behavior: print version and exit.
    print(__version__)
    raise SystemExit(0)


