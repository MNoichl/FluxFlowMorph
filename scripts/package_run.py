#!/usr/bin/env python3
"""Build and validate a compact FlowMorph Klein archive."""

from __future__ import annotations


def main() -> int:
    from flowmorph_klein.cli import package_run_command

    return package_run_command()


if __name__ == "__main__":
    raise SystemExit(main())
