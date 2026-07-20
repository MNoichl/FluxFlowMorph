#!/usr/bin/env python3
"""Run the complete FlowMorph Klein workflow."""

from __future__ import annotations


def main() -> int:
    from flowmorph_klein.cli import run_command

    return run_command()


if __name__ == "__main__":
    raise SystemExit(main())
