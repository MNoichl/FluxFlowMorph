#!/usr/bin/env python3
"""Run the exact production-shape Colab backward preflight."""

from __future__ import annotations


def main() -> int:
    from flowmorph_klein.cli import validate_colab_command

    return validate_colab_command()


if __name__ == "__main__":
    raise SystemExit(main())
